"""Train migrated AlertBERT short/host MLM on configured AIT-ADS CSV scenes."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import FEATURES, get_sequence, load_split, sequence_slices
from .models import MaskedLangModelParams, MaskedLanguageModel
from .preprocessing import Vocabulary, MaskedLangModelingSequenceCollate, default_collate_fn


def masked_batch(sequence, vocabs, seed):
    collate = MaskedLangModelingSequenceCollate(
        {**vocabs, "raw_time": default_collate_fn}, target_ratio=.2,
        generator=torch.Generator().manual_seed(seed),
    )
    batch = collate([sequence])
    mask_index = batch.mask_index
    if mask_index.shape[1] == 0:
        batch["mask"][0, 0] = True
        mask_index = torch.tensor([[0], [0]])
        for f in FEATURES:
            batch[f + "_mask"][0, 0] = 0
    return batch, mask_index


def forward_loss(model, batch, mask_index, vocabs):
    outputs = model(**{f: batch[f + "_mask"] for f in FEATURES}, raw_time=batch["raw_time"])
    indices = tuple(mask_index)
    losses, counts = [], {}
    for f, logits in zip(FEATURES, outputs):
        targets = vocabs[f].compute_targets(batch[f][indices])
        predictions = logits[indices]
        losses.append(torch.nn.functional.cross_entropy(predictions, targets))
        counts[f] = (int((predictions.argmax(-1) == targets).sum()), targets.numel())
    return torch.stack(losses).mean(), counts


@torch.no_grad()
def evaluate(model, scenes, vocabs, context_size, device, limit=32):
    model.eval()
    result = {}
    for name, scene in scenes.items():
        selections = list(sequence_slices(scene, context_size))
        indices = np.linspace(0, len(selections)-1, min(limit, len(selections)), dtype=int)
        losses, total = [], {f: [0, 0] for f in FEATURES}
        for i in indices:
            batch, mask_index = masked_batch(get_sequence(scene, selections[i]), vocabs, 9000 + int(i))
            batch = batch.to(device)
            mask_index = mask_index.to(device)
            loss, counts = forward_loss(model, batch, mask_index, vocabs)
            losses.append((float(loss), counts[FEATURES[0]][1]))
            for f, (correct, count) in counts.items():
                total[f][0] += correct
                total[f][1] += count
        result[name] = dict(
            evaluated_sequences=len(indices),
            masked_loss=sum(v*n for v,n in losses)/sum(n for _,n in losses),
            masked_accuracy={f: c/n for f,(c,n) in total.items()},
            unknown_rate={f: float(np.mean([v not in vocabs[f].word2idx for v in scene[f]])) for f in FEATURES},
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ait_ads.json")
    parser.add_argument("--output", type=Path, default=Path("results/semantic/aitads_mlm_1k"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.context_size < 1:
        parser.error("steps and context-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with GPU permissions or explicitly select --device cpu")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    scenes = load_split(args.config, "train")
    vocabs = {f: Vocabulary(min_freq=1) for f in FEATURES}
    for f in FEATURES:
        for scene in scenes.values():
            vocabs[f].build_from_iterator(scene[f])
    params = MaskedLangModelParams(
        id_suffix="aitads_csv", augment=None, context_size=args.context_size,
        batch_size=1, n_heads=2, dim_per_head=16, features=FEATURES, targets=FEATURES,
        min_freq=1, save_intervals=((0, args.steps),), lr=.001,
    )
    model = MaskedLanguageModel(params, vocabs).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    samples = [(name, selection) for name, scene in scenes.items()
               for selection in sequence_slices(scene, args.context_size)]
    if not samples:
        raise ValueError("Empty training split")
    args.output.mkdir(parents=True)
    for f, vocab in vocabs.items():
        vocab.save(str(args.output / f"vocab_{f}.json"))
    before = evaluate(model, scenes, vocabs, args.context_size, args.device)
    print(f"Device: {args.device}; scenes: {list(scenes)}; sequences: {len(samples)}", flush=True)
    start = time.monotonic()
    history = []
    seen_alerts = 0
    for step in range(args.steps):
        if step % len(samples) == 0:
            random.shuffle(samples)
        name, selection = samples[step % len(samples)]
        sequence = get_sequence(scenes[name], selection)
        seen_alerts += len(sequence["raw_time"])
        batch, mask_index = masked_batch(sequence, vocabs, 42 + step)
        batch = batch.to(args.device)
        mask_index = mask_index.to(args.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward_loss(model, batch, mask_index, vocabs)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            history.append(dict(step=step+1, loss=float(loss.detach())))
            print(history[-1], flush=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    after = evaluate(model, scenes, vocabs, args.context_size, args.device)
    test_scenes = load_split(args.config, "test")
    test = evaluate(model, test_scenes, vocabs, args.context_size, args.device)
    report = dict(params=params.dict, device=args.device,
                  gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  torch_version=torch.__version__, steps=args.steps,
                  elapsed_seconds=time.monotonic()-start, processed_alerts=seen_alerts,
                  train_alerts={n: len(s["raw_time"]) for n,s in scenes.items()},
                  test_alerts={n: len(s["raw_time"]) for n,s in test_scenes.items()},
                  vocab_sizes={f: len(v) for f,v in vocabs.items()}, history=history,
                  train_diagnostic_before=before, train_diagnostic_after=after, test=test,
                  evaluation="32 evenly spaced sequences per scene; MLM metrics, not attack classification",
                  window_seconds=600, seed=42, optimizer="AdamW", lr=.001, weight_decay=.01)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

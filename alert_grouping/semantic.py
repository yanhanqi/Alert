"""Load an AlertBERT checkpoint and return one contextual vector per input alert."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import FEATURES, sequence_slices, get_sequence
from .models import MaskedLangModelParams, MaskedLanguageModel
from .preprocessing import Vocabulary, BaseSequenceCollate, default_collate_fn


class AlertSemanticEncoder:
    def __init__(self, model_dir, device="cpu"):
        path = Path(model_dir)
        report = json.loads((path / "report.json").read_text())
        params = MaskedLangModelParams(id_suffix="load")
        params.dict = report["params"]
        self.context_size = params["context_size"]
        self.device = device
        vocabs = {f: Vocabulary(params["min_freq"]) for f in FEATURES}
        for f,v in vocabs.items():
            v.load(str(path / f"vocab_{f}.json"))
        self.collate = BaseSequenceCollate({**vocabs, "raw_time": default_collate_fn})
        self.model = MaskedLanguageModel(params, vocabs).to(device)
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
        self.model.eval()
        self.dimension = params["d_model"]

    @torch.no_grad()
    def encode(self, alerts):
        """Pass one scene/window as a JSON array; preserve input order and ignore labels."""
        if not alerts:
            return np.empty((0, self.dimension), dtype=np.float32)
        times = np.array([a.get("raw_time", a.get("time")) for a in alerts], dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError("All alerts require finite timestamps")
        order = np.argsort(times, kind="stable")
        scene = {f: np.array([str(alerts[i][f]) for i in order]) for f in FEATURES}
        scene["raw_time"] = times[order]
        result = np.empty((len(alerts), self.dimension), dtype=np.float32)
        for selection in sequence_slices(scene, self.context_size):
            batch = self.collate([get_sequence(scene, selection)]).to(self.device)
            x = self.model.embedding(**{f: batch[f] for f in FEATURES})
            x = self.model.encoder(x, batch["raw_time"])
            result[order[selection]] = x[0].cpu().numpy()
        return result

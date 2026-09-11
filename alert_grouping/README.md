# Alert grouping semantic module

This root-level package reuses the AlertBERT semantic encoder. It does not
depend on the removed `alertbert` package or its AIT-ADS-A loader.

## Files

| File | Purpose |
| --- | --- |
| `models.py` | Migrated AlertBERT embeddings, time-aware Transformer and MLM head |
| `preprocessing.py` | Migrated categorical vocabularies, time encoding and masking |
| `data.py` | Load AIT-ADS CSV files using the configured scene split |
| `train_mlm.py` | Self-supervised MLM training and sampled diagnostics |
| `semantic.py` | Load a checkpoint and encode a JSON array into vectors |
| `cooccurrence.py` | Count historical window-level alert-type co-occurrences |
| `grouping.py` | Semantic distance, optional co-occurrence fusion and connected components |

## Data and training

`configs/ait_ads.json` defines six training scenes and two test scenes.
Only `short`, `host` and the numeric `time` column are read for training.
Vocabularies are built from training scenes only. No attack or stage label
is supplied to the encoder. Unseen values map to the unknown token.

This encoder learns contextual **categorical** alert representations, not
free-text or full JSON representations. The output dimension is 32 with
the current training defaults (one Transformer layer, two attention heads).

Training chunks stay inside a scene and a fixed 600-second timestamp bucket,
with at most 256 alerts per chunk. These are non-overlapping training chunks,
not the priority pipeline's overlapping deployment windows. Long, dense
windows lose cross-chunk attention. Inference uses the same chunking policy.

From the repository root, using the existing conda environment:

```bash
conda activate alertbert
python -m alert_grouping.train_mlm --steps 1000 --device cuda --output results/semantic/aitads_mlm_new
python -m unittest discover -s tests -v
```

The output directory must not already exist. It contains `model.pt`,
`vocab_short.json`, `vocab_host.json` and `report.json`. Existing weights are
in `results/semantic/aitads_mlm_1k`; do not overwrite them to run a check.

The actual optimizer is AdamW with constant learning rate 0.001 and weight
decay 0.01. The nested legacy `params` also contains unused original trainer
defaults (scheduler, optimizer, warmup); the top-level report and training
loop describe the actual optimization procedure.

## Calling the encoder

```python
from alert_grouping.semantic import AlertSemanticEncoder

encoder = AlertSemanticEncoder("results/semantic/aitads_mlm_1k", device="cuda")
alerts = [{"short": "S-Htt-Mat", "host": "inet-firewall", "time": 1642724061}]
vectors = encoder.encode(alerts)  # shape: (len(alerts), 32)
```

Pass alerts from one scene/current processing window. Output rows preserve
input order. `raw_time`, when supplied, takes precedence over `time`.
The grouping API does not enforce a deployment time horizon; its caller must
restrict inputs to the intended processing window.

## Current evidence and limits

The existing report records 1,000 GPU updates and 169,790 processed alert
positions; this is a pilot, not a full pass over the training data or a
reproduction of all paper experiments. Each diagnostic evaluates only 32
evenly spaced sequences per scene.

| Held-out scene | MLM loss | short accuracy | host accuracy |
| --- | ---: | ---: | ---: |
| wheeler | 0.4985 | 0.8258 | 0.8939 |
| wilson | 0.4776 | 0.8332 | 0.9069 |

These are masked-token prediction metrics, not attack classification or
attack-chain reconstruction metrics. Held-out host unknown rates are 4.07%
and 6.30%, respectively. Good MLM accuracy does not establish good grouping.

`experiments/semantic_grouping_trial.py` is a diagnostic script, not the
deployment entry point. By default it selects a time span from ground-truth
attack annotations; that span can exceed ten minutes. Its purity/AMI output
must not be reported as label-free, full-timeline deployment performance.
The current fusion weight and association threshold are manually supplied;
supervised threshold learning and the third patent module are not implemented
by this semantic training step.

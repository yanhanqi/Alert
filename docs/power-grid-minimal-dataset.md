# Power Grid Minimal Dataset Configuration

`power-grid-minimal` is a lightweight AIT-ADS-A configuration for exploring an electric power grid application scenario with minimal repository changes.

The configuration lives at:

```text
aitads_augmented/configs/power-grid-minimal.json
```

It does not copy, rewrite, or mutate any file in `aitads_augmented/data`. Instead, it reuses existing AIT-ADS-A alert fragments and recombines them to mimic properties that often matter in power-grid and SCADA alert grouping:

- routine background alerts with strong time-of-day structure;
- multiple substations or control-network segments represented by different source scenarios;
- dense bursts during dispatch or control-action windows;
- overlapping attack phases that make pure time-delta grouping ambiguous;
- telemetry or PMU-style exfiltration placed near control-network compromise.

## Label Mapping

The underlying token values remain the original AIT-ADS labels. For scenario design and reporting, the labels are interpreted as follows:

| AIT-ADS fragment | Power-grid interpretation |
| --- | --- |
| `service_scan`, `dns_scan` | substation or control-network reconnaissance |
| `wpscan`, `dirb` | IED, HMI, or gateway asset enumeration |
| `webshell_cmd` | control gateway compromise |
| `crack_passwords`, `online_cracking` | operator account brute force |
| `attacker_change_user` | operator role switch |
| `escalated_sudo_command` | control privilege escalation |
| `dnsteal_start`, `dnsteal_active`, `dnsteal_end` | telemetry, PMU, or historian data exfiltration |

## Scope

This is a configuration-level reconstruction, not a field-level electric-grid dataset rewrite. The model still sees the existing `short`, `host`, and time features from AIT-ADS-A. The purpose is to create a small, reproducible scenario that reflects power-grid timing and concurrency patterns without paying the cost of copying or transforming hundreds of megabytes of JSON data.

For a field-level reconstruction where model-visible alert fields are rewritten with electric power grid terminology, use `power-grid-full`.

## Usage

Load the dataset with the existing factory:

```python
from alertbert.aitads import AITAlertDataset

train = AITAlertDataset(split="train", configuration="power-grid-minimal")
val = AITAlertDataset(split="val", configuration="power-grid-minimal")
test = AITAlertDataset(split="test", configuration="power-grid-minimal")
```

For grouping evaluation, build the ground-truth label vocabularies once:

```bash
conda run -n alertbert python -m alertbert.model_eval_utils power-grid-minimal
```

# Window Priority API

The original experiment is unchanged: `python -m alert_priority.main`.

For one window, pass a parsed JSON array (a Python list of dictionaries):

```python
from alert_priority.main import prioritize_alerts

result = prioritize_alerts(alerts)
ranked = result["ranked"]
high = result["high_priority"]
low = result["low_priority"]
```

Each input alert needs `ip`, `host`, `short`, `name`, `time`, and `raw_time`.
Timestamps must be finite numeric values on a common timeline. Labels are not
required. Supply a unique string `alert_id` to retain identity between calls;
otherwise the API generates IDs local to the input array.

The function merges equal `ip/host/short` alerts within the supplied window,
recomputes the existing 10 features, and fits a 200-tree Isolation Forest.
It does not read files, print metrics, or mutate the input. Results are merged
representatives ordered by descending anomaly score. Every result retains
`member_ids` and `member_indices` for the original alerts, plus its merge time
range and count. Labels in the representative alert are not group ground truth.

The callable defaults to `contamination="auto"`; optionally pass a fixed ratio
in `(0, 0.5]`. It never derives this value from window labels. This deliberately
differs from the original label-informed experiment and supports unlabelled,
noise-only and single-alert windows. Priorities are predictions, not true labels.
Scores and thresholds are window-local. The pipeline averages them as described
below; this is an aggregation rule, not a calibrated attack probability.

## Pipeline

Run from the repository root:

```bash
# All scenarios of the test split, processed separately, for their full duration:
python main.py --output window-priorities.jsonl

# One 30-minute round containing russellmitchell attacks (03:50 +01:00):
python main.py --scenario-index 1 --start-time 1739328600 --max-rounds 1

# A user-provided JSON array with a unified timeline:
python main.py --input alerts.json --max-rounds 1
```

Without `--input`, the pipeline uses the project's AIT-ADS-A configuration loader
(`simul-attacks`, test split by default), including the configured time offsets.
It does not concatenate raw relative-time fragments. Loading into memory is for
offline replay; the model only receives the current window, never the full data.

Every 30-minute round contains 21 windows: `[0,10)`, `[1,11)`, ... `[20,30)`
in minutes. The next round starts at minute 30; windows do not cross rounds.
The default origin is the first alert's timestamp rounded down to 30 minutes.
Empty windows are returned as empty results. The final round is completed on
its nominal timeline even if later windows are empty. Results are available at
window end, so this is not a per-alert immediate decision system.

`run_pipeline(alerts)` preserves the original window-only API. To include final
rankings, call `run_pipeline(alerts, include_aggregates=True)`. The CLI always
enables this option. Its JSONL output has two record types:

- `record_type="window"`: the original window-local merged-group results.
- `record_type="round_summary"`: one final original-alert ranking after each
  30-minute round. Every original alert occurs once in that round's ranking.

Each group score is first assigned to all its `member_ids`. For each original
alert, the accumulator then computes `anomaly_score = score_sum / window_count`.
Only windows containing the alert are counted: no zero padding, no weighting by
group size and no voting over high/low predictions. The original alert object,
ID, score sum, window count, mean threshold and final priority are preserved.
Final rankings are sorted by decreasing mean score. High and low lists are
available in the summary's `priority_result`, as in the per-window results.

With `contamination="auto"`, the threshold remains 0.5. When fixed contamination
produces different window thresholds, the final decision compares the mean score
with the mean threshold from the same containing windows. Equality is low.
The summary's `available_at` is the round end; it must not be treated as a decision
available at the first window. Totals are released between rounds and scenarios.

```python
from main import run_pipeline

for record in run_pipeline(alerts, include_aggregates=True):
    if record["record_type"] == "round_summary":
        final_ranking = record["priority_result"]["ranked"]
```

The optional JSONL output refuses to overwrite an existing file. Do not sum window
counts as unique alerts; use the summary's `input_count` instead. Incident grouping
and recall remain unimplemented; neither module is currently executed.

Tests: `python -m unittest discover -s tests -v`.

## Simulated Storm Gate

```bash
python main.py --scenario-index 1 --storm-threshold 1000 --output storm-priorities.jsonl
```

The optional gate scans the full selected scenario every minute. A ten-minute
window with at least 1,000 raw alerts triggers priority processing. Counts are
computed before duplicate merging and do not use labels. This is a configurable
experimental assumption, not a validated attack detector. Without the option,
the original ungated replay is unchanged.

At the trigger window's end, the system processes that buffered ten-minute window
and continues for twenty more minutes. The round thus covers thirty minutes,
including the initial buffer, with 21 priority windows and a final mean ranking.
`storm_trigger` records the count, threshold and actual trigger time. The gate does
not use the following twenty minutes to make its trigger decision.

Selected rounds never overlap. After a round, scanning resumes with windows starting
at or after its end; this also prevents previously processed alerts being returned
again in a later round. `--max-rounds` limits triggered rounds in this mode, not
scanned windows. When none qualify, no priority model is run. The final replay
window/round may extend past the last recorded alert, with the remainder empty.

Evaluation must include all selected periods, including noise-only storms. Alerts
outside selected rounds are unprocessed, not automatically correct negatives.
Report trigger coverage separately from priority accuracy within selected rounds.

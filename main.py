"""Offline window replay and orchestration; only priority ranking is connected."""

import argparse
from bisect import bisect_left
from contextlib import nullcontext
import json
import math
from itertools import chain
from pathlib import Path

from alert_priority.main import prioritize_alerts


WINDOW_SECONDS = 600
STEP_SECONDS = 60
ROUND_SECONDS = 1800
ROOT = Path(__file__).resolve().parent


def iter_windows(alerts, *, start_time=None, max_rounds=None, stream_id="input"):
    """Yield 21 half-open 10-minute windows in each consecutive 30-minute round.

    Input timestamps must already share a timeline. Different scenarios must be
    passed separately. Repeated alerts retain IDs; windows never cross rounds.
    Results become available at window_end, not at window_start.
    """
    if not isinstance(alerts, list) or any(not isinstance(row, dict) for row in alerts):
        raise ValueError("Expected a JSON array of alert objects.")
    if max_rounds is not None and (type(max_rounds) is not int or max_rounds <= 0):
        raise ValueError("max_rounds must be a positive integer.")
    if start_time is not None and not math.isfinite(start_time):
        raise ValueError("start_time must be finite.")
    rows = []
    seen_ids = set()
    for i, alert in enumerate(alerts):
        row = dict(alert)
        timestamp = float(row["raw_time"])
        if not math.isfinite(timestamp):
            raise ValueError("raw_time must be finite.")
        row["raw_time"] = timestamp
        row.setdefault("alert_id", f"{stream_id}:{i}")
        if not isinstance(row["alert_id"], str) or not row["alert_id"] or row["alert_id"] in seen_ids:
            raise ValueError("alert_id values must be unique nonempty strings.")
        seen_ids.add(row["alert_id"])
        rows.append(row)
    if not rows:
        return
    rows.sort(key=lambda row: row["raw_time"])
    times = [row["raw_time"] for row in rows]
    origin = float(start_time) if start_time is not None else math.floor(times[0] / ROUND_SECONDS) * ROUND_SECONDS
    round_index = 0
    while origin + round_index * ROUND_SECONDS <= times[-1]:
        if max_rounds is not None and round_index >= max_rounds:
            break
        round_start = origin + round_index * ROUND_SECONDS
        for offset in range(0, ROUND_SECONDS - WINDOW_SECONDS + 1, STEP_SECONDS):
            window_start = round_start + offset
            window_end = window_start + WINDOW_SECONDS
            left, right = bisect_left(times, window_start), bisect_left(times, window_end)
            yield dict(
                stream_id=stream_id, round_index=round_index,
                round_start=round_start, round_end=round_start + ROUND_SECONDS,
                window_start=window_start, window_end=window_end,
                alert_count=right - left, alerts=rows[left:right],
            )
        round_index += 1


class WindowScoreAccumulator:
    """Average each original alert's group score once per containing window."""

    def __init__(self):
        self.totals = {}
        self.windows = set()

    def add(self, window, alerts):
        key = (window["window_start"], window["window_end"])
        if key in self.windows:
            raise ValueError("This window has already been accumulated.")
        by_id = {row["alert_id"]: row for row in alerts}
        if len(by_id) != len(alerts):
            raise ValueError("Original alert IDs must be unique within a window.")
        result = window["priority_result"]
        threshold = result["score_threshold"]
        updates = {}
        for group in result["ranked"]:
            score = group["anomaly_score"]
            if threshold is None or not math.isfinite(threshold) or not math.isfinite(score):
                raise ValueError("Nonempty window scores and threshold must be finite.")
            for aid in group["member_ids"]:
                if aid not in by_id or aid in updates:
                    raise ValueError("Each original member must occur exactly once per window.")
                updates[aid] = score
        if set(updates) != set(by_id):
            raise ValueError("Window results do not cover all original alerts.")
        # Validate the entire window before changing any running totals.
        for aid, score in updates.items():
            total = self.totals.setdefault(aid, dict(
                alert=dict(by_id[aid]), alert_id=aid, score_sum=0.0,
                threshold_sum=0.0, window_count=0,
            ))
            total["score_sum"] += score
            total["threshold_sum"] += threshold
            total["window_count"] += 1
        self.windows.add(key)

    def finalize(self):
        ranked = []
        for total in self.totals.values():
            item = dict(total)
            item["anomaly_score"] = total["score_sum"] / total["window_count"]
            item["score_threshold"] = item.pop("threshold_sum") / total["window_count"]
            item["priority"] = "high" if item["anomaly_score"] > item["score_threshold"] else "low"
            ranked.append(item)
        ranked.sort(key=lambda item: item["anomaly_score"], reverse=True)
        return dict(
            aggregation="mean_per_original_alert", input_count=len(ranked),
            window_count=len(self.windows), ranked=ranked,
            high_priority=[item for item in ranked if item["priority"] == "high"],
            low_priority=[item for item in ranked if item["priority"] == "low"],
        )


def run_pipeline(alerts, *, contamination="auto", include_aggregates=False, **window_options):
    """Yield window results and, optionally, one unique-alert ranking per round.

    The CLI enables aggregation. False preserves the original window-only API.
    Final means use only observed windows and are available at round_end.
    """
    accumulator = WindowScoreAccumulator()
    round_info = None
    for window in chain(iter_windows(alerts, **window_options), [None]):
        if include_aggregates and round_info is not None and (
            window is None or window["round_index"] != round_info["round_index"]
        ):
            yield dict(round_info, record_type="round_summary",
                       available_at=round_info["round_end"],
                       priority_result=accumulator.finalize())
            accumulator = WindowScoreAccumulator()
            round_info = None
        if window is None:
            break
        current_alerts = window.pop("alerts")
        window["priority_result"] = prioritize_alerts(current_alerts, contamination=contamination)
        if include_aggregates:
            if round_info is None:
                round_info = {key: window[key] for key in ("stream_id", "round_index", "round_start", "round_end")}
            accumulator.add(window, current_alerts)
            window["record_type"] = "window"
        yield window


def select_storm_periods(alerts, *, threshold=1000, start_time=None, max_rounds=None):
    """Simulate a raw-volume trigger without consulting labels or future counts.

    At each window end, count [end-600, end). A selected round contains that
    buffered trigger window and the following 20 minutes. Selected rounds do
    not overlap; sparse periods are skipped, never classified as noise here.
    """
    if type(threshold) is not int or threshold <= 0:
        raise ValueError("Storm threshold must be a positive integer.")
    if max_rounds is not None and (type(max_rounds) is not int or max_rounds <= 0):
        raise ValueError("max_rounds must be a positive integer.")
    if not isinstance(alerts, list) or any(not isinstance(row, dict) for row in alerts):
        raise ValueError("Expected a JSON array of alert objects.")
    times = sorted(float(row["raw_time"]) for row in alerts)
    if not all(math.isfinite(t) for t in times):
        raise ValueError("raw_time must be finite.")
    if start_time is not None and not math.isfinite(start_time):
        raise ValueError("start_time must be finite.")
    if not times:
        return
    start = float(start_time) if start_time is not None else math.floor(times[0] / STEP_SECONDS) * STEP_SECONDS
    selected = 0
    while start <= times[-1]:
        end = start + WINDOW_SECONDS
        count = bisect_left(times, end) - bisect_left(times, start)
        if count >= threshold:
            yield dict(round_start=start, round_end=start + ROUND_SECONDS,
                       trigger_time=end, trigger_count=count, threshold=threshold)
            selected += 1
            if max_rounds is not None and selected >= max_rounds:
                return
            start += ROUND_SECONDS
        else:
            start += STEP_SECONDS


def run_storm_pipeline(alerts, *, threshold=1000, start_time=None, max_rounds=None,
                       contamination="auto", stream_id="input"):
    """Run priority ranking only on volume-triggered, nonoverlapping rounds."""
    for index, trigger in enumerate(select_storm_periods(
        alerts, threshold=threshold, start_time=start_time, max_rounds=max_rounds
    )):
        for record in run_pipeline(
            alerts, start_time=trigger["round_start"], max_rounds=1,
            contamination=contamination, stream_id=stream_id, include_aggregates=True,
        ):
            record["round_index"] = index
            record["storm_trigger"] = dict(trigger)
            yield record


def load_scenarios(configuration="simul-attacks", split="test", scenario_index=None):
    """Use the repository loader's configured offsets, never concatenate relative fragments."""
    import pandas as pd
    from alertbert.aitads import AITAlertDatasetAugmented

    dataset = AITAlertDatasetAugmented(
        split=split, configuration=configuration, path=str(ROOT / "aitads_augmented")
    )
    indices = range(dataset.n_scenarios) if scenario_index is None else [scenario_index]
    for index in indices:
        if not 0 <= index < dataset.n_scenarios:
            raise ValueError(f"scenario_index must be in [0, {dataset.n_scenarios}).")
        yield f"{configuration}:{split}:{index}", pd.DataFrame(dataset.scenarios[index].data).to_dict("records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON array with unified timestamps; otherwise use configured AIT-ADS-A.")
    parser.add_argument("--configuration", default="simul-attacks")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--scenario-index", type=int, help="Zero-based scenario index; default: every scenario in split separately.")
    parser.add_argument("--start-time", type=float, help="Unix timestamp of the first round; default: floor first alert to 30 minutes.")
    parser.add_argument("--max-rounds", type=int, help="Limit rounds per scenario for a smoke run; default: entire scenario.")
    parser.add_argument("--storm-threshold", type=int, help="Enable simulated storm gating at this raw count per 10 minutes (e.g. 1000).")
    parser.add_argument("--contamination", default="auto", help="auto or a fixed ratio in (0, 0.5]; never inferred from test labels.")
    parser.add_argument("--output", type=Path, help="New JSONL file containing window records and averaged round_summary records.")
    args = parser.parse_args()
    contamination = "auto" if args.contamination == "auto" else float(args.contamination)
    if args.input:
        with args.input.open(encoding="utf-8") as handle:
            streams = [(args.input.stem, json.load(handle))]
    else:
        streams = load_scenarios(args.configuration, args.split, args.scenario_index)
    output_context = args.output.open("x", encoding="utf-8") if args.output else nullcontext(None)
    with output_context as output:
        print("Window=600s; step=60s; round=1800s; priority only; results available at window end.")
        print(f"Contamination={contamination}; window-local refitting; overlapping counts are NOT unique alerts.")
        print("Final rankings average each original alert's scores across its containing windows, once per round.")
        if args.storm_threshold is not None:
            print(f"Storm simulation: raw_count >= {args.storm_threshold}/600s; no label filtering; max-rounds limits triggered rounds.")
        for stream_id, alerts in streams:
            print(f"Scenario: {stream_id}; source alerts: {len(alerts)}", flush=True)
            count = 0
            options = dict(contamination=contamination, start_time=args.start_time,
                           max_rounds=args.max_rounds, stream_id=stream_id)
            if args.storm_threshold is None:
                records = run_pipeline(alerts, include_aggregates=True, **options)
            else:
                records = run_storm_pipeline(alerts, threshold=args.storm_threshold, **options)
            for window in records:
                result = window["priority_result"]
                if "storm_trigger" in window and window["record_type"] == "window" and window["window_start"] == window["round_start"]:
                    trigger = window["storm_trigger"]
                    print(f"Triggered at {trigger['trigger_time']:.0f}: raw_count={trigger['trigger_count']}; "
                          f"round=[{trigger['round_start']:.0f}, {trigger['round_end']:.0f})", flush=True)
                if window["record_type"] == "round_summary":
                    print(f"[Round {window['round_index']} averaged] unique_alerts={result['input_count']} "
                          f"high={len(result['high_priority'])} low={len(result['low_priority'])}", flush=True)
                else:
                    print(f"[{window['window_start']:.0f}, {window['window_end']:.0f}) "
                          f"alerts={window['alert_count']} merged={len(result['ranked'])} "
                          f"high={len(result['high_priority'])} low={len(result['low_priority'])}", flush=True)
                    count += 1
                if output:
                    output.write(json.dumps(window, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"Completed {count} windows for {stream_id}.")
            if args.storm_threshold is not None and not count:
                print("No storm periods met the threshold; priority ranking was not run.")


if __name__ == "__main__":
    main()

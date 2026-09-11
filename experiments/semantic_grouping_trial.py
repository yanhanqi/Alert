"""Trial: merge high-priority alerts inside attack windows by semantic distance.

Pipeline for one scenario:
  1. Cut the labelled attack window(s) from the AIT-ADS CSV.
  2. Priority ranking (patent module one) -> high-priority representatives.
  3. Semantic encoding + thresholded distance grouping (patent module two,
     semantic leg only) -> alert events.
  4. Compare the groups against the ground-truth event labels.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from alert_grouping.cooccurrence import TimeCooccurrenceModel
from alert_grouping.grouping import (
    combined_association_scores,
    group_high_priority_alerts,
)
from alert_grouping.semantic import AlertSemanticEncoder
from alert_priority.main import prioritize_alerts


def load_window(scenario, attack, start=None, end=None):
    """Return rows of one attack window, the union span of all attacks, or a
    caller-provided [start, end] range."""
    frame = pd.read_csv(
        ROOT / "datasets" / "alerts_csv" / f"{scenario}_alerts.txt",
        keep_default_na=False,
    )
    frame["time"] = pd.to_numeric(frame["time"], errors="raise")
    if start is None or end is None:
        labels = pd.read_csv(ROOT / "datasets" / "labels.csv", keep_default_na=False)
        scene = labels[labels.scenario == scenario]
        if scene.empty:
            raise ValueError(f"No attack windows for scenario {scenario!r}.")
        if attack != "all":
            scene = scene[scene.attack == attack]
            if scene.empty:
                raise ValueError(f"No attack window {attack!r} for {scenario!r}.")
        start, end = float(scene.start.min()), float(scene.end.max())
    start, end = float(start), float(end)
    rows = frame[(frame.time >= start) & (frame.time <= end)]
    if rows.empty:
        raise ValueError("The attack window contains no alerts.")
    return rows.reset_index(drop=True), start, end


def evaluate(groups, representatives, alert_id_to_label):
    """Purity and ground-truth coverage of semantic groups."""
    label_counter = Counter()
    rows = []
    for group in groups:
        labels = Counter()
        for member_index in group:
            for alert_id in representatives[member_index]["member_ids"]:
                labels[alert_id_to_label[alert_id]] += 1
        dominant, count = labels.most_common(1)[0]
        rows.append(dict(
            size=sum(labels.values()),
            dominant_label=dominant,
            purity=count / sum(labels.values()),
            labels=dict(labels),
        ))
        label_counter[dominant] += 1
    sizes = np.array([row["size"] for row in rows], dtype=float)
    purities = np.array([row["purity"] for row in rows])
    attack_purities = []
    for row in rows:
        attack_counts = {k: v for k, v in row["labels"].items() if k != "-"}
        if attack_counts:
            dominant = max(attack_counts, key=attack_counts.get)
            attack_purities.append(attack_counts[dominant] / sum(row["labels"].values()))
    return dict(
        n_groups=len(rows),
        n_singletons=int((sizes == 1).sum()),
        size_mean=float(sizes.mean()),
        purity_mean=float(purities.mean()),
        purity_mean_no_noise=float(np.mean(attack_purities)) if attack_purities else 0.0,
        group_labels=label_counter,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="russellmitchell")
    parser.add_argument("--attack", default="all", help="Attack name in labels.csv or 'all'.")
    parser.add_argument("--model", default=str(ROOT / "results/semantic/aitads_mlm_1k"))
    parser.add_argument("--contamination", type=float, default=0.2)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.9",
                        help="Comma-separated thresholds: semantic distance upper "
                             "bounds, and fused-score lower bounds with --cooc.")
    parser.add_argument("--cooc", action="store_true",
                        help="Fuse historical time co-occurrence with semantic distance.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight of normalized semantic distance in the fused score.")
    parser.add_argument("--history-window", type=float, default=600.0)
    parser.add_argument("--history-step", type=float, default=60.0)
    parser.add_argument("--start-time", type=float, help="Override window start (Unix seconds).")
    parser.add_argument("--end-time", type=float, help="Override window end (Unix seconds).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    window, start, end = load_window(args.scenario, args.attack, args.start_time, args.end_time)
    alerts = [
        dict(
            alert_id=f"{args.scenario}:{i}",
            time=float(row.time), raw_time=float(row.time),
            ip=row.ip, host=row.host, short=row.short, name=row.name,
            event_label=row.event_label,
        )
        for i, row in window.iterrows()
    ]
    print(f"Window [{start:.0f}, {end:.0f}]: {len(alerts)} alerts")
    result = prioritize_alerts(alerts, contamination=args.contamination)
    high = result["high_priority"]
    print(f"Priority: high={len(high)}, low={len(result['low_priority'])}, "
          f"threshold={result['score_threshold']:.4f}")
    if not high:
        print("No high-priority alerts; nothing to merge.")
        return

    encoder = AlertSemanticEncoder(args.model)
    representatives = high
    alert_id_to_label = {alert["alert_id"]: alert.get("event_label", "-") for alert in alerts}
    member_label = lambda item: Counter(  # noqa: E731 - dominant member label per rep
        alert_id_to_label[i] for i in item["member_ids"]
    ).most_common(1)[0][0]
    y_true = [member_label(item) for item in representatives]

    cooccurrence = None
    if args.cooc:
        history = pd.read_csv(
            ROOT / "datasets" / "alerts_csv" / f"{args.scenario}_alerts.txt",
            keep_default_na=False,
        )
        history["time"] = pd.to_numeric(history["time"], errors="raise")
        history = history[history.time < start]
        history_alerts = [
            dict(short=row.short, raw_time=float(row.time))
            for _, row in history.iterrows()
        ]
        cooccurrence = TimeCooccurrenceModel(args.history_window, args.history_step)
        cooccurrence.fit(history_alerts)
        print(f"Co-occurrence: {len(history_alerts)} historical alerts, "
              f"{cooccurrence.n_windows} windows, "
              f"{len(cooccurrence.type_window_counts)} types")

    for threshold in map(float, args.thresholds.split(",")):
        groups, labels, distances = group_high_priority_alerts(
            [item["alert"] for item in representatives], encoder, threshold,
            cooccurrence=cooccurrence, alpha=args.alpha,
        )
        summary = evaluate(groups, representatives, alert_id_to_label)
        ami = adjusted_mutual_info_score(y_true, labels)
        print(f"\n[{'fused' if args.cooc else 'semantic'} threshold={threshold}] "
              f"groups={summary['n_groups']} singletons={summary['n_singletons']} "
              f"size_mean={summary['size_mean']:.1f} "
              f"purity={summary['purity_mean']:.3f} "
              f"purity_no_noise={summary['purity_mean_no_noise']:.3f} "
              f"AMI={ami:.3f}")
        for dominant, count in summary["group_labels"].most_common(8):
            print(f"  {dominant}: {count} groups")
        if cooccurrence is not None:
            types = [item["alert"]["short"] for item in representatives]
            cooc_matrix = cooccurrence.pairwise_matrix(types)
            scores = combined_association_scores(distances, cooc_matrix, args.alpha)
            flat = scores[np.triu_indices_from(scores, 1)]
            print(f"  score mean={flat.mean():.3f} med={np.median(flat):.3f} "
                  f"edges={(flat >= threshold).sum()}/{flat.size}")
        else:
            flat = distances[np.triu_indices_from(distances, 1)]
            print(f"  d mean={flat.mean():.3f} med={np.median(flat):.3f} "
                  f"edges={(flat <= threshold).sum()}/{flat.size}")


if __name__ == "__main__":
    main()

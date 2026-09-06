"""Single-scenario Isolation Forest experiment with label-informed contamination."""

import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "aitads_augmented" / "data"
SCENARIO = "russellmitchell"
BASE_CATEGORICAL_FEATURES = ("ip", "host", "short")
EXPANDED_CATEGORICAL_FEATURES = ("ip", "host", "short", "name")
EXPANDED_NUMERIC_FEATURES = ("time", "raw_time")
LABEL_FIELDS = {"event_label", "time_label", "hierarchical_event_label"}
NOISE_LABEL = "-"
CONTEXT_WINDOW_SECONDS = 600
MERGE_WINDOW_SECONDS = 600
MERGE_KEYS = ("ip", "host", "short")


def load_alerts(files):
    """Merge the top-level JSON arrays without modifying the source files."""
    alerts = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Expected an array of alert objects: {path}")
        alerts.extend(rows)
    if not alerts:
        raise ValueError("No alerts loaded; check DATA_DIR and SCENARIO.")
    return alerts


def build_features(data, categorical_fields, numeric_fields=()):
    """Encode categorical frequencies and scale numeric fields to [0, 1]."""
    selected_fields = set(categorical_fields) | set(numeric_fields)
    if LABEL_FIELDS.intersection(selected_fields):
        raise ValueError("Ground-truth labels must not be used as input features.")
    missing = selected_fields - set(data.columns)
    if missing:
        raise ValueError(f"Missing feature fields: {sorted(missing)}")

    features = pd.DataFrame(index=data.index)
    for field in categorical_fields:
        values = data[field].fillna("<MISSING>").astype(str)
        features[f"{field}_frequency"] = values.map(values.value_counts(normalize=True))

    for field in numeric_fields:
        values = pd.to_numeric(data[field], errors="coerce")
        if values.isna().all():
            raise ValueError(f"Numeric field contains no valid values: {field}")
        values = values.fillna(values.median())
        minimum, maximum = values.min(), values.max()
        features[field] = 0.0 if minimum == maximum else (values - minimum) / (maximum - minimum)
    return features


def build_host_context(data, window_seconds=CONTEXT_WINDOW_SECONDS):
    """Same-host counts in [t - window_seconds, t); exclude equal timestamps."""
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Context window must be finite and positive.")
    ordered = data.loc[:, ["raw_time", "host", "short", "ip"]].copy()
    ordered["raw_time"] = pd.to_numeric(ordered["raw_time"], errors="raise")
    if not np.isfinite(ordered["raw_time"].to_numpy(dtype=float)).all():
        raise ValueError("Context timestamps must be finite.")
    for field in ("host", "short", "ip"):
        ordered[field] = ordered[field].fillna("<MISSING>").astype(str)
    ordered["position"] = np.arange(len(data))
    counts = np.zeros((len(data), 4), dtype=np.int64)
    for _, host_data in ordered.groupby("host", sort=False):
        records = list(
            host_data.sort_values("raw_time", kind="stable")
            [["position", "raw_time", "short", "ip"]]
            .itertuples(index=False, name=None)
        )
        window = deque()
        short_counts, ip_counts = Counter(), Counter()
        next_record = 0
        for position, timestamp, current_short, _ in records:
            # Delay all same-time records until a strictly later timestamp.
            while next_record < len(records) and records[next_record][1] < timestamp:
                _, previous_time, short, ip = records[next_record]
                window.append((previous_time, short, ip))
                short_counts[short] += 1
                ip_counts[ip] += 1
                next_record += 1
            while window and window[0][0] < timestamp - window_seconds:
                _, short, ip = window.popleft()
                for counter, value in ((short_counts, short), (ip_counts, ip)):
                    counter[value] -= 1
                    if counter[value] == 0:
                        del counter[value]
            counts[position] = (
                len(window), len(short_counts), len(ip_counts), short_counts[current_short]
            )
    return pd.DataFrame(
        counts, index=data.index,
        columns=(
            "host_past_alert_count", "host_past_short_count", "host_past_ip_count",
            "host_short_past_alert_count",
        ),
    )


def merge_duplicate_alerts(data, window_seconds=MERGE_WINDOW_SECONDS):
    """Offline session merge; consecutive equal-key alerts may be <=600s apart.

    Returns representatives at session end and a positional raw-row-to-group map.
    Labels never affect grouping. Metadata is excluded from model features.
    """
    if data.empty:
        raise ValueError("Cannot merge an empty alert dataset.")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Merge window must be finite and positive.")
    required = set(MERGE_KEYS) | {"raw_time"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing merge fields: {sorted(missing)}")
    if data.loc[:, list(MERGE_KEYS)].isna().any().any():
        raise ValueError("Merge keys must not contain missing values.")
    ordered = data.reset_index(drop=True).copy()
    ordered["raw_time"] = pd.to_numeric(ordered["raw_time"], errors="raise")
    if not np.isfinite(ordered["raw_time"].to_numpy(dtype=float)).all():
        raise ValueError("Merge timestamps must be finite.")
    ordered["_position"] = np.arange(len(data))
    ordered = ordered.sort_values(list(MERGE_KEYS) + ["raw_time"], kind="stable")
    gaps = ordered.groupby(list(MERGE_KEYS), sort=False)["raw_time"].diff()
    ordered["_merge_group"] = (gaps.isna() | gaps.gt(window_seconds)).cumsum() - 1
    groups = ordered.groupby("_merge_group", sort=False)
    representatives = groups.tail(1).copy()
    statistics = groups["raw_time"].agg(["size", "min", "max"])
    for output, source in (
        ("merged_count", "size"),
        ("merge_start_time", "min"),
        ("merge_end_time", "max"),
    ):
        representatives[output] = representatives["_merge_group"].map(statistics[source])
    representatives = representatives.sort_values("raw_time", kind="stable").reset_index(drop=True)
    new_ids = pd.Series(representatives.index.to_numpy(), index=representatives["_merge_group"])
    membership = np.empty(len(data), dtype=np.int64)
    membership[ordered["_position"].to_numpy()] = ordered["_merge_group"].map(new_ids).to_numpy()
    representatives = representatives.drop(columns=["_position", "_merge_group"])
    return representatives, membership


def prediction_metrics(labels, predictions, anomaly_scores):
    """Evaluate predictions on a stated unit: raw alerts or merged groups."""
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
        "ROC AUC": roc_auc_score(labels, anomaly_scores),
        "Average precision": average_precision_score(labels, anomaly_scores),
    }
    confusion = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return metrics, confusion


def evaluate(features, labels, contamination, return_scores=False):
    """Fit one Isolation Forest and return its in-sample metrics."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    predictions = (model.fit_predict(features) == -1).astype(int)
    anomaly_scores = -model.score_samples(features)
    metrics, confusion = prediction_metrics(labels, predictions, anomaly_scores)
    if return_scores:
        return metrics, confusion, predictions, anomaly_scores
    return metrics, confusion, predictions


def evaluate_merged_alerts(data, labels):
    merged, membership = merge_duplicate_alerts(data)
    sizes = np.bincount(membership, minlength=len(merged))
    attack_counts = np.bincount(membership, weights=np.asarray(labels), minlength=len(merged))
    merged_labels = (attack_counts > 0).astype(int)
    mixed = (attack_counts > 0) & (attack_counts < sizes)
    contamination = float(merged_labels.mean())
    print("\n[Duplicate merge preprocessing]")
    print(f"Merge keys: {', '.join(MERGE_KEYS)}; consecutive gap <= {MERGE_WINDOW_SECONDS}s")
    print("Session merging may span more than 600s; representative = last alert.")
    print("Offline evaluation: each session is finalized after its last occurrence.")
    print(f"Alerts: {len(data)} -> {len(merged)}; reduction: {1 - len(merged) / len(data):.2%}")
    print(f"Attack groups: {merged_labels.sum()}; noise groups: {(merged_labels == 0).sum()}")
    print(f"Mixed-label groups: {mixed.sum()}; group truth = contains any attack")
    print(f"Contamination (merged-group non-noise ratio): {contamination:.6f}")
    if not 0 < contamination <= 0.5:
        raise ValueError("Merged-group non-noise ratio must be in (0, 0.5].")

    features = pd.concat([
        build_features(merged, EXPANDED_CATEGORICAL_FEATURES, EXPANDED_NUMERIC_FEATURES),
        build_host_context(merged),
    ], axis=1)
    print(f"Feature dimensions: {features.shape[1]}; context counts merged representatives")
    metrics, confusion, predictions, scores = evaluate(
        features, merged_labels, contamination, return_scores=True
    )
    print("\n[Merged 10-feature model: merged-group evaluation]")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    tn, fp, fn, tp = confusion
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    raw_metrics, raw_confusion = prediction_metrics(labels, predictions[membership], scores[membership])
    print("\n[Merged model predictions mapped back to original alerts]")
    for name, value in raw_metrics.items():
        print(f"{name}: {value:.6f}")
    tn, fp, fn, tp = raw_confusion
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return raw_metrics


def main():
    files = sorted(DATA_DIR.glob(f"{SCENARIO}-*.json"))
    alerts = load_alerts(files)
    data = pd.DataFrame(alerts)
    required = (
        set(EXPANDED_CATEGORICAL_FEATURES)
        | set(EXPANDED_NUMERIC_FEATURES)
        | {"event_label"}
    )
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if data["event_label"].isna().any() or data["event_label"].eq("").any():
        raise ValueError("Every alert needs an event_label for evaluation.")

    labels = data["event_label"].ne(NOISE_LABEL).astype(int)
    contamination = float(labels.mean())
    if not 0 < contamination <= 0.5:
        raise ValueError(
            f"Non-noise ratio is {contamination:.6f}; IsolationForest requires "
            "0 < contamination <= 0.5. The ratio will not be silently capped."
        )

    print(f"Scenario: {SCENARIO}; JSON files: {len(files)}")
    print(f"Alerts: {len(alerts)}; noise: {(labels == 0).sum()}; non-noise: {labels.sum()}")
    print(f"Contamination (ground-truth non-noise ratio): {contamination:.6f}")
    print("Evaluation: fit and evaluate on the same data; positive = non-noise")
    print(f"Host context window: [t - {CONTEXT_WINDOW_SECONDS}s, t); excludes time ties")
    print("Context is exploratory: fragment-relative timestamps are merged without offsets;")
    print("these counts do not represent the original scenario's chronological history.")

    baseline = build_features(data, BASE_CATEGORICAL_FEATURES)
    expanded = build_features(data, EXPANDED_CATEGORICAL_FEATURES, EXPANDED_NUMERIC_FEATURES)
    context = build_host_context(data)
    configurations = (
        ("Baseline", baseline),
        ("Expanded", expanded),
        ("Expanded + host context", pd.concat([
            expanded, context.drop(columns="host_short_past_alert_count")
        ], axis=1)),
        ("Expanded + host context + host-short count", pd.concat([expanded, context], axis=1)),
    )
    results = {}
    for model_name, features in configurations:
        metrics, confusion, predictions = evaluate(features, labels, contamination)
        results[model_name] = metrics
        print(f"\n[{model_name}]")
        print(f"Features: {', '.join(features.columns)}")
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")
        tn, fp, fn, tp = confusion
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Predicted non-noise: {predictions.sum()} ({predictions.mean():.2%})")

    print("\n[Expanded - Baseline]")
    for name in results["Baseline"]:
        difference = results["Expanded"][name] - results["Baseline"][name]
        print(f"{name}: {difference:+.6f}")

    print("\n[Expanded + host context - Expanded]")
    for name in results["Expanded"]:
        difference = results["Expanded + host context"][name] - results["Expanded"][name]
        print(f"{name}: {difference:+.6f}")

    print("\n[Adding host-short count to existing context]")
    for name in results["Expanded + host context"]:
        difference = (
            results["Expanded + host context + host-short count"][name]
            - results["Expanded + host context"][name]
        )
        print(f"{name}: {difference:+.6f}")


    merged_raw_metrics = evaluate_merged_alerts(data, labels)
    print("\n[Merged - unmerged 10-feature model; both evaluated on original alerts]")
    for name, value in merged_raw_metrics.items():
        difference = value - results["Expanded + host context + host-short count"][name]
        print(f"{name}: {difference:+.6f}")


if __name__ == "__main__":
    main()

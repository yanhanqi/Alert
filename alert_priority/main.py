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
    counts = np.zeros((len(data), 3), dtype=np.int64)
    for _, host_data in ordered.groupby("host", sort=False):
        records = list(
            host_data.sort_values("raw_time", kind="stable")
            [["position", "raw_time", "short", "ip"]]
            .itertuples(index=False, name=None)
        )
        window = deque()
        short_counts, ip_counts = Counter(), Counter()
        next_record = 0
        for position, timestamp, _, _ in records:
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
            counts[position] = len(window), len(short_counts), len(ip_counts)
    return pd.DataFrame(
        counts, index=data.index,
        columns=("host_past_alert_count", "host_past_short_count", "host_past_ip_count"),
    )


def evaluate(features, labels, contamination):
    """Fit one Isolation Forest and return its in-sample metrics."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    predictions = (model.fit_predict(features) == -1).astype(int)
    anomaly_scores = -model.score_samples(features)
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
    return metrics, confusion, predictions


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
        ("Expanded + host context", pd.concat([expanded, context], axis=1)),
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


if __name__ == "__main__":
    main()

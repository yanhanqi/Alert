"""Single-scenario Isolation Forest experiment with label-informed contamination."""

import json
from pathlib import Path

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
FEATURES = ("ip", "host", "short")
NOISE_LABEL = "-"


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


def build_features(data):
    """Represent each selected categorical value by its dataset frequency."""
    forbidden = {"event_label", "time_label", "hierarchical_event_label"}
    if forbidden.intersection(FEATURES):
        raise ValueError("Ground-truth labels must not be used as input features.")
    features = pd.DataFrame(index=data.index)
    for field in FEATURES:
        values = data[field].fillna("<MISSING>").astype(str)
        features[f"{field}_frequency"] = values.map(values.value_counts(normalize=True))
    return features


def main():
    files = sorted(DATA_DIR.glob(f"{SCENARIO}-*.json"))
    alerts = load_alerts(files)
    data = pd.DataFrame(alerts)
    required = set(FEATURES) | {"event_label"}
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

    features = build_features(data)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    # This is an in-sample diagnostic, not a held-out generalization estimate.
    predictions = (model.fit_predict(features) == -1).astype(int)
    anomaly_scores = -model.score_samples(features)

    print(f"Scenario: {SCENARIO}; JSON files: {len(files)}")
    print(f"Alerts: {len(alerts)}; noise: {(labels == 0).sum()}; non-noise: {labels.sum()}")
    print(f"Features: {', '.join(features.columns)}")
    print(f"Contamination (ground-truth non-noise ratio): {contamination:.6f}")
    print("Evaluation: fit and evaluate on the same data; positive = non-noise")
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
        "ROC AUC": roc_auc_score(labels, anomaly_scores),
        "Average precision": average_precision_score(labels, anomaly_scores),
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Predicted non-noise: {predictions.sum()} ({predictions.mean():.2%})")


if __name__ == "__main__":
    main()

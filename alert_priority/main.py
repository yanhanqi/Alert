"""Single-scenario Isolation Forest experiment with label-informed contamination."""

import json
from pathlib import Path

import pandas as pd
from alertbert.aitads import AITAlertDatasetAugmented
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AITADS_A_DIR = PROJECT_ROOT / "aitads_augmented"
SCENARIO = "russellmitchell"
DATASET_SPLIT = "test"
SCENARIO_INDEX = 1
BASE_CATEGORICAL_FEATURES = ("ip", "host", "short")
EXPANDED_CATEGORICAL_FEATURES = ("ip", "host", "short", "name")
EXPANDED_NUMERIC_FEATURES = ("time", "raw_time")
LABEL_FIELDS = {"event_label", "time_label", "hierarchical_event_label"}
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


def load_original_aitads_scenario():
    """Reconstruct one original AIT-ADS scenario with the official loader."""
    dataset = AITAlertDatasetAugmented(
        split=DATASET_SPLIT,
        configuration="original",
        path=str(AITADS_A_DIR),
    )
    scenario_template = dataset.config[DATASET_SPLIT][SCENARIO_INDEX]
    fragment_names = [
        name
        for day in scenario_template
        for name in day["noise"] + [attack[0] for attack in day["attacks"]]
    ]
    if not fragment_names or any(
        not name.startswith(f"{SCENARIO}-") for name in fragment_names
    ):
        raise ValueError(
            f"Scenario index {SCENARIO_INDEX} does not refer to {SCENARIO}."
        )
    scenario = dataset.scenarios[SCENARIO_INDEX]
    return pd.DataFrame({field: values for field, values in scenario.data.items()})


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
    data = load_original_aitads_scenario()
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

    print(f"Dataset: AIT-ADS reconstructed by configuration=original")
    print(f"Scenario: {SCENARIO}; split: {DATASET_SPLIT}")
    print(f"Alerts: {len(data)}; noise: {(labels == 0).sum()}; non-noise: {labels.sum()}")
    print(f"Contamination (ground-truth non-noise ratio): {contamination:.6f}")
    print("Evaluation: fit and evaluate on the same data; positive = non-noise")

    configurations = (
        ("Baseline", BASE_CATEGORICAL_FEATURES, ()),
        ("Expanded", EXPANDED_CATEGORICAL_FEATURES, EXPANDED_NUMERIC_FEATURES),
    )
    results = {}
    for model_name, categorical_fields, numeric_fields in configurations:
        features = build_features(data, categorical_fields, numeric_fields)
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


if __name__ == "__main__":
    main()

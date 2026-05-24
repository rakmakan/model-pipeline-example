import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.reader import load_val
from model_pipeline.loader import load_model


def find_threshold(y_true: pd.Series, y_proba: np.ndarray, target_recall: float) -> float | None:
    thresholds = np.arange(0.05, 0.96, 0.01)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if recall >= target_recall:
            return round(float(t), 4)
    return None


def run(model_version: str, model_registry_path: Path = Path("model_registry.json")) -> float:
    with open(model_registry_path) as f:
        registry = json.load(f)

    entry = registry["models"].get(model_version)
    if entry is None:
        raise ValueError(f"Model version {model_version} not found in registry.")

    data_version = entry["data_version"]
    artifact_path = Path(f"models/{model_version}")
    config = json.loads((artifact_path / "train_config.json").read_text())
    target_recall = config["training"]["target_recall"]

    model = load_model(model_version)
    X_val, y_val = load_val(data_version)
    y_proba = model.predict_proba(X_val)

    threshold = find_threshold(y_val, y_proba, target_recall)

    if threshold is None:
        best_t = 0.05
        best_recall = 0.0
        thresholds = np.arange(0.05, 0.96, 0.01)
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tp = ((y_pred == 1) & (y_val == 1)).sum()
            fn = ((y_pred == 0) & (y_val == 1)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall > best_recall:
                best_recall, best_t = recall, t
        raise ValueError(
            f"No threshold achieves target recall {target_recall} on val set. "
            f"Best achievable: {best_recall:.3f} at threshold {best_t:.2f}"
        )

    metadata = json.loads((artifact_path / "metadata.json").read_text())
    metadata["threshold"] = threshold
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Threshold set to {threshold} (achieves recall ≥ {target_recall} on val set)")
    return threshold

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_pipeline.reader import load_test
from model_pipeline.loader import get_threshold, load_model
from replay_pipeline.reader import load_replay

MODEL_REGISTRY_PATH = Path("model_registry.json")


def _metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc = None
    return {"recall": round(recall, 4), "fpr": round(fpr, 4), "precision": round(precision, 4), "f1": round(f1, 4), "auc": round(auc, 4) if auc else None}


def run(model_version: str, replay_version: str) -> dict:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    entry = registry["models"].get(model_version)
    if entry is None:
        raise ValueError(f"Model version {model_version} not in registry.")

    threshold = get_threshold(model_version)
    data_version = entry["data_version"]
    model = load_model(model_version)

    X_test, y_test = load_test(data_version)
    y_proba_test = model.predict_proba(X_test)
    test_metrics = _metrics(y_test, y_proba_test, threshold)

    X_replay, y_replay, replay_meta = load_replay(replay_version)
    y_proba_replay = model.predict_proba(X_replay)
    replay_metrics = _metrics(y_replay, y_proba_replay, threshold)

    report = {
        "model_version": model_version,
        "data_version": data_version,
        "replay_version": replay_version,
        "threshold": threshold,
        "test": test_metrics,
        "replay": replay_metrics,
        "replay_meta": replay_meta,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    Path("reports").mkdir(exist_ok=True)
    report_path = Path(f"reports/eval_{model_version}.json")
    report_path.write_text(json.dumps(report, indent=2))

    registry["models"][model_version]["eval_report_path"] = str(report_path)
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n=== EVALUATION REPORT: {model_version} ===")
    print(f"Threshold : {threshold}")
    print(f"\nTest set ({data_version}):")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  FPR       : {test_metrics['fpr']:.4f}")
    print(f"  AUC       : {test_metrics['auc']}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"\nReplay ({replay_version}, {replay_meta['rows']} rows, {replay_meta['weeks_spanned']} weeks):")
    print(f"  Recall    : {replay_metrics['recall']:.4f}")
    print(f"  FPR       : {replay_metrics['fpr']:.4f}")
    print(f"\nReport saved → {report_path}")
    return report

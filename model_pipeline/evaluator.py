import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_pipeline.reader import load_test
from model_pipeline.loader import get_threshold, load_model
from replay_pipeline.reader import load_replay

MODEL_REGISTRY_PATH = Path("model_registry.json")
logger = logging.getLogger(__name__)


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
        if np.isnan(auc):
            logger.warning("AUC is NaN — only one class present in y_true")
            auc = None
    except ValueError as e:
        logger.warning("AUC could not be computed: %s", e)
        auc = None
    return {"recall": round(recall, 4), "fpr": round(fpr, 4), "precision": round(precision, 4), "f1": round(f1, 4), "auc": round(auc, 4) if auc is not None else None}


def run(model_version: str, replay_version: str) -> dict:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    entry = registry["models"].get(model_version)
    if entry is None:
        raise ValueError(f"Model version {model_version} not in registry.")

    threshold = get_threshold(model_version)
    data_version = entry["data_version"]
    logger.info("Evaluating model %s (data=%s, replay=%s, threshold=%s)", model_version, data_version, replay_version, threshold)
    model = load_model(model_version)

    logger.debug("Scoring test set (%s)", data_version)
    X_test, y_test = load_test(data_version)
    y_proba_test = model.predict_proba(X_test)
    test_metrics = _metrics(y_test, y_proba_test, threshold)
    logger.info("Test set  — recall=%.4f  fpr=%.4f  auc=%s  f1=%.4f", test_metrics["recall"], test_metrics["fpr"], test_metrics["auc"], test_metrics["f1"])

    logger.debug("Scoring replay dataset (%s)", replay_version)
    X_replay, y_replay, replay_meta = load_replay(replay_version)
    y_proba_replay = model.predict_proba(X_replay)
    replay_metrics = _metrics(y_replay, y_proba_replay, threshold)
    logger.info("Replay    — recall=%.4f  fpr=%.4f  auc=%s  (%d rows, %.1f weeks)", replay_metrics["recall"], replay_metrics["fpr"], replay_metrics["auc"], replay_meta["rows"], replay_meta["weeks_spanned"])

    if replay_metrics["recall"] < 0.80:
        logger.warning("Replay recall %.4f is below the gate floor of 0.80 — promotion will fail C1", replay_metrics["recall"])

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
    logger.debug("Report written to %s", report_path)

    registry["models"][model_version]["eval_report_path"] = str(report_path)
    registry["models"][model_version]["replay_metrics"] = {
        "recall": replay_metrics["recall"],
        "fpr": replay_metrics["fpr"],
        "auc": replay_metrics["auc"],
    }
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Report saved → %s", report_path)
    return report

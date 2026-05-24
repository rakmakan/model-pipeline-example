import json
import numpy as np
import pandas as pd
import pytest

from model_pipeline.validator import find_threshold, run


def test_find_threshold_returns_lowest_meeting_recall():
    # 10 fraud cases all scoring 0.35, 90 legit scoring 0.1
    # At threshold=0.3 (sweep start area), all 10 fraud caught → recall=1.0 ≥ 0.8
    y_true = pd.Series([1] * 10 + [0] * 90)
    y_proba = np.array([0.35] * 10 + [0.1] * 90)
    t = find_threshold(y_true, y_proba, target_recall=0.80)
    assert t is not None
    # Verify the threshold actually achieves target
    y_pred = (y_proba >= t).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    recall = tp / (tp + fn)
    assert recall >= 0.80


def test_find_threshold_returns_none_when_impossible():
    # All fraud scores well below 0.05 — no threshold achieves recall
    y_true = pd.Series([1] * 10 + [0] * 90)
    y_proba = np.array([0.001] * 100)
    t = find_threshold(y_true, y_proba, target_recall=0.80)
    assert t is None


def test_find_threshold_exact_boundary():
    # 5 fraud cases all scoring exactly 0.4, legit scoring 0.1
    y_true = pd.Series([1] * 5 + [0] * 95)
    y_proba = np.array([0.4] * 5 + [0.1] * 95)
    t = find_threshold(y_true, y_proba, target_recall=1.0)
    assert t is not None
    y_pred = (y_proba >= t).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    assert tp / (tp + fn) == 1.0


def test_run_writes_threshold_to_metadata(packed_artifact):
    artifact_path, version, _ = packed_artifact
    # Clear existing threshold
    meta = json.loads((artifact_path / "metadata.json").read_text())
    meta["threshold"] = None
    (artifact_path / "metadata.json").write_text(json.dumps(meta))

    run(version)  # default model_registry_path → tmp_path/model_registry.json

    meta_after = json.loads((artifact_path / "metadata.json").read_text())
    assert meta_after["threshold"] is not None
    assert isinstance(meta_after["threshold"], float)


def test_run_threshold_achieves_target_recall(packed_artifact, tiny_df):
    artifact_path, version, _ = packed_artifact
    meta = json.loads((artifact_path / "metadata.json").read_text())
    meta["threshold"] = None
    (artifact_path / "metadata.json").write_text(json.dumps(meta))

    threshold = run(version)

    from model_pipeline.loader import load_model
    from data_pipeline.reader import FEATURES
    model = load_model(version)
    X_val = tiny_df[FEATURES]
    y_val = tiny_df["label"]
    y_proba = model.predict_proba(X_val)
    y_pred = (y_proba >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_val == 1)).sum()
    fn = ((y_pred == 0) & (y_val == 1)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    assert recall >= 0.80, f"Threshold {threshold} achieves recall {recall}, not ≥ 0.80"


def test_run_raises_when_target_unachievable(packed_artifact, monkeypatch):
    _, version, _ = packed_artifact
    import model_pipeline.validator as validator_module
    class ZeroModel:
        def predict_proba(self, X):
            return np.zeros(len(X))
        _metadata = {"threshold": None, "features": [], "model_class_name": "Zero",
                     "schema_version": "v1", "data_version": "v_test"}
    monkeypatch.setattr(validator_module, "load_model", lambda v: ZeroModel())
    with pytest.raises(ValueError, match="Best achievable"):
        run(version)


def test_run_missing_model_version_raises(packed_artifact):
    with pytest.raises(ValueError, match="not found"):
        run("v_nonexistent")

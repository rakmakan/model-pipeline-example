import json
import logging
import numpy as np
import pandas as pd
import pytest

from model_pipeline.evaluator import _metrics, run


def test_metrics_perfect_recall():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    m = _metrics(pd.Series(y_true), y_proba, threshold=0.5)
    assert m["recall"] == 1.0


def test_metrics_zero_recall():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    m = _metrics(pd.Series(y_true), y_proba, threshold=0.5)
    assert m["recall"] == 0.0


def test_metrics_perfect_fpr():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    m = _metrics(pd.Series(y_true), y_proba, threshold=0.5)
    assert m["fpr"] == 0.0


def test_metrics_known_values():
    # 10 legit (8 TN, 2 FP), 10 fraud (8 TP, 2 FN)
    y_true  = np.array([0]*10 + [1]*10)
    y_proba = np.array([0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4])
    m = _metrics(pd.Series(y_true), y_proba, threshold=0.5)
    assert m["recall"] == pytest.approx(0.8, abs=0.01)
    assert m["fpr"] == pytest.approx(0.2, abs=0.01)


def test_metrics_auc_single_class_logs_warning_not_crash(caplog):
    y_true = np.zeros(10)
    y_proba = np.random.rand(10)
    with caplog.at_level(logging.WARNING, logger="model_pipeline.evaluator"):
        m = _metrics(pd.Series(y_true), y_proba, threshold=0.5)
    assert m["auc"] is None
    assert any("AUC" in r.message or "auc" in r.message.lower() for r in caplog.records)


def test_metrics_f1_zero_when_no_positives_predicted():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.1, 0.2])
    m = _metrics(pd.Series(y_true), y_proba, threshold=0.5)
    assert m["f1"] == 0.0


def test_run_creates_report_file(packed_artifact):
    _, version, tmp_path = packed_artifact
    (tmp_path / "reports").mkdir(exist_ok=True)
    run(version, version)
    assert (tmp_path / "reports" / f"eval_{version}.json").exists()


def test_run_report_has_required_keys(packed_artifact):
    _, version, tmp_path = packed_artifact
    (tmp_path / "reports").mkdir(exist_ok=True)
    report = run(version, version)
    for key in ("model_version", "data_version", "replay_version", "threshold", "test", "replay", "replay_meta"):
        assert key in report, f"Missing key: {key}"


def test_run_updates_registry_eval_report_path(packed_artifact):
    _, version, tmp_path = packed_artifact
    (tmp_path / "reports").mkdir(exist_ok=True)
    run(version, version)
    reg = json.loads((tmp_path / "model_registry.json").read_text())
    assert reg["models"][version]["eval_report_path"] is not None


def test_run_updates_registry_replay_metrics(packed_artifact):
    _, version, tmp_path = packed_artifact
    (tmp_path / "reports").mkdir(exist_ok=True)
    run(version, version)
    reg = json.loads((tmp_path / "model_registry.json").read_text())
    rm = reg["models"][version]["replay_metrics"]
    assert rm is not None
    assert all(k in rm for k in ("recall", "fpr", "auc"))


def test_run_missing_model_raises(packed_artifact):
    _, _, tmp_path = packed_artifact
    (tmp_path / "reports").mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="not in registry"):
        run("v_nonexistent", "v_test")

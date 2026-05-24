import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from replay_pipeline.pipeline import run
from replay_pipeline.reader import FEATURES, load_replay


def _make_replay_inputs(tmp_path, n_legit=150, n_fraud=60, weeks=10.0, include_unclear=False):
    """Build predictions.csv and feedback.csv in tmp_path."""
    n = n_legit + n_fraud
    start = datetime(2025, 1, 1)
    total_seconds = int(weeks * 7 * 24 * 3600)
    step = total_seconds // max(n - 1, 1)
    timestamps = [start + timedelta(seconds=i * step) for i in range(n)]

    predictions = pd.DataFrame({
        "prediction_id": [f"p{i:04d}" for i in range(n)],
        "timestamp": [t.isoformat() for t in timestamps],
        "model_version": "v1",
        "application_completion_seconds": [100.0] * n,
        "hour_of_day": [10] * n,
        "email_domain_risk_score": [0.2] * n,
        "account_age_days": [100] * n,
        "num_applications_last_24h": [1] * n,
        "ip_location_mismatch_km": [10.0] * n,
        "is_vpn_or_proxy": [0] * n,
        "profile_trust_score": [0.8] * n,
        "score": [0.5] * n,
        "decision": [0] * n,
        "label": [0] * n_legit + [1] * n_fraud,
    })
    predictions.to_csv(tmp_path / "predictions.csv", index=False)

    rows = []
    for _, row in predictions.iterrows():
        if include_unclear and int(row["prediction_id"][1:]) % 10 == 0:
            verdict = "unclear"
        else:
            verdict = "fraud" if row["label"] == 1 else "legit"
        rows.append({
            "prediction_id": row["prediction_id"],
            "feedback_timestamp": (datetime.fromisoformat(row["timestamp"]) + timedelta(days=5)).isoformat(),
            "verdict": verdict,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "feedback.csv", index=False)
    return str(tmp_path / "predictions.csv"), str(tmp_path / "feedback.csv")


def test_run_joins_on_prediction_id(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path)
    run(pred_path, fb_path, version="v1")
    replay = pd.read_csv(tmp_path / "data" / "replay" / "v1" / "replay.csv")
    assert len(replay) == 210  # n_legit + n_fraud = 150 + 60


def test_run_drops_unclear_verdicts(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path, n_legit=200, n_fraud=60, include_unclear=True)
    run(pred_path, fb_path, version="v1")
    replay = pd.read_csv(tmp_path / "data" / "replay" / "v1" / "replay.csv")
    assert len(replay) < 260


def test_run_labels_fraud_as_1_legit_as_0(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path)
    run(pred_path, fb_path, version="v1")
    replay = pd.read_csv(tmp_path / "data" / "replay" / "v1" / "replay.csv")
    assert set(replay["label"].unique()).issubset({0, 1})


def test_run_writes_replay_csv_and_metadata(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path)
    run(pred_path, fb_path, version="v1")
    assert (tmp_path / "data" / "replay" / "v1" / "replay.csv").exists()
    assert (tmp_path / "data" / "replay" / "v1" / "metadata.json").exists()


def test_run_writes_registry_entry(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path)
    run(pred_path, fb_path, version="v1")
    reg = json.loads((tmp_path / "data_registry.json").read_text())
    assert reg["replay"]["latest"] == "v1"
    entry = reg["replay"]["versions"]["v1"]
    assert "rows" in entry and "fraud_rows" in entry and "weeks_spanned" in entry


def test_run_fails_below_min_rows(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path, n_legit=80, n_fraud=20)
    with pytest.raises(ValueError, match="100"):
        run(pred_path, fb_path, version="v1")


def test_run_fails_below_min_weeks(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path, n_legit=150, n_fraud=60, weeks=2.0)
    with pytest.raises(ValueError, match="weeks"):
        run(pred_path, fb_path, version="v1")


def test_run_duplicate_version_raises(tmp_registries, tmp_path):
    pred_path, fb_path = _make_replay_inputs(tmp_path)
    run(pred_path, fb_path, version="v1")
    with pytest.raises(ValueError, match="v1"):
        run(pred_path, fb_path, version="v1")


def test_load_replay_returns_x_y_meta(packed_artifact):
    _, version, tmp_path = packed_artifact
    X, y, meta = load_replay(version)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(meta, dict)


def test_load_replay_x_has_feature_columns(packed_artifact):
    _, version, _ = packed_artifact
    X, _, _ = load_replay(version)
    assert list(X.columns) == FEATURES


def test_load_replay_meta_has_required_keys(packed_artifact):
    _, version, _ = packed_artifact
    _, _, meta = load_replay(version)
    for key in ("rows", "fraud_rows", "date_range", "weeks_spanned"):
        assert key in meta, f"Missing key: {key}"


def test_load_replay_missing_version_raises(tmp_registries, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_replay("v99")

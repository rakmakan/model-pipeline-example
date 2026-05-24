import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timedelta

FEATURES = [
    "application_completion_seconds",
    "hour_of_day",
    "email_domain_risk_score",
    "account_age_days",
    "num_applications_last_24h",
    "ip_location_mismatch_km",
    "is_vpn_or_proxy",
    "profile_trust_score",
]


@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(0)
    n, n_fraud = 100, 10
    legit = pd.DataFrame({
        "application_completion_seconds": rng.lognormal(6.5, 0.5, n - n_fraud),
        "hour_of_day": rng.integers(7, 23, n - n_fraud).tolist(),
        "email_domain_risk_score": rng.beta(2, 8, n - n_fraud),
        "account_age_days": rng.integers(30, 1000, n - n_fraud).tolist(),
        "num_applications_last_24h": rng.poisson(2, n - n_fraud).tolist(),
        "ip_location_mismatch_km": rng.exponential(15, n - n_fraud),
        "is_vpn_or_proxy": rng.binomial(1, 0.05, n - n_fraud).tolist(),
        "profile_trust_score": rng.beta(8, 2, n - n_fraud),
        "label": np.zeros(n - n_fraud, dtype=int),
    })
    fraud = pd.DataFrame({
        "application_completion_seconds": rng.lognormal(5.0, 0.8, n_fraud),
        "hour_of_day": rng.integers(0, 6, n_fraud).tolist(),
        "email_domain_risk_score": rng.beta(5, 3, n_fraud),
        "account_age_days": rng.integers(1, 30, n_fraud).tolist(),
        "num_applications_last_24h": rng.poisson(8, n_fraud).tolist(),
        "ip_location_mismatch_km": rng.exponential(150, n_fraud),
        "is_vpn_or_proxy": rng.binomial(1, 0.7, n_fraud).tolist(),
        "profile_trust_score": rng.beta(2, 5, n_fraud),
        "label": np.ones(n_fraud, dtype=int),
    })
    return pd.concat([legit, fraud], ignore_index=True)


@pytest.fixture
def tmp_registries(tmp_path, monkeypatch):
    """Empty registries in tmp_path. Changes cwd so all relative paths resolve there."""
    data_reg = {"latest": None, "versions": {}, "replay": {"latest": None, "versions": {}}}
    model_reg = {"active": None, "models": {}, "history": []}
    (tmp_path / "data_registry.json").write_text(json.dumps(data_reg))
    (tmp_path / "model_registry.json").write_text(json.dumps(model_reg))
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data_registry.json", tmp_path / "model_registry.json"


@pytest.fixture
def packed_artifact(tmp_path, tiny_df, monkeypatch):
    """
    Full test environment in tmp_path:
    - models/v_test/ packed artifact (threshold=0.3)
    - data/v_test/train.csv, val.csv, test.csv
    - data/replay/v_test/replay.csv + metadata.json
    - data_registry.json + model_registry.json

    Packing is done BEFORE chdir (shutil.copy needs repo-root relative paths).
    chdir happens last so all module file I/O resolves to tmp_path.
    """
    from train import LogisticRegressionModel, _pack_artifact

    version = "v_test"
    artifact_path = tmp_path / "models" / version

    # Train on tiny_df — must happen before chdir
    X = tiny_df[FEATURES]
    y = tiny_df["label"]
    model = LogisticRegressionModel({"max_iter": 100, "class_weight": "balanced"})
    model.fit(X, y)

    config = {
        "model": {"type": "LogisticRegression",
                  "hyperparameters": {"max_iter": 100, "class_weight": "balanced"}},
        "preprocessor": {"type": "StandardScaler"},
        "training": {"split_seed": 42, "target_recall": 0.80,
                     "_data_version": version, "_model_version": version},
    }
    _pack_artifact(model, artifact_path, version, config)

    # Set a threshold in metadata so loader tests can use get_threshold()
    meta = json.loads((artifact_path / "metadata.json").read_text())
    meta["threshold"] = 0.3
    (artifact_path / "metadata.json").write_text(json.dumps(meta))

    # Data splits
    data_dir = tmp_path / "data" / version
    data_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        tiny_df.to_csv(data_dir / f"{split}.csv", index=False)

    # Replay data
    replay_dir = tmp_path / "data" / "replay" / version
    replay_dir.mkdir(parents=True)
    tiny_df[FEATURES + ["label"]].to_csv(replay_dir / "replay.csv", index=False)
    start = datetime(2025, 1, 1)
    replay_meta = {
        "rows": len(tiny_df),
        "fraud_rows": int(tiny_df["label"].sum()),
        "date_range": [start.isoformat(), (start + timedelta(weeks=9)).isoformat()],
        "weeks_spanned": 9.0,
        "sources": {"predictions_hash": "sha256:test", "feedback_hash": "sha256:test"},
        "created_at": start.isoformat(),
    }
    (replay_dir / "metadata.json").write_text(json.dumps(replay_meta))

    # Registries
    data_reg = {
        "latest": version,
        "versions": {
            version: {
                "version": version,
                "input_file": "data/raw/test.csv",
                "data_hash": "sha256:test123",
                "schema_version": "v1",
                "split_ratio": [0.7, 0.15, 0.15],
                "split_seed": 42,
                "splits": {
                    "train": {"path": f"data/{version}/train.csv", "rows": len(tiny_df), "fraud_rate": 0.1},
                    "val":   {"path": f"data/{version}/val.csv",   "rows": len(tiny_df), "fraud_rate": 0.1},
                    "test":  {"path": f"data/{version}/test.csv",  "rows": len(tiny_df), "fraud_rate": 0.1},
                },
            }
        },
        "replay": {
            "latest": version,
            "versions": {version: {"path": f"data/replay/{version}/replay.csv", **replay_meta}},
        },
    }
    model_reg = {
        "active": None,
        "models": {
            version: {
                "version": version,
                "status": "candidate",
                "data_version": version,
                "data_hash": "sha256:test123",
                "artifact_path": f"models/{version}",
                "eval_report_path": None,
                "replay_metrics": None,
                "created_at": start.isoformat(),
                "promoted_at": None,
                "promotion_gate": None,
            }
        },
        "history": [],
    }
    (tmp_path / "data_registry.json").write_text(json.dumps(data_reg))
    (tmp_path / "model_registry.json").write_text(json.dumps(model_reg))

    # chdir LAST — after all file setup using repo-root relative paths
    monkeypatch.chdir(tmp_path)

    return artifact_path, version, tmp_path

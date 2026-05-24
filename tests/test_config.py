import json
import pytest
from pathlib import Path

from model_pipeline.config import load_and_validate

BASE_CONFIG = {
    "model": {
        "type": "LogisticRegression",
        "hyperparameters": {"max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced"},
    },
    "preprocessor": {"type": "StandardScaler"},
    "training": {"split_seed": 42, "target_recall": 0.80},
}


def _write(tmp_path, cfg) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def _mutate(base, **overrides):
    import copy
    cfg = copy.deepcopy(base)
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d[part]
        if value is None:
            del d[parts[-1]]
        else:
            d[parts[-1]] = value
    return cfg


def test_valid_config_returns_dict(tmp_path):
    p = _write(tmp_path, BASE_CONFIG)
    result = load_and_validate(p)
    assert result["training"]["target_recall"] == 0.80


def test_invalid_model_type_raises(tmp_path):
    cfg = _mutate(BASE_CONFIG, **{"model.type": "XGBoost"})
    with pytest.raises(ValueError, match="XGBoost"):
        load_and_validate(_write(tmp_path, cfg))


def test_invalid_preprocessor_type_raises(tmp_path):
    cfg = _mutate(BASE_CONFIG, **{"preprocessor.type": "MinMaxScaler"})
    with pytest.raises(ValueError, match="MinMaxScaler"):
        load_and_validate(_write(tmp_path, cfg))


def test_missing_target_recall_raises(tmp_path):
    cfg = _mutate(BASE_CONFIG, **{"training.target_recall": None})
    with pytest.raises(ValueError, match="target_recall"):
        load_and_validate(_write(tmp_path, cfg))


def test_target_recall_zero_raises(tmp_path):
    cfg = _mutate(BASE_CONFIG, **{"training.target_recall": 0.0})
    with pytest.raises(ValueError, match="target_recall"):
        load_and_validate(_write(tmp_path, cfg))


def test_target_recall_above_one_raises(tmp_path):
    cfg = _mutate(BASE_CONFIG, **{"training.target_recall": 1.5})
    with pytest.raises(ValueError, match="target_recall"):
        load_and_validate(_write(tmp_path, cfg))


def test_missing_split_seed_raises(tmp_path):
    cfg = _mutate(BASE_CONFIG, **{"training.split_seed": None})
    with pytest.raises(ValueError, match="split_seed"):
        load_and_validate(_write(tmp_path, cfg))


def test_multiple_errors_in_one_raise(tmp_path):
    cfg = _mutate(BASE_CONFIG, **{"model.type": "XGBoost", "training.target_recall": None})
    with pytest.raises(ValueError) as exc:
        load_and_validate(_write(tmp_path, cfg))
    msg = str(exc.value)
    assert "XGBoost" in msg
    assert "target_recall" in msg

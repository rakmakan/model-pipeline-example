import json
import pandas as pd
import pytest

from data_pipeline.pipeline import run
from data_pipeline.reader import FEATURES, load_train, load_val, load_test
from data_pipeline.schema import SCHEMAS


def test_run_creates_split_files(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    run(str(tmp_path / "input.csv"), version="v1")
    assert (tmp_path / "data" / "v1" / "train.csv").exists()
    assert (tmp_path / "data" / "v1" / "val.csv").exists()
    assert (tmp_path / "data" / "v1" / "test.csv").exists()


def test_run_split_ratios_approx_70_15_15(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    run(str(tmp_path / "input.csv"), version="v1")
    n = len(tiny_df)
    train = pd.read_csv(tmp_path / "data" / "v1" / "train.csv")
    val   = pd.read_csv(tmp_path / "data" / "v1" / "val.csv")
    test  = pd.read_csv(tmp_path / "data" / "v1" / "test.csv")
    assert abs(len(train) / n - 0.70) < 0.03
    assert abs(len(val)   / n - 0.15) < 0.03
    assert abs(len(test)  / n - 0.15) < 0.03


def test_run_stratified_fraud_rate(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    run(str(tmp_path / "input.csv"), version="v1")
    fraud_rate = tiny_df["label"].mean()
    for split in ("train", "val", "test"):
        df = pd.read_csv(tmp_path / "data" / "v1" / f"{split}.csv")
        assert abs(df["label"].mean() - fraud_rate) < 0.05


def test_run_writes_registry_entry(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    run(str(tmp_path / "input.csv"), version="v1")
    reg = json.loads((tmp_path / "data_registry.json").read_text())
    assert "v1" in reg["versions"]
    entry = reg["versions"]["v1"]
    assert "data_hash" in entry
    assert entry["data_hash"].startswith("sha256:")
    assert entry["schema_version"] == "v1"
    assert entry["split_seed"] == 42


def test_run_auto_increments_version(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    v1 = run(str(tmp_path / "input.csv"))
    v2 = run(str(tmp_path / "input.csv"))
    assert v1 == "v1"
    assert v2 == "v2"


def test_run_duplicate_version_raises(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    run(str(tmp_path / "input.csv"), version="v1")
    with pytest.raises(ValueError, match="v1"):
        run(str(tmp_path / "input.csv"), version="v1")


def test_run_invalid_schema_raises(tmp_registries, tiny_df, tmp_path):
    bad = tiny_df.drop(columns=["label"])
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        run(str(tmp_path / "bad.csv"), version="v1")


def test_load_train_returns_features_and_label(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    run(str(tmp_path / "input.csv"), version="v1")
    X, y = load_train("v1")
    assert list(X.columns) == FEATURES
    assert y.name == "label"
    assert len(X) == len(y)


def test_load_val_returns_correct_shape(tmp_registries, tiny_df, tmp_path):
    tiny_df.to_csv(tmp_path / "input.csv", index=False)
    run(str(tmp_path / "input.csv"), version="v1")
    X_val, _ = load_val("v1")
    X_test, _ = load_test("v1")
    assert len(X_val) > 0
    assert len(X_test) > 0
    X_train, _ = load_train("v1")
    assert abs(len(X_train) + len(X_val) + len(X_test) - len(tiny_df)) <= 1


def test_load_missing_version_raises(tmp_registries):
    with pytest.raises(FileNotFoundError):
        load_train("v99")


def test_features_list_matches_schema():
    schema_cols = set(SCHEMAS["v1"].keys()) - {"label"}
    assert set(FEATURES) == schema_cols

import json
import pytest
import pandas as pd

from data_pipeline.reader import FEATURES
from model_pipeline.loader import get_features, get_threshold, load_model


def test_load_model_returns_predict_proba(packed_artifact):
    _, version, _ = packed_artifact
    model = load_model(version)
    assert callable(getattr(model, "predict_proba", None))


def test_load_model_scores_in_zero_one_range(packed_artifact, tiny_df):
    _, version, _ = packed_artifact
    model = load_model(version)
    score = model.predict_proba(tiny_df[FEATURES].iloc[:1])[0]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_load_model_missing_artifact_raises(packed_artifact):
    with pytest.raises(FileNotFoundError, match="v_nonexistent"):
        load_model("v_nonexistent")


def test_get_threshold_returns_float(packed_artifact):
    _, version, _ = packed_artifact
    t = get_threshold(version)
    assert isinstance(t, float)
    assert t == pytest.approx(0.3)


def test_get_threshold_none_raises(packed_artifact):
    artifact_path, version, _ = packed_artifact
    meta = json.loads((artifact_path / "metadata.json").read_text())
    meta["threshold"] = None
    (artifact_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="validate.py"):
        get_threshold(version)


def test_get_features_returns_eight_columns(packed_artifact):
    _, version, _ = packed_artifact
    features = get_features(version)
    assert len(features) == 8
    assert features == FEATURES


def test_load_uses_artifact_snapshot_not_codebase(packed_artifact, tiny_df):
    """
    Modify the artifact's model_class.py to add a class-level marker.
    Verify load_model() uses the modified artifact snapshot, not the codebase.
    """
    artifact_path, version, _ = packed_artifact
    src = (artifact_path / "model_class.py").read_text()
    # Insert as class-level attribute so it's accessible via model.__class__
    modified = src.replace(
        "class LogisticRegressionModel(BaseModel):\n",
        "class LogisticRegressionModel(BaseModel):\n    LOADED_FROM_ARTIFACT = True\n",
    )
    (artifact_path / "model_class.py").write_text(modified)

    model = load_model(version)
    assert getattr(model.__class__, "LOADED_FROM_ARTIFACT", False) is True, (
        "load_model() did not load from the artifact snapshot — "
        "it may be importing from the codebase instead"
    )

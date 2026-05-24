import numpy as np
import pytest

from data_pipeline.reader import FEATURES
from model_pipeline.preprocessor import Preprocessor


def test_fit_transform_output_mean_near_zero(tiny_df):
    p = Preprocessor()
    X = tiny_df[FEATURES]
    out = p.fit_transform(X)
    for col in out.columns:
        assert abs(out[col].mean()) < 0.01, f"Column {col} mean not near zero: {out[col].mean()}"


def test_fit_transform_output_std_near_one(tiny_df):
    p = Preprocessor()
    X = tiny_df[FEATURES]
    out = p.fit_transform(X)
    for col in out.columns:
        assert abs(out[col].std() - 1.0) < 0.05, f"Column {col} std not near one: {out[col].std()}"


def test_fit_transform_preserves_columns(tiny_df):
    p = Preprocessor()
    X = tiny_df[FEATURES]
    out = p.fit_transform(X)
    assert list(out.columns) == list(X.columns)


def test_fit_transform_preserves_index(tiny_df):
    p = Preprocessor()
    X = tiny_df[FEATURES]
    out = p.fit_transform(X)
    assert list(out.index) == list(X.index)


def test_transform_before_fit_raises(tiny_df):
    p = Preprocessor()
    with pytest.raises(RuntimeError, match="fit_transform"):
        p.transform(tiny_df[FEATURES])


def test_transform_consistent_with_fit(tiny_df):
    p = Preprocessor()
    X = tiny_df[FEATURES]
    out_fit = p.fit_transform(X)
    out_transform = p.transform(X)
    assert (out_fit.values == out_transform.values).all()


def test_save_load_roundtrip(tmp_path, tiny_df):
    p = Preprocessor()
    X = tiny_df[FEATURES]
    p.fit_transform(X)
    p.save(tmp_path)

    p2 = Preprocessor.load(tmp_path)
    out1 = p.transform(X)
    out2 = p2.transform(X)
    np.testing.assert_array_almost_equal(out1.values, out2.values)


def test_load_fitted_flag_set(tmp_path, tiny_df):
    p = Preprocessor()
    p.fit_transform(tiny_df[FEATURES])
    p.save(tmp_path)
    p2 = Preprocessor.load(tmp_path)
    assert p2._fitted is True
    p2.transform(tiny_df[FEATURES])  # must not raise

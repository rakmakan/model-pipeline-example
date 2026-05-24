import pandas as pd
import pytest

from data_pipeline.schema import LATEST_SCHEMA, SCHEMAS, detect_schema_version, validate

VALID_COLS = list(SCHEMAS["v1"].keys())


def _valid_df():
    return pd.DataFrame({col: [1.0] for col in VALID_COLS})


def test_detect_exact_match():
    df = _valid_df()
    assert detect_schema_version(df) == "v1"


def test_detect_subset_match_returns_version():
    df = _valid_df()
    df["extra_column"] = 0
    result = detect_schema_version(df)
    assert result == "v1"


def test_detect_no_match_returns_latest():
    df = pd.DataFrame({"col_a": [1], "col_b": [2]})
    assert detect_schema_version(df) == LATEST_SCHEMA


def test_validate_passes_clean_df():
    df = _valid_df()
    validate(df, "v1")  # must not raise


def test_validate_missing_column_named_in_error():
    df = _valid_df().drop(columns=["label"])
    with pytest.raises(ValueError) as exc:
        validate(df, "v1")
    assert "Missing columns" in str(exc.value)
    assert "label" in str(exc.value)


def test_validate_unexpected_column_named_in_error():
    df = _valid_df()
    df["surprise"] = 0
    with pytest.raises(ValueError) as exc:
        validate(df, "v1")
    assert "Unexpected columns" in str(exc.value)
    assert "surprise" in str(exc.value)


def test_validate_collects_all_violations():
    df = _valid_df().drop(columns=["label"])
    df["surprise"] = 0
    with pytest.raises(ValueError) as exc:
        validate(df, "v1")
    msg = str(exc.value)
    assert "Missing columns" in msg
    assert "Unexpected columns" in msg

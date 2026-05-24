import pandas as pd

SCHEMAS = {
    "v1": {
        "application_completion_seconds": float,
        "hour_of_day": int,
        "email_domain_risk_score": float,
        "account_age_days": int,
        "num_applications_last_24h": int,
        "ip_location_mismatch_km": float,
        "is_vpn_or_proxy": int,
        "profile_trust_score": float,
        "label": int,
    }
}

LATEST_SCHEMA = "v1"


def detect_schema_version(df: pd.DataFrame) -> str:
    for version, schema in SCHEMAS.items():
        if set(schema.keys()) == set(df.columns):
            return version
    cols = set(df.columns)
    for version, schema in SCHEMAS.items():
        expected = set(schema.keys())
        if expected.issubset(cols):
            return version
    return LATEST_SCHEMA


def validate(df: pd.DataFrame, schema_version: str) -> None:
    if schema_version not in SCHEMAS:
        raise ValueError(f"Unknown schema version: {schema_version}")
    schema = SCHEMAS[schema_version]
    expected_cols = set(schema.keys())
    actual_cols = set(df.columns)

    missing = sorted(expected_cols - actual_cols)
    unexpected = sorted(actual_cols - expected_cols)
    type_mismatches = []
    for col, expected_type in schema.items():
        if col not in df.columns:
            continue
        actual_dtype = df[col].dtype
        if expected_type == float and not pd.api.types.is_float_dtype(actual_dtype):
            if not pd.api.types.is_numeric_dtype(actual_dtype):
                type_mismatches.append(f"  '{col}': expected float, got {actual_dtype}")
        elif expected_type == int and not pd.api.types.is_integer_dtype(actual_dtype):
            if not pd.api.types.is_numeric_dtype(actual_dtype):
                type_mismatches.append(f"  '{col}': expected int, got {actual_dtype}")

    if missing or unexpected or type_mismatches:
        lines = ["Schema mismatch:"]
        if missing:
            lines.append(f"  Missing columns  : {missing}")
        if unexpected:
            lines.append(f"  Unexpected columns: {unexpected}")
        if type_mismatches:
            lines.append("  Type mismatches  :")
            lines.extend(type_mismatches)
        raise ValueError("\n".join(lines))

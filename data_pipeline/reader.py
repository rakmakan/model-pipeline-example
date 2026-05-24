from pathlib import Path
import pandas as pd

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


def _load(version: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(f"data/{version}/{split}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}. Run run_data_pipe.py first.")
    df = pd.read_csv(path)
    return df[FEATURES], df["label"]


def load_train(version: str) -> tuple[pd.DataFrame, pd.Series]:
    return _load(version, "train")

def load_val(version: str) -> tuple[pd.DataFrame, pd.Series]:
    return _load(version, "val")

def load_test(version: str) -> tuple[pd.DataFrame, pd.Series]:
    return _load(version, "test")

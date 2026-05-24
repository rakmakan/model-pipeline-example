import json
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


def load_replay(version: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    base = Path(f"data/replay/{version}")
    if not base.exists():
        raise FileNotFoundError(f"Replay version {version} not found. Run run_replay_pipe.py first.")
    df = pd.read_csv(base / "replay.csv")
    meta = json.loads((base / "metadata.json").read_text())
    return df[FEATURES], df["label"], meta

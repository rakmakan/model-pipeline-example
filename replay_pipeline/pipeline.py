import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_REGISTRY_PATH = Path("data_registry.json")
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
MIN_MINORITY_CASES = 30  # minimum fraud cases for stable recall computation
MIN_ROWS_FALLBACK = 200  # used only when no training version exists in registry
MIN_WEEKS = 4.0


def _resample_to_ratio(df: pd.DataFrame, target_fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    """
    Resample to match target_fraud_rate while maximising total rows.

    Two cases:
    - Fraud over-represented (replay fraud% > training fraud%):
      keep all legit, downsample fraud to match ratio.
    - Legit insufficient for all fraud cases:
      keep all fraud, downsample legit to match ratio.
    """
    fraud = df[df["label"] == 1]
    legit = df[df["label"] == 0]

    legit_needed = int(len(fraud) * (1 - target_fraud_rate) / target_fraud_rate)

    if legit_needed <= len(legit):
        legit_out = legit.sample(n=legit_needed, random_state=seed)
        fraud_out = fraud
    else:
        fraud_needed = max(1, int(len(legit) * target_fraud_rate / (1 - target_fraud_rate)))
        fraud_out = fraud.sample(n=min(fraud_needed, len(fraud)), random_state=seed)
        legit_out = legit

    return pd.concat([fraud_out, legit_out]).sample(frac=1, random_state=seed).reset_index(drop=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


def _next_version(registry: dict) -> str:
    versions = list(registry.get("replay", {}).get("versions", {}).keys())
    if not versions:
        return "v1"
    nums = [int(v[1:]) for v in versions if v[1:].isdigit()]
    return f"v{max(nums) + 1}"


def run(predictions_path: str, feedback_path: str, version: str | None = None) -> str:
    predictions_path = Path(predictions_path)
    feedback_path = Path(feedback_path)

    logger.info("Loading predictions from %s", predictions_path)
    predictions = pd.read_csv(predictions_path)
    logger.info("Loading feedback from %s", feedback_path)
    feedback = pd.read_csv(feedback_path)

    df = predictions.merge(feedback, on="prediction_id", how="inner")
    logger.debug("Joined %d predictions to %d feedback rows → %d matches", len(predictions), len(feedback), len(df))

    unclear_count = (df["verdict"] == "unclear").sum()
    if unclear_count:
        logger.warning("Dropping %d 'unclear' verdicts", unclear_count)
    df = df[df["verdict"] != "unclear"].copy()
    df["label"] = (df["verdict"] == "fraud").astype(int)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    date_range = [df["timestamp"].min(), df["timestamp"].max()]
    weeks_spanned = (date_range[1] - date_range[0]).days / 7

    raw_rows = len(df)
    raw_fraud = int(df["label"].sum())
    logger.debug(
        "Raw labeled data: %d rows, %d fraud (%.1f%%), %.1f weeks (%s – %s)",
        raw_rows, raw_fraud, raw_fraud / raw_rows * 100 if raw_rows else 0,
        weeks_spanned, date_range[0].date(), date_range[1].date(),
    )

    # Load registry early — needed to derive dynamic minimum from training fraud rate
    with open(DATA_REGISTRY_PATH) as f:
        registry = json.load(f)

    # Derive minimum rows from training class ratio; fall back to fixed floor if no training data exists
    latest_data = registry.get("latest")
    if latest_data and latest_data in registry.get("versions", {}):
        train_fraud_rate = registry["versions"][latest_data]["splits"]["train"]["fraud_rate"]
        min_rows = math.ceil(MIN_MINORITY_CASES / train_fraud_rate)
        logger.info(
            "Training fraud rate %.1f%% (data=%s) → minimum replay rows: %d",
            train_fraud_rate * 100, latest_data, min_rows,
        )
    else:
        train_fraud_rate = None
        min_rows = MIN_ROWS_FALLBACK
        logger.warning("No training data version in registry — using fallback minimum of %d rows", min_rows)

    errors = []
    if raw_rows < min_rows:
        errors.append(f"Only {raw_rows} labeled rows — need at least {min_rows}.")
    if train_fraud_rate and raw_fraud < MIN_MINORITY_CASES:
        errors.append(f"Only {raw_fraud} fraud cases — need at least {MIN_MINORITY_CASES} for stable recall.")
    if weeks_spanned < MIN_WEEKS:
        errors.append(f"Dataset spans {weeks_spanned:.1f} weeks — need at least {MIN_WEEKS}.")
    if errors:
        raise ValueError("Replay dataset does not meet floor conditions:\n" + "\n".join(f"  {e}" for e in errors))

    if version is None:
        version = _next_version(registry)

    if version in registry.get("replay", {}).get("versions", {}):
        raise ValueError(f"Replay version {version} already exists in registry.")

    # Resample to match training class ratio, maximising total rows
    features_df = df[FEATURES + ["label"]].reset_index(drop=True)
    if train_fraud_rate is not None:
        replay_df = _resample_to_ratio(features_df, target_fraud_rate=train_fraud_rate)
        logger.info(
            "Resampled: %d → %d rows | fraud: %d (%.1f%%) → %d (%.1f%%)",
            raw_rows, len(replay_df),
            raw_fraud, raw_fraud / raw_rows * 100,
            int(replay_df["label"].sum()), replay_df["label"].mean() * 100,
        )
    else:
        replay_df = features_df

    out_dir = Path(f"data/replay/{version}")
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_df.to_csv(out_dir / "replay.csv", index=False)

    meta = {
        "rows": len(replay_df),
        "fraud_rows": int(replay_df["label"].sum()),
        "fraud_rate": round(replay_df["label"].mean(), 4),
        "raw_rows": raw_rows,
        "raw_fraud_rows": raw_fraud,
        "training_data_version": latest_data,
        "training_fraud_rate": train_fraud_rate,
        "date_range": [date_range[0].isoformat(), date_range[1].isoformat()],
        "weeks_spanned": round(weeks_spanned, 1),
        "sources": {
            "predictions_hash": _sha256(predictions_path),
            "feedback_hash": _sha256(feedback_path),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    if "replay" not in registry:
        registry["replay"] = {"latest": None, "versions": {}}
    registry["replay"]["versions"][version] = {"path": str(out_dir / "replay.csv"), **meta}
    registry["replay"]["latest"] = version

    with open(DATA_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(
        "Created replay version %s: %d rows, %d fraud (%.1f%%), %.1f weeks",
        version, len(replay_df), meta["fraud_rows"],
        meta["fraud_rows"] / len(replay_df) * 100, weeks_spanned,
    )
    return version

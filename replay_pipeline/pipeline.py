import hashlib
import json
import logging
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
MIN_ROWS = 200
MIN_WEEKS = 4.0


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

    logger.debug(
        "Replay dataset: %d rows, %d fraud, %.1f weeks (%s – %s)",
        len(df), df["label"].sum(), weeks_spanned,
        date_range[0].date(), date_range[1].date(),
    )

    errors = []
    if len(df) < MIN_ROWS:
        errors.append(f"Only {len(df)} labeled rows — need at least {MIN_ROWS}.")
    if weeks_spanned < MIN_WEEKS:
        errors.append(f"Dataset spans {weeks_spanned:.1f} weeks — need at least {MIN_WEEKS}.")
    if errors:
        raise ValueError("Replay dataset does not meet floor conditions:\n" + "\n".join(f"  {e}" for e in errors))

    with open(DATA_REGISTRY_PATH) as f:
        registry = json.load(f)

    if version is None:
        version = _next_version(registry)

    if version in registry.get("replay", {}).get("versions", {}):
        raise ValueError(f"Replay version {version} already exists in registry.")

    out_dir = Path(f"data/replay/{version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    replay_df = df[FEATURES + ["label"]].reset_index(drop=True)
    replay_df.to_csv(out_dir / "replay.csv", index=False)

    meta = {
        "rows": len(replay_df),
        "fraud_rows": int(replay_df["label"].sum()),
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

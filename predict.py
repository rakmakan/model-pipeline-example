"""
Scores a job application using the currently active model.

Usage:
    python predict.py
"""
import json
import logging
from pathlib import Path

import pandas as pd

from model_pipeline.loader import get_features, get_threshold, load_model

MODEL_REGISTRY_PATH = Path("model_registry.json")
logger = logging.getLogger(__name__)


def score(application: dict) -> dict:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    active_version = registry.get("active")
    if active_version is None:
        raise RuntimeError("No active model. Run promote.py --bootstrap first.")

    logger.debug("Loading active model %s", active_version)
    model = load_model(active_version)
    threshold = get_threshold(active_version)
    features = get_features(active_version)

    X = pd.DataFrame([application])[features]
    proba = float(model.predict_proba(X)[0])
    decision = "block" if proba >= threshold else "allow"
    logger.debug("score=%.4f threshold=%s decision=%s model=%s", proba, threshold, decision, active_version)
    return {
        "model_version": active_version,
        "score": proba,
        "threshold": threshold,
        "decision": decision,
    }


if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging()
    sample = {
        "application_completion_seconds": 45.0,
        "hour_of_day": 3,
        "email_domain_risk_score": 0.7,
        "account_age_days": 4,
        "num_applications_last_24h": 9,
        "ip_location_mismatch_km": 3200.0,
        "is_vpn_or_proxy": 1,
        "profile_trust_score": 0.2,
    }
    print(score(sample))

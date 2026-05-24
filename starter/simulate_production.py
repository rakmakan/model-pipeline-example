"""
Simulates production traffic: the live model (v1) scoring job applications
sampled from a slightly drifted distribution (v2), with delayed and
incomplete feedback coming back from downstream consumers (recruiters,
hiring managers, identity-verification reviewers).

You don't need to run this — the CSVs are checked in.
"""
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "model.pkl"

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


def main():
    rng = np.random.default_rng(7)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # The live model has been seeing v2-distribution traffic for ~90 days
    live_data = pd.read_csv(DATA_DIR / "applications_v2.csv").sample(
        n=6000, random_state=7
    ).reset_index(drop=True)

    scores = model.predict_proba(live_data[FEATURES])[:, 1]
    decisions = (scores >= 0.5).astype(int)

    start = datetime(2025, 9, 1)
    timestamps = [
        start + timedelta(minutes=int(rng.integers(0, 90 * 24 * 60)))
        for _ in range(len(live_data))
    ]
    timestamps.sort()

    predictions = pd.DataFrame({
        "prediction_id": [f"pred_{i:06d}" for i in range(len(live_data))],
        "timestamp": [t.isoformat() for t in timestamps],
        "model_version": "v1",
        **{f: live_data[f].values for f in FEATURES},
        "score": scores,
        "decision": decisions,
    })
    predictions.to_csv(DATA_DIR / "predictions.csv", index=False)
    print(f"predictions.csv: {len(predictions)} rows")

    # Feedback: only ~40% of predictions get labeled, with 7-30 day lag.
    # Older predictions are more likely to have feedback by now.
    cutoff = start + timedelta(days=90)
    feedback_rows = []
    for i, row in predictions.iterrows():
        pred_time = datetime.fromisoformat(row["timestamp"])
        lag = timedelta(days=int(rng.integers(7, 31)))
        feedback_time = pred_time + lag
        if feedback_time > cutoff:
            continue  # feedback hasn't arrived yet
        if rng.random() > 0.40:
            continue  # this application never got reviewed/labeled

        true_label = int(live_data.loc[i, "label"])
        # 4% label noise — reviewers occasionally mislabel
        if rng.random() < 0.04:
            verdict = "unclear"
        else:
            noisy = true_label if rng.random() > 0.04 else 1 - true_label
            verdict = "fraud" if noisy == 1 else "legit"

        feedback_rows.append({
            "prediction_id": row["prediction_id"],
            "feedback_timestamp": feedback_time.isoformat(),
            "verdict": verdict,
        })

    feedback = pd.DataFrame(feedback_rows)
    feedback.to_csv(DATA_DIR / "feedback.csv", index=False)
    print(f"feedback.csv:    {len(feedback)} rows ({len(feedback)/len(predictions):.1%} coverage)")
    print(f"  verdict distribution: {feedback['verdict'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

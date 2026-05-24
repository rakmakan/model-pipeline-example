"""
Generates the synthetic job-applicant-fraud datasets used by the starter repo.
You don't need to run this — the CSVs are checked in. This script exists
so the data is reproducible if anyone wants to regenerate it.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).parent / "data"


def make_dataset(n_rows: int, fraud_rate: float, drift: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # Legit applicants — thoughtful application, working hours, clean signals
    legit = pd.DataFrame({
        "application_completion_seconds": rng.lognormal(mean=6.5, sigma=0.5, size=n_legit),
        "hour_of_day": rng.choice(list(range(7, 23)), size=n_legit),
        "email_domain_risk_score": rng.beta(2, 8, size=n_legit),
        "account_age_days": rng.integers(30, 1000, size=n_legit),
        "num_applications_last_24h": rng.poisson(2, size=n_legit),
        "ip_location_mismatch_km": rng.exponential(15, size=n_legit),
        "is_vpn_or_proxy": rng.binomial(1, 0.05, size=n_legit),
        "profile_trust_score": rng.beta(8, 2, size=n_legit),
        "label": 0,
    })

    # Fraud applicants — bot-fast or copy-paste, odd hours, sketchy email,
    # brand-new accounts, spamming apps, IP far from claimed location, VPN,
    # thin/inconsistent profile signals.
    fraud = pd.DataFrame({
        "application_completion_seconds": rng.lognormal(mean=5.0, sigma=0.8, size=n_fraud),
        "hour_of_day": rng.choice(list(range(0, 6)) + list(range(22, 24)), size=n_fraud),
        "email_domain_risk_score": rng.beta(5, 3, size=n_fraud),
        "account_age_days": rng.integers(1, 200, size=n_fraud),
        "num_applications_last_24h": rng.poisson(8, size=n_fraud),
        "ip_location_mismatch_km": rng.exponential(150, size=n_fraud),
        "is_vpn_or_proxy": rng.binomial(1, 0.35, size=n_fraud),
        "profile_trust_score": rng.beta(2, 5, size=n_fraud),
        "label": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)

    # Drift: more legit applicants using VPNs (privacy-conscious / remote
    # workers behind corp VPNs), and applications generally take longer
    # (longer forms, more deliberate review). Simulates a real-world
    # distribution shift between v1 and v2.
    if drift:
        df["application_completion_seconds"] = df["application_completion_seconds"] * 1.4
        legit_mask = df["label"] == 0
        flip = rng.binomial(1, 0.08, size=legit_mask.sum()).astype(bool)
        df.loc[legit_mask, "is_vpn_or_proxy"] = np.where(
            flip, 1, df.loc[legit_mask, "is_vpn_or_proxy"]
        )

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)

    v1 = make_dataset(n_rows=8000, fraud_rate=0.04, drift=False, seed=1)
    v1.to_csv(DATA_DIR / "applications_v1.csv", index=False)
    print(f"applications_v1.csv: {len(v1)} rows, {v1['label'].mean():.3%} fraud")

    v2 = make_dataset(n_rows=12000, fraud_rate=0.035, drift=True, seed=2)
    v2.to_csv(DATA_DIR / "applications_v2.csv", index=False)
    print(f"applications_v2.csv: {len(v2)} rows, {v2['label'].mean():.3%} fraud")


if __name__ == "__main__":
    main()

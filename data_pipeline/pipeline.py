import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_pipeline.schema import detect_schema_version, validate

logger = logging.getLogger(__name__)

DATA_REGISTRY_PATH = Path("data_registry.json")
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


def _next_version(registry: dict) -> str:
    versions = list(registry.get("versions", {}).keys())
    if not versions:
        return "v1"
    nums = [int(v[1:]) for v in versions if v[1:].isdigit()]
    return f"v{max(nums) + 1}"


def run(input_path: str, version: str | None = None) -> str:
    input_path = Path(input_path)
    logger.info("Loading %s", input_path)
    df = pd.read_csv(input_path)
    logger.debug("Loaded %d rows, %d columns", len(df), len(df.columns))

    schema_version = detect_schema_version(df)
    validate(df, schema_version)
    logger.debug("Schema validation passed (version=%s)", schema_version)

    with open(DATA_REGISTRY_PATH) as f:
        registry = json.load(f)

    if version is None:
        version = _next_version(registry)
        logger.debug("Auto-assigned data version %s", version)

    if version in registry["versions"]:
        raise ValueError(f"Data version {version} already exists in registry.")

    out_dir = Path(f"data/{version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_val, test = train_test_split(df, test_size=SPLIT_RATIOS[2], stratify=df["label"], random_state=SPLIT_SEED)
    val_size = SPLIT_RATIOS[1] / (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val["label"], random_state=SPLIT_SEED)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False)
        logger.debug("Wrote %s (%d rows)", out_dir / f"{split_name}.csv", len(split_df))

    registry["versions"][version] = {
        "version": version,
        "input_file": str(input_path),
        "data_hash": _sha256(input_path),
        "schema_version": schema_version,
        "split_ratio": list(SPLIT_RATIOS),
        "split_seed": SPLIT_SEED,
        "splits": {
            "train": {"path": str(out_dir / "train.csv"), "rows": len(train), "fraud_rate": round(train["label"].mean(), 4)},
            "val":   {"path": str(out_dir / "val.csv"),   "rows": len(val),   "fraud_rate": round(val["label"].mean(), 4)},
            "test":  {"path": str(out_dir / "test.csv"),  "rows": len(test),  "fraud_rate": round(test["label"].mean(), 4)},
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    registry["latest"] = version

    with open(DATA_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(
        "Created data version %s: %d train / %d val / %d test (fraud rate: train=%.1f%%, val=%.1f%%, test=%.1f%%)",
        version, len(train), len(val), len(test),
        train["label"].mean() * 100, val["label"].mean() * 100, test["label"].mean() * 100,
    )
    return version

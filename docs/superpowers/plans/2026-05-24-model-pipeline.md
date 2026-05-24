# Model Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive one-shot training and accuracy-only promotion gate with a modular, versioned pipeline that trains, validates, evaluates, and promotes fraud detection models using a recall-first gate backed by retrospective replay.

**Architecture:** Three Python packages (`data_pipeline`, `replay_pipeline`, `model_pipeline`) provide clean module boundaries with explicit contracts. CLI scripts at root call into these packages. Models are packed as self-contained artifact folders. Two JSON registries (`data_registry.json`, `model_registry.json`) track versions and status. The registry is index/status only — all hyperparameters and metrics live in the artifact and eval report.

**Tech Stack:** Python 3.10+, scikit-learn, pandas, numpy, scipy (McNemar's test), pytest

---

## File Map

**New packages:**
- `data_pipeline/__init__.py`, `schema.py`, `pipeline.py`, `reader.py`
- `replay_pipeline/__init__.py`, `pipeline.py`, `reader.py`
- `model_pipeline/__init__.py`, `base_model.py`, `preprocessor.py`, `config.py`, `validator.py`, `evaluator.py`, `loader.py`

**New CLI scripts:**
- `run_data_pipe.py`, `run_replay_pipe.py`, `validate.py`, `evaluate.py`

**Rewritten scripts:**
- `train.py` (complete rewrite — adds LogisticRegressionModel class + pack_model)
- `promote.py` (complete rewrite — recall-first gate, --promote flag)
- `predict.py` (update to use loader.py)

**New data/config files:**
- `config.json`, `data_registry.json`, `model_registry.json`
- `data/raw/applications_v1.csv`, `data/raw/applications_v2.csv` (moved from `data/`)

**New tests:**
- `tests/__init__.py`, `tests/test_promote.py`

---

## Task 1: Repo Scaffold

**Files:**
- Create: `data/raw/` (move existing CSVs)
- Create: `data_pipeline/__init__.py`, `replay_pipeline/__init__.py`, `model_pipeline/__init__.py`, `tests/__init__.py`
- Create: `data_registry.json`, `model_registry.json`
- Modify: `requirements.txt`

- [ ] **Step 1: Move raw training CSVs to data/raw/**

```bash
mkdir -p data/raw
mv data/applications_v1.csv data/raw/applications_v1.csv
mv data/applications_v2.csv data/raw/applications_v2.csv
```

- [ ] **Step 2: Create package init files**

```bash
mkdir -p data_pipeline replay_pipeline model_pipeline tests
touch data_pipeline/__init__.py replay_pipeline/__init__.py model_pipeline/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create empty registries**

`data_registry.json`:
```json
{
  "latest": null,
  "versions": {},
  "replay": {
    "latest": null,
    "versions": {}
  }
}
```

`model_registry.json`:
```json
{
  "active": null,
  "models": {},
  "history": []
}
```

- [ ] **Step 4: Add scipy to requirements.txt**

```
pandas>=2.0
scikit-learn>=1.3
numpy>=1.24
scipy>=1.11
pytest>=7.0
```

- [ ] **Step 5: Install**

```bash
.venv/bin/pip install -r requirements.txt -q
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add data_pipeline/ replay_pipeline/ model_pipeline/ tests/ data_registry.json model_registry.json requirements.txt data/raw/
git commit -m "scaffold: add package structure, registries, move raw data"
```

---

## Task 2: Data Pipeline — Schema

**Files:**
- Create: `data_pipeline/schema.py`

- [ ] **Step 1: Write `data_pipeline/schema.py`**

```python
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
```

- [ ] **Step 2: Verify schema validates correctly**

```bash
.venv/bin/python -c "
import pandas as pd
from data_pipeline.schema import validate, detect_schema_version
df = pd.read_csv('data/raw/applications_v1.csv')
print('Detected:', detect_schema_version(df))
validate(df, 'v1')
print('Validation passed')
"
```

Expected output:
```
Detected: v1
Validation passed
```

- [ ] **Step 3: Commit**

```bash
git add data_pipeline/schema.py
git commit -m "feat(data_pipeline): add versioned schema validation with diff-style errors"
```

---

## Task 3: Data Pipeline — Pipeline and Reader

**Files:**
- Create: `data_pipeline/pipeline.py`, `data_pipeline/reader.py`

- [ ] **Step 1: Write `data_pipeline/pipeline.py`**

```python
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_pipeline.schema import detect_schema_version, validate

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
    df = pd.read_csv(input_path)
    schema_version = detect_schema_version(df)
    validate(df, schema_version)

    with open(DATA_REGISTRY_PATH) as f:
        registry = json.load(f)

    if version is None:
        version = _next_version(registry)

    if version in registry["versions"]:
        raise ValueError(f"Data version {version} already exists in registry.")

    out_dir = Path(f"data/{version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_val, test = train_test_split(df, test_size=SPLIT_RATIOS[2], stratify=df["label"], random_state=SPLIT_SEED)
    val_size = SPLIT_RATIOS[1] / (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val["label"], random_state=SPLIT_SEED)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False)

    def _stats(split_df):
        return {"path": str(out_dir / f"{split_name}.csv"), "rows": len(split_df), "fraud_rate": round(split_df["label"].mean(), 4)}

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

    print(f"Created data version {version}: {len(train)} train / {len(val)} val / {len(test)} test")
    return version
```

- [ ] **Step 2: Write `data_pipeline/reader.py`**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add data_pipeline/pipeline.py data_pipeline/reader.py
git commit -m "feat(data_pipeline): add stratified split pipeline and reader contract"
```

---

## Task 4: run_data_pipe.py — Run on v1 and v2

**Files:**
- Create: `run_data_pipe.py`

- [ ] **Step 1: Write `run_data_pipe.py`**

```python
"""
Validates, splits, and versions a raw training CSV.

Usage:
    python run_data_pipe.py --input data/raw/applications_v2.csv
    python run_data_pipe.py --input data/raw/applications_v2.csv --version v2
"""
import argparse
from data_pipeline.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--version", default=None, help="Version name (auto-increments if omitted)")
    args = parser.parse_args()
    run(args.input, args.version)
```

- [ ] **Step 2: Run on v1**

```bash
.venv/bin/python run_data_pipe.py --input data/raw/applications_v1.csv --version v1
```

Expected:
```
Created data version v1: 5600 train / 1200 val / 1200 test
```

- [ ] **Step 3: Run on v2**

```bash
.venv/bin/python run_data_pipe.py --input data/raw/applications_v2.csv --version v2
```

Expected:
```
Created data version v2: 8400 train / 1800 val / 1800 test
```

- [ ] **Step 4: Verify data_registry.json**

```bash
.venv/bin/python -c "import json; r=json.load(open('data_registry.json')); print('latest:', r['latest']); [print(v, r['versions'][v]['splits']['train']['rows']) for v in r['versions']]"
```

Expected:
```
latest: v2
v1 5600
v2 8400
```

- [ ] **Step 5: Commit**

```bash
git add run_data_pipe.py data/ data_registry.json
git commit -m "feat: add run_data_pipe.py, create data versions v1 and v2"
```

---

## Task 5: Replay Pipeline — Pipeline and Reader

**Files:**
- Create: `replay_pipeline/pipeline.py`, `replay_pipeline/reader.py`

- [ ] **Step 1: Write `replay_pipeline/pipeline.py`**

```python
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

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

    predictions = pd.read_csv(predictions_path)
    feedback = pd.read_csv(feedback_path)

    df = predictions.merge(feedback, on="prediction_id", how="inner")
    df = df[df["verdict"] != "unclear"].copy()
    df["label"] = (df["verdict"] == "fraud").astype(int)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    date_range = [df["timestamp"].min(), df["timestamp"].max()]
    weeks_spanned = (date_range[1] - date_range[0]).days / 7

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
    import json as _json
    (out_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))

    if "replay" not in registry:
        registry["replay"] = {"latest": None, "versions": {}}
    registry["replay"]["versions"][version] = {"path": str(out_dir / "replay.csv"), **meta}
    registry["replay"]["latest"] = version

    with open(DATA_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Created replay version {version}: {len(replay_df)} rows, {meta['fraud_rows']} fraud, {weeks_spanned:.1f} weeks")
    return version
```

- [ ] **Step 2: Write `replay_pipeline/reader.py`**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add replay_pipeline/pipeline.py replay_pipeline/reader.py
git commit -m "feat(replay_pipeline): add versioned replay dataset pipeline and reader"
```

---

## Task 6: run_replay_pipe.py — Run on Production Data

**Files:**
- Create: `run_replay_pipe.py`

- [ ] **Step 1: Write `run_replay_pipe.py`**

```python
"""
Joins predictions and feedback into a versioned replay dataset for gate evaluation.

Usage:
    python run_replay_pipe.py --predictions data/predictions.csv --feedback data/feedback.csv
"""
import argparse
from replay_pipeline.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    run(args.predictions, args.feedback, args.version)
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python run_replay_pipe.py --predictions data/predictions.csv --feedback data/feedback.csv
```

Expected:
```
Created replay version v1: 1829 rows, 145 fraud, 12.9 weeks
```

- [ ] **Step 3: Commit**

```bash
git add run_replay_pipe.py data/replay/ data_registry.json
git commit -m "feat: add run_replay_pipe.py, create replay version v1"
```

---

## Task 7: Model Pipeline — Base Model and Preprocessor

**Files:**
- Create: `model_pipeline/base_model.py`, `model_pipeline/preprocessor.py`

- [ ] **Step 1: Write `model_pipeline/base_model.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model. Implementations must call self.preprocess() internally."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each row. Shape: (n,)"""

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features. Called inside fit() and predict_proba(). Never called directly by consumers."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model weights and preprocessor state to artifact directory."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a fully ready model from an artifact directory."""
```

- [ ] **Step 2: Write `model_pipeline/preprocessor.py`**

```python
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self._scaler.fit_transform(X)
        self._fitted = True
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")
        scaled = self._scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def save(self, path: Path) -> None:
        with open(path / "preprocessor.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        obj = cls.__new__(cls)
        with open(path / "preprocessor.pkl", "rb") as f:
            obj._scaler = pickle.load(f)
        obj._fitted = True
        return obj
```

- [ ] **Step 3: Commit**

```bash
git add model_pipeline/base_model.py model_pipeline/preprocessor.py
git commit -m "feat(model_pipeline): add BaseModel ABC and Preprocessor"
```

---

## Task 8: Config Schema

**Files:**
- Create: `config.json`, `model_pipeline/config.py`

- [ ] **Step 1: Write `config.json`**

```json
{
  "model": {
    "type": "LogisticRegression",
    "hyperparameters": {
      "max_iter": 1000,
      "solver": "lbfgs",
      "class_weight": "balanced"
    }
  },
  "preprocessor": {
    "type": "StandardScaler"
  },
  "training": {
    "split_seed": 42,
    "target_recall": 0.80
  }
}
```

- [ ] **Step 2: Write `model_pipeline/config.py`**

```python
import json
from pathlib import Path

VALID_MODEL_TYPES = {"LogisticRegression"}
VALID_PREPROCESSOR_TYPES = {"StandardScaler"}

DEFAULT_CONFIG_PATH = Path("config.json")


def load_and_validate(path: Path | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        config = json.load(f)

    errors = []

    model_type = config.get("model", {}).get("type")
    if model_type not in VALID_MODEL_TYPES:
        errors.append(f"model.type must be one of {VALID_MODEL_TYPES}, got: {model_type!r}")

    preprocessor_type = config.get("preprocessor", {}).get("type")
    if preprocessor_type not in VALID_PREPROCESSOR_TYPES:
        errors.append(f"preprocessor.type must be one of {VALID_PREPROCESSOR_TYPES}, got: {preprocessor_type!r}")

    training = config.get("training", {})
    if "target_recall" not in training:
        errors.append("training.target_recall is required")
    elif not (0.0 < training["target_recall"] <= 1.0):
        errors.append(f"training.target_recall must be in (0, 1], got: {training['target_recall']}")

    if "split_seed" not in training:
        errors.append("training.split_seed is required")

    hparams = config.get("model", {}).get("hyperparameters", {})
    if "max_iter" in hparams and not isinstance(hparams["max_iter"], int):
        errors.append("model.hyperparameters.max_iter must be an integer")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
```

- [ ] **Step 3: Verify config validates**

```bash
.venv/bin/python -c "from model_pipeline.config import load_and_validate; c = load_and_validate(); print('Config valid. target_recall:', c['training']['target_recall'])"
```

Expected:
```
Config valid. target_recall: 0.8
```

- [ ] **Step 4: Commit**

```bash
git add config.json model_pipeline/config.py
git commit -m "feat(model_pipeline): add config schema with validation"
```

---

## Task 9: Rewrite train.py

**Files:**
- Modify: `train.py` (full rewrite)

This is the most important file. It defines `LogisticRegressionModel`, trains it, and packs the artifact.

- [ ] **Step 1: Write `train.py`**

```python
"""
Trains a fraud detection model, packs it as a self-contained artifact, and registers it as a candidate.

Usage:
    python train.py --data-version v2 --model-version v2
    python train.py --data-version v2 --model-version v2 --config configs/high_recall.json
    python train.py --config models/v2/train_config.json   # reproduce exact run
"""
import argparse
import inspect
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_pipeline.reader import FEATURES, load_train
from model_pipeline.base_model import BaseModel
from model_pipeline.config import load_and_validate
from model_pipeline.preprocessor import Preprocessor

MODEL_REGISTRY_PATH = Path("model_registry.json")


class LogisticRegressionModel(BaseModel):
    def __init__(self, hyperparameters: dict):
        self.preprocessor = Preprocessor()
        self.model = LogisticRegression(**hyperparameters)
        self._hyperparameters = hyperparameters

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(X)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.preprocess(X))[:, 1]

    def save(self, path: Path) -> None:
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        self.preprocessor.save(path)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        obj = cls.__new__(cls)
        with open(path / "model.pkl", "rb") as f:
            obj.model = pickle.load(f)
        obj.preprocessor = Preprocessor.load(path)
        obj._hyperparameters = {}
        return obj


def _pack_artifact(model: LogisticRegressionModel, artifact_path: Path, data_version: str, config: dict) -> None:
    artifact_path.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)

    # Snapshot class definitions from source
    shutil.copy("model_pipeline/base_model.py", artifact_path / "base_model.py")
    shutil.copy("model_pipeline/preprocessor.py", artifact_path / "preprocessor.py")

    # Extract LogisticRegressionModel class source and write model_class.py
    class_source = inspect.getsource(LogisticRegressionModel)
    model_class_content = (
        "import pickle\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from base_model import BaseModel\n"
        "from preprocessor import Preprocessor\n"
        "from sklearn.linear_model import LogisticRegression\n\n\n"
        + class_source
    )
    (artifact_path / "model_class.py").write_text(model_class_content)

    metadata = {
        "model_class_name": "LogisticRegressionModel",
        "schema_version": "v1",
        "data_version": data_version,
        "features": FEATURES,
        "threshold": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (artifact_path / "train_config.json").write_text(json.dumps(config, indent=2))


def _register_candidate(version: str, data_version: str, artifact_path: Path) -> None:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    if version in registry["models"]:
        raise ValueError(f"Model version {version} already exists in registry.")

    registry["models"][version] = {
        "version": version,
        "status": "candidate",
        "data_version": data_version,
        "artifact_path": str(artifact_path),
        "eval_report_path": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "promoted_at": None,
        "promotion_gate": None,
    }

    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", help="Data version from data_registry.json")
    parser.add_argument("--model-version", help="Version name for the new model artifact")
    parser.add_argument("--config", default=None, help="Path to config JSON (defaults to config.json)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    # Reproduce mode: load everything from a saved train_config.json
    if args.config and Path(args.config).name == "train_config.json":
        saved = json.loads(Path(args.config).read_text())
        data_version = saved.get("_data_version", args.data_version)
        model_version = args.model_version or saved.get("_model_version")
        config = load_and_validate(config_path)
    else:
        if not args.data_version or not args.model_version:
            parser.error("--data-version and --model-version are required unless reproducing from train_config.json")
        data_version = args.data_version
        model_version = args.model_version
        config = load_and_validate(config_path)

    X_train, y_train = load_train(data_version)
    fraud_rate = y_train.mean()
    print(f"Training data: {len(X_train)} rows, {fraud_rate:.3%} fraud")

    hparams = config["model"]["hyperparameters"]
    model = LogisticRegressionModel(hparams)
    model.fit(X_train, y_train)

    artifact_path = Path(f"models/{model_version}")
    # Embed data/model version into config snapshot for reproduction
    config_snapshot = dict(config)
    config_snapshot["_data_version"] = data_version
    config_snapshot["_model_version"] = model_version
    _pack_artifact(model, artifact_path, data_version, config_snapshot)
    _register_candidate(model_version, data_version, artifact_path)

    print(f"Packed artifact → {artifact_path}")
    print(f"Registered {model_version} as candidate in model_registry.json")
    print(f"Next step: python validate.py --model-version {model_version}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train v1**

```bash
.venv/bin/python train.py --data-version v1 --model-version v1
```

Expected:
```
Training data: 5600 rows, 4.018% fraud
Packed artifact → models/v1
Registered v1 as candidate in model_registry.json
Next step: python validate.py --model-version v1
```

- [ ] **Step 3: Verify artifact contents**

```bash
ls models/v1/
```

Expected: `base_model.py  metadata.json  model.pkl  model_class.py  preprocessor.pkl  preprocessor.py  train_config.json`

- [ ] **Step 4: Commit**

```bash
git add train.py models/v1/ model_registry.json
git commit -m "feat: rewrite train.py with LogisticRegressionModel, pack artifact, register candidate"
```

---

## Task 10: Model Pipeline — Loader

**Files:**
- Create: `model_pipeline/loader.py`

- [ ] **Step 1: Write `model_pipeline/loader.py`**

```python
import importlib.util
import json
import sys
from pathlib import Path

from model_pipeline.base_model import BaseModel


def load_model(version: str) -> BaseModel:
    artifact_path = Path(f"models/{version}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}. Run train.py first.")

    metadata = json.loads((artifact_path / "metadata.json").read_text())
    class_name = metadata["model_class_name"]

    # Load model_class.py from the artifact — not from current codebase —
    # so code changes after training never affect old artifacts.
    sys.path.insert(0, str(artifact_path))
    try:
        spec = importlib.util.spec_from_file_location(
            f"artifact_{version}_model_class", artifact_path / "model_class.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ModelClass = getattr(module, class_name)
    finally:
        sys.path.pop(0)

    model = ModelClass.load(artifact_path)
    model._metadata = metadata
    return model


def get_threshold(version: str) -> float:
    artifact_path = Path(f"models/{version}")
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    threshold = metadata.get("threshold")
    if threshold is None:
        raise ValueError(f"Model {version} has no threshold set. Run validate.py first.")
    return threshold


def get_features(version: str) -> list[str]:
    artifact_path = Path(f"models/{version}")
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    return metadata["features"]
```

- [ ] **Step 2: Verify loader works**

```bash
.venv/bin/python -c "
from model_pipeline.loader import load_model
m = load_model('v1')
print('Loaded:', type(m).__name__)
import pandas as pd
sample = pd.DataFrame([{
    'application_completion_seconds': 45.0,
    'hour_of_day': 3,
    'email_domain_risk_score': 0.7,
    'account_age_days': 4,
    'num_applications_last_24h': 9,
    'ip_location_mismatch_km': 3200.0,
    'is_vpn_or_proxy': 1,
    'profile_trust_score': 0.2,
}])
print('Score:', m.predict_proba(sample)[0])
"
```

Expected:
```
Loaded: LogisticRegressionModel
Score: <float between 0 and 1>
```

- [ ] **Step 3: Commit**

```bash
git add model_pipeline/loader.py
git commit -m "feat(model_pipeline): add loader — loads self-contained artifact, not current codebase"
```

---

## Task 11: Validator — Threshold Optimisation

**Files:**
- Create: `model_pipeline/validator.py`, `validate.py`

- [ ] **Step 1: Write `model_pipeline/validator.py`**

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.reader import load_val
from model_pipeline.loader import load_model


def find_threshold(y_true: pd.Series, y_proba: np.ndarray, target_recall: float) -> float | None:
    thresholds = np.arange(0.05, 0.96, 0.01)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if recall >= target_recall:
            return round(float(t), 4)
    return None


def run(model_version: str, model_registry_path: Path = Path("model_registry.json")) -> float:
    with open(model_registry_path) as f:
        registry = json.load(f)

    entry = registry["models"].get(model_version)
    if entry is None:
        raise ValueError(f"Model version {model_version} not found in registry.")

    data_version = entry["data_version"]
    artifact_path = Path(f"models/{model_version}")
    config = json.loads((artifact_path / "train_config.json").read_text())
    target_recall = config["training"]["target_recall"]

    model = load_model(model_version)
    X_val, y_val = load_val(data_version)
    y_proba = model.predict_proba(X_val)

    threshold = find_threshold(y_val, y_proba, target_recall)

    if threshold is None:
        best_t = 0.05
        best_recall = 0.0
        thresholds = np.arange(0.05, 0.96, 0.01)
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tp = ((y_pred == 1) & (y_val == 1)).sum()
            fn = ((y_pred == 0) & (y_val == 1)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall > best_recall:
                best_recall, best_t = recall, t
        raise ValueError(
            f"No threshold achieves target recall {target_recall} on val set. "
            f"Best achievable: {best_recall:.3f} at threshold {best_t:.2f}"
        )

    # Write threshold to artifact metadata
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    metadata["threshold"] = threshold
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Threshold set to {threshold} (achieves recall ≥ {target_recall} on val set)")
    return threshold
```

- [ ] **Step 2: Write `validate.py`**

```python
"""
Finds the optimal operating threshold for a candidate model on the validation set.

Usage:
    python validate.py --model-version v1
"""
import argparse
from model_pipeline.validator import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    run(args.model_version)
```

- [ ] **Step 3: Run on v1**

```bash
.venv/bin/python validate.py --model-version v1
```

Expected:
```
Threshold set to <float> (achieves recall ≥ 0.8 on val set)
```

- [ ] **Step 4: Verify threshold written to metadata**

```bash
.venv/bin/python -c "import json; m=json.load(open('models/v1/metadata.json')); print('threshold:', m['threshold'])"
```

Expected: `threshold: <some float like 0.32>`

- [ ] **Step 5: Commit**

```bash
git add model_pipeline/validator.py validate.py models/v1/metadata.json
git commit -m "feat: add validator — sweep threshold to hit target recall on val set"
```

---

## Task 12: Evaluator — Test Set + Replay Report

**Files:**
- Create: `model_pipeline/evaluator.py`, `evaluate.py`, `reports/`

- [ ] **Step 1: Write `model_pipeline/evaluator.py`**

```python
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_pipeline.reader import load_test
from model_pipeline.loader import get_threshold, load_model
from replay_pipeline.reader import load_replay

MODEL_REGISTRY_PATH = Path("model_registry.json")


def _metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc = None
    return {"recall": round(recall, 4), "fpr": round(fpr, 4), "precision": round(precision, 4), "f1": round(f1, 4), "auc": round(auc, 4) if auc else None}


def run(model_version: str, replay_version: str) -> dict:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    entry = registry["models"].get(model_version)
    if entry is None:
        raise ValueError(f"Model version {model_version} not in registry.")

    threshold = get_threshold(model_version)
    data_version = entry["data_version"]
    model = load_model(model_version)

    # Test set evaluation
    X_test, y_test = load_test(data_version)
    y_proba_test = model.predict_proba(X_test)
    test_metrics = _metrics(y_test, y_proba_test, threshold)

    # Retrospective replay evaluation
    X_replay, y_replay, replay_meta = load_replay(replay_version)
    y_proba_replay = model.predict_proba(X_replay)
    replay_metrics = _metrics(y_replay, y_proba_replay, threshold)

    report = {
        "model_version": model_version,
        "data_version": data_version,
        "replay_version": replay_version,
        "threshold": threshold,
        "test": test_metrics,
        "replay": replay_metrics,
        "replay_meta": replay_meta,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    Path("reports").mkdir(exist_ok=True)
    report_path = Path(f"reports/eval_{model_version}.json")
    report_path.write_text(json.dumps(report, indent=2))

    # Update registry
    registry["models"][model_version]["eval_report_path"] = str(report_path)
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    # Console output
    print(f"\n=== EVALUATION REPORT: {model_version} ===")
    print(f"Threshold : {threshold}")
    print(f"\nTest set ({data_version}):")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  FPR       : {test_metrics['fpr']:.4f}")
    print(f"  AUC       : {test_metrics['auc']}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"\nReplay ({replay_version}, {replay_meta['rows']} rows, {replay_meta['weeks_spanned']} weeks):")
    print(f"  Recall    : {replay_metrics['recall']:.4f}")
    print(f"  FPR       : {replay_metrics['fpr']:.4f}")
    print(f"\nReport saved → {report_path}")
    return report
```

- [ ] **Step 2: Write `evaluate.py`**

```python
"""
Evaluates a candidate model on its test set and the retrospective replay dataset.

Usage:
    python evaluate.py --model-version v1 --replay-version v1
"""
import argparse
from model_pipeline.evaluator import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--replay-version", required=True)
    args = parser.parse_args()
    run(args.model_version, args.replay_version)
```

- [ ] **Step 3: Run on v1**

```bash
.venv/bin/python evaluate.py --model-version v1 --replay-version v1
```

Expected: evaluation table printed, `reports/eval_v1.json` created.

- [ ] **Step 4: Commit**

```bash
git add model_pipeline/evaluator.py evaluate.py reports/eval_v1.json model_registry.json
git commit -m "feat: add evaluator — test set + retrospective replay metrics, JSON report"
```

---

## Task 13: Gate Tests (TDD — Write First)

**Files:**
- Create: `tests/test_promote.py`

Write the tests before `promote.py` exists. They should fail at import.

- [ ] **Step 1: Write `tests/test_promote.py`**

```python
"""
Two focused tests for the promotion gate logic.

Test 1: All conditions pass — candidate improves recall, FPR within guardrail → PROMOTE
Test 2: Recall improves but FPR guardrail exceeded → REJECT with C2 failure identified
"""
import numpy as np
import pytest
from promote import GateResult, run_gate


def _make_predictions(n_legit: int, n_fraud: int, active_fp: int, active_tp: int, cand_fp: int, cand_tp: int):
    """Build synthetic (y_true, y_pred_active, y_pred_candidate) arrays."""
    y_true = np.array([0] * n_legit + [1] * n_fraud)

    active = np.zeros(n_legit + n_fraud, dtype=int)
    active[:active_fp] = 1                              # false positives
    active[n_legit:n_legit + active_tp] = 1             # true positives

    cand = np.zeros(n_legit + n_fraud, dtype=int)
    cand[:cand_fp] = 1                                   # false positives
    cand[n_legit:n_legit + cand_tp] = 1                  # true positives

    return y_true, active, cand


def test_all_conditions_pass_returns_promote():
    """Candidate improves recall (0.75→0.875), FPR increase 7pp (within 10pp guardrail) → PROMOTE."""
    # 210 legit, 40 fraud
    # Active: 10 FP (FPR=0.048), 30 TP (recall=0.75)
    # Candidate: 25 FP (FPR=0.119), 35 TP (recall=0.875)
    y_true, y_pred_active, y_pred_cand = _make_predictions(
        n_legit=210, n_fraud=40,
        active_fp=10, active_tp=30,
        cand_fp=25, cand_tp=35,
    )
    result = run_gate(y_true, y_pred_active, y_pred_cand)
    assert result.verdict == "PROMOTE", f"Expected PROMOTE, got {result.verdict}\n{result}"
    assert result.c1_pass is True
    assert result.c2_pass is True
    assert result.c3_pass is True


def test_fpr_guardrail_exceeded_returns_reject():
    """Candidate improves recall but FPR increase 13pp exceeds 10pp guardrail → REJECT on C2."""
    # Active: 10 FP (FPR=0.048), 30 TP (recall=0.75)
    # Candidate: 37 FP (FPR=0.176), 38 TP (recall=0.95) — 12.8pp FPR increase
    y_true, y_pred_active, y_pred_cand = _make_predictions(
        n_legit=210, n_fraud=40,
        active_fp=10, active_tp=30,
        cand_fp=37, cand_tp=38,
    )
    result = run_gate(y_true, y_pred_active, y_pred_cand)
    assert result.verdict == "REJECT"
    assert result.c1_pass is True, "C1 (recall) should pass"
    assert result.c2_pass is False, "C2 (FPR guardrail) should fail"
    assert "C2" in result.failure_reason, f"Failure reason should mention C2, got: {result.failure_reason}"
```

- [ ] **Step 2: Run tests — verify they fail at import (promote not yet written)**

```bash
.venv/bin/python -m pytest tests/test_promote.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'GateResult' from 'promote'` or `ModuleNotFoundError`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_promote.py
git commit -m "test(promote): add TDD gate tests — all-pass and FPR-guardrail cases"
```

---

## Task 14: Rewrite promote.py — Gate Logic

**Files:**
- Modify: `promote.py` (full rewrite)

- [ ] **Step 1: Write `promote.py`**

```python
"""
Promotion gate: compares candidate model against active model on retrospective replay.
All three conditions must pass for auto-promotion.

Usage:
    python promote.py --candidate v2 --replay-version v1           # dry run
    python promote.py --candidate v2 --replay-version v1 --promote # execute promotion
    python promote.py --candidate v1 --bootstrap                   # promote first model without gate
"""
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from model_pipeline.loader import get_threshold, load_model
from replay_pipeline.reader import load_replay

MODEL_REGISTRY_PATH = Path("model_registry.json")
RECALL_FLOOR = 0.80
FPR_GUARDRAIL = 0.10
MCNEMAR_P_THRESHOLD = 0.05
MIN_EXAMPLES = 200
MIN_DISAGREEMENTS = 30


@dataclass
class GateResult:
    verdict: str          # PROMOTE | REJECT | INSUFFICIENT_DATA
    c1_pass: bool
    c2_pass: bool
    c3_pass: bool
    active_recall: float
    active_fpr: float
    candidate_recall: float
    candidate_fpr: float
    recall_delta: float
    fpr_delta: float
    p_value: float | None
    n_examples: int
    n_disagreements: int
    failure_reason: str

    def __str__(self):
        lines = [
            f"\n  Active recall={self.active_recall:.4f}  fpr={self.active_fpr:.4f}",
            f"  Cand  recall={self.candidate_recall:.4f}  fpr={self.candidate_fpr:.4f}",
            f"  Delta recall={self.recall_delta:+.4f}  fpr={self.fpr_delta:+.4f}",
            f"  C1={'PASS' if self.c1_pass else 'FAIL'}  C2={'PASS' if self.c2_pass else 'FAIL'}  C3={'PASS' if self.c3_pass else 'FAIL'}",
            f"  McNemar p={self.p_value}  n={self.n_examples}  disagreements={self.n_disagreements}",
            f"  Verdict: {self.verdict}",
        ]
        if self.failure_reason:
            lines.append(f"  Reason: {self.failure_reason}")
        return "\n".join(lines)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return round(recall, 4), round(fpr, 4)


def _mcnemar_p(y_pred_active: np.ndarray, y_pred_cand: np.ndarray) -> tuple[float, int]:
    b = int(((y_pred_active == 1) & (y_pred_cand == 0)).sum())  # active correct, cand wrong
    c = int(((y_pred_active == 0) & (y_pred_cand == 1)).sum())  # cand correct, active wrong
    n_disagreements = b + c
    if n_disagreements == 0:
        return 1.0, 0
    # McNemar's test with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(chi2_stat, df=1))
    return round(p_value, 4), n_disagreements


def run_gate(
    y_true: np.ndarray,
    y_pred_active: np.ndarray,
    y_pred_cand: np.ndarray,
) -> GateResult:
    n_examples = len(y_true)
    active_recall, active_fpr = _compute_metrics(y_true, y_pred_active)
    cand_recall, cand_fpr = _compute_metrics(y_true, y_pred_cand)
    recall_delta = round(cand_recall - active_recall, 4)
    fpr_delta = round(cand_fpr - active_fpr, 4)
    p_value, n_disagreements = _mcnemar_p(y_pred_active, y_pred_cand)

    if n_examples < MIN_EXAMPLES:
        return GateResult(
            verdict="INSUFFICIENT_DATA",
            c1_pass=False, c2_pass=False, c3_pass=False,
            active_recall=active_recall, active_fpr=active_fpr,
            candidate_recall=cand_recall, candidate_fpr=cand_fpr,
            recall_delta=recall_delta, fpr_delta=fpr_delta,
            p_value=p_value, n_examples=n_examples, n_disagreements=n_disagreements,
            failure_reason=f"Only {n_examples} examples — need at least {MIN_EXAMPLES}",
        )

    c1_pass = cand_recall >= active_recall and cand_recall >= RECALL_FLOOR
    c2_pass = fpr_delta <= FPR_GUARDRAIL
    c3_pass = p_value < MCNEMAR_P_THRESHOLD and n_disagreements >= MIN_DISAGREEMENTS

    failures = []
    if not c1_pass:
        if cand_recall < RECALL_FLOOR:
            failures.append(f"C1: candidate recall {cand_recall} below absolute floor {RECALL_FLOOR}")
        else:
            failures.append(f"C1: candidate recall {cand_recall} < active recall {active_recall}")
    if not c2_pass:
        failures.append(f"C2: FPR increase {fpr_delta:.4f} exceeds {FPR_GUARDRAIL} guardrail")
    if not c3_pass:
        if n_disagreements < MIN_DISAGREEMENTS:
            failures.append(f"C3: only {n_disagreements} disagreements — need at least {MIN_DISAGREEMENTS}")
        else:
            failures.append(f"C3: McNemar p={p_value} >= {MCNEMAR_P_THRESHOLD}")

    verdict = "PROMOTE" if (c1_pass and c2_pass and c3_pass) else "REJECT"
    return GateResult(
        verdict=verdict,
        c1_pass=c1_pass, c2_pass=c2_pass, c3_pass=c3_pass,
        active_recall=active_recall, active_fpr=active_fpr,
        candidate_recall=cand_recall, candidate_fpr=cand_fpr,
        recall_delta=recall_delta, fpr_delta=fpr_delta,
        p_value=p_value, n_examples=n_examples, n_disagreements=n_disagreements,
        failure_reason="; ".join(failures),
    )


def _print_gate_report(candidate_version: str, active_version: str, replay_version: str,
                       replay_meta: dict, result: GateResult) -> None:
    print(f"\n{'='*50}")
    print(f"PROMOTION GATE REPORT")
    print(f"{'='*50}")
    print(f"Candidate : {candidate_version}")
    print(f"Active    : {active_version}")
    print(f"Replay    : {replay_version} — {replay_meta['rows']} rows | "
          f"{replay_meta['date_range'][0][:10]} – {replay_meta['date_range'][1][:10]} "
          f"({replay_meta['weeks_spanned']} weeks)")
    print(f"\n{'':20} {'Active':>12} {'Candidate':>12} {'Delta':>8}")
    print(f"{'Recall':20} {result.active_recall:>12.4f} {result.candidate_recall:>12.4f} {result.recall_delta:>+8.4f}")
    print(f"{'FPR':20} {result.active_fpr:>12.4f} {result.candidate_fpr:>12.4f} {result.fpr_delta:>+8.4f}")
    print(f"\nGate conditions:")
    print(f"  [{'PASS' if result.c1_pass else 'FAIL'}] C1: recall {result.candidate_recall:.4f} >= active {result.active_recall:.4f}, floor {RECALL_FLOOR}")
    print(f"  [{'PASS' if result.c2_pass else 'FAIL'}] C2: FPR increase {result.fpr_delta:+.4f} within {FPR_GUARDRAIL} guardrail")
    print(f"  [{'PASS' if result.c3_pass else 'FAIL'}] C3: McNemar p={result.p_value} | {result.n_examples} examples | {result.n_disagreements} disagreements")
    print(f"\nVerdict: {result.verdict}")
    if result.failure_reason:
        print(f"Reason : {result.failure_reason}")


def _do_promote(candidate_version: str, active_version: str, replay_version: str, result: GateResult) -> None:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    now = datetime.now().isoformat(timespec="seconds")

    # Retire active model
    if active_version and active_version in registry["models"]:
        registry["models"][active_version]["status"] = "retired"
        registry["models"][active_version]["retired_at"] = now
        registry["history"].append(registry["models"][active_version])

    # Promote candidate
    registry["models"][candidate_version]["status"] = "active"
    registry["models"][candidate_version]["promoted_at"] = now
    registry["models"][candidate_version]["promotion_gate"] = {
        "verdict": result.verdict,
        "c1_pass": result.c1_pass,
        "c2_pass": result.c2_pass,
        "c3_pass": result.c3_pass,
        "active_recall": result.active_recall,
        "active_fpr": result.active_fpr,
        "candidate_recall": result.candidate_recall,
        "candidate_fpr": result.candidate_fpr,
        "p_value": result.p_value,
        "n_examples": result.n_examples,
        "replay_version": replay_version,
        "promoted_at": now,
    }
    registry["active"] = candidate_version

    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n✓ Promoted {candidate_version} → active. Previous active ({active_version}) → history.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--replay-version", default=None)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Promote first model without running the gate (no active model to compare against)")
    args = parser.parse_args()

    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    entry = registry["models"].get(args.candidate)
    if entry is None:
        raise SystemExit(f"Candidate {args.candidate} not in model_registry.json. Run train.py first.")

    if entry.get("eval_report_path") is None and not args.bootstrap:
        raise SystemExit(
            f"Candidate {args.candidate} has no eval report. Run:\n"
            f"  python evaluate.py --model-version {args.candidate} --replay-version <version>"
        )

    if args.bootstrap:
        if args.promote:
            _do_promote(args.candidate, active_version=None, replay_version=None,
                       result=GateResult("BOOTSTRAP", True, True, True, 0, 0, 0, 0, 0, 0, None, 0, 0, ""))
            print(f"Bootstrap promoted {args.candidate} as initial active model.")
        else:
            print(f"Bootstrap mode: {args.candidate} would be set as initial active model. Rerun with --promote to execute.")
        return

    active_version = registry.get("active")
    if active_version is None:
        raise SystemExit("No active model found. Use --bootstrap to promote the first model.")

    # Prerequisite: threshold must be set
    artifact_path = Path(f"models/{args.candidate}")
    metadata = json.loads((artifact_path / "metadata.json").read_text())
    if metadata.get("threshold") is None:
        raise SystemExit(
            f"Candidate {args.candidate} has no threshold. Run:\n"
            f"  python validate.py --model-version {args.candidate}"
        )

    replay_version = args.replay_version
    if replay_version is None:
        dr = json.loads(Path("data_registry.json").read_text())
        replay_version = dr.get("replay", {}).get("latest")
        if replay_version is None:
            raise SystemExit("No replay version found. Run run_replay_pipe.py first.")
        print(f"Using latest replay version: {replay_version}")

    X_replay, y_replay, replay_meta = load_replay(replay_version)

    active_model = load_model(active_version)
    active_threshold = get_threshold(active_version)
    candidate_model = load_model(args.candidate)
    candidate_threshold = get_threshold(args.candidate)

    y_pred_active = (active_model.predict_proba(X_replay) >= active_threshold).astype(int)
    y_pred_cand = (candidate_model.predict_proba(X_replay) >= candidate_threshold).astype(int)

    result = run_gate(y_replay.values, y_pred_active, y_pred_cand)
    _print_gate_report(args.candidate, active_version, replay_version, replay_meta, result)

    if args.promote:
        if result.verdict != "PROMOTE":
            raise SystemExit(f"\nCannot promote: gate did not pass. Verdict: {result.verdict}")
        _do_promote(args.candidate, active_version, replay_version, result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate tests — they should now pass**

```bash
.venv/bin/python -m pytest tests/test_promote.py -v
```

Expected:
```
tests/test_promote.py::test_all_conditions_pass_returns_promote PASSED
tests/test_promote.py::test_fpr_guardrail_exceeded_returns_reject PASSED
2 passed
```

- [ ] **Step 3: Bootstrap v1 as initial active model**

```bash
.venv/bin/python promote.py --candidate v1 --bootstrap --promote
```

Expected:
```
Bootstrap promoted v1 as initial active model.
```

- [ ] **Step 4: Commit**

```bash
git add promote.py model_registry.json
git commit -m "feat: rewrite promote.py — recall-first gate, McNemar test, --promote flag; all gate tests pass"
```

---

## Task 15: Train v2 and Run Full Pipeline

End-to-end: train v2, validate, evaluate, run gate against v1.

- [ ] **Step 1: Train v2**

```bash
.venv/bin/python train.py --data-version v2 --model-version v2
```

Expected:
```
Training data: 8400 rows, ...% fraud
Packed artifact → models/v2
Registered v2 as candidate in model_registry.json
```

- [ ] **Step 2: Validate v2**

```bash
.venv/bin/python validate.py --model-version v2
```

Expected: `Threshold set to <float> (achieves recall ≥ 0.8 on val set)`

- [ ] **Step 3: Evaluate v2**

```bash
.venv/bin/python evaluate.py --model-version v2 --replay-version v1
```

Expected: evaluation table with test and replay metrics printed.

- [ ] **Step 4: Run gate (dry run)**

```bash
.venv/bin/python promote.py --candidate v2 --replay-version v1
```

Expected: full gate report printed with PROMOTE or REJECT verdict.

- [ ] **Step 5: Promote if gate passes (or note outcome)**

If verdict is PROMOTE:
```bash
.venv/bin/python promote.py --candidate v2 --replay-version v1 --promote
```

If verdict is REJECT: note which conditions failed — this is expected output for the README and video.

- [ ] **Step 6: Commit**

```bash
git add models/v2/ model_registry.json reports/eval_v2.json
git commit -m "feat: train v2, validate, evaluate; run promotion gate against v1"
```

---

## Task 16: Update predict.py

**Files:**
- Modify: `predict.py`

- [ ] **Step 1: Rewrite `predict.py` to use loader**

```python
"""
Scores a job application using the currently active model.

Usage:
    python predict.py
"""
import json
from pathlib import Path

import pandas as pd

from model_pipeline.loader import get_features, get_threshold, load_model

MODEL_REGISTRY_PATH = Path("model_registry.json")


def score(application: dict) -> dict:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)

    active_version = registry.get("active")
    if active_version is None:
        raise RuntimeError("No active model. Run promote.py --bootstrap first.")

    model = load_model(active_version)
    threshold = get_threshold(active_version)
    features = get_features(active_version)

    X = pd.DataFrame([application])[features]
    proba = float(model.predict_proba(X)[0])
    return {
        "model_version": active_version,
        "score": proba,
        "threshold": threshold,
        "decision": "block" if proba >= threshold else "allow",
    }


if __name__ == "__main__":
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
```

- [ ] **Step 2: Run to verify**

```bash
.venv/bin/python predict.py
```

Expected: `{'model_version': 'v1' or 'v2', 'score': <float>, 'threshold': <float>, 'decision': 'block' or 'allow'}`

- [ ] **Step 3: Commit**

```bash
git add predict.py
git commit -m "feat: update predict.py to use loader — reads active model from registry"
```

---

## Task 17: Cleanup — Remove Starter Files

- [ ] **Step 1: Remove starter registry and old model artifact**

```bash
git rm registry.json
git rm models/model.pkl
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove starter registry.json and models/model.pkl — replaced by versioned pipeline"
```

---

## Self-Review Checklist

After writing this plan, checking spec coverage:

| Spec requirement | Task |
|---|---|
| Versioned data splits (70/15/15, stratified) | Task 3 |
| Data schema validation with diff-style errors | Task 2 |
| Schema versioning + evolution process | Task 2 |
| Versioned replay dataset | Task 5–6 |
| BaseModel ABC with preprocess hook | Task 7 |
| Preprocessor owned by concrete model | Task 9 |
| Self-contained packed artifact (snapshots) | Task 9 |
| Config schema validation | Task 8 |
| Threshold optimisation to target recall | Task 11 |
| Test + replay eval report (JSON + console) | Task 12 |
| Loader from artifact folder (not codebase) | Task 10 |
| Gate: C1 recall ≥ active + floor 0.80 | Task 14 |
| Gate: C2 FPR guardrail 10pp | Task 14 |
| Gate: C3 McNemar p<0.05, min 200/30 | Task 14 |
| Gate dry-run by default, --promote to execute | Task 14 |
| Registry: index/status only | Task 1, 9, 14 |
| data_registry.json + model_registry.json | Task 1 |
| Test 1: all pass → PROMOTE | Task 13 |
| Test 2: FPR guardrail exceeded → REJECT C2 | Task 13 |
| predict.py reads active from registry | Task 16 |
| Bootstrap first model without gate | Task 14 |

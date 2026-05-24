# Model Pipeline Design Spec
**Date:** 2026-05-24
**Project:** Job-Applicant Fraud Detection — Closed-Loop Retraining Pipeline

---

## 1. Repository Layout

```
challenge/
├── data/
│   ├── raw/                        # raw inputs (immutable)
│   │   ├── applications_v1.csv
│   │   └── applications_v2.csv
│   ├── v1/                         # output of run_data_pipe.py
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── v2/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── replay/
│       └── v1/
│           ├── replay.csv
│           └── metadata.json
│
├── data_pipeline/                  # data module (package)
│   ├── __init__.py
│   ├── schema.py                   # versioned feature schemas + diff-style validation
│   ├── pipeline.py                 # stratified split, versioned folder creation
│   └── reader.py                   # load_train/val/test(version) → (X, y)
│
├── replay_pipeline/                # replay dataset module (package)
│   ├── __init__.py
│   ├── pipeline.py                 # join predictions + feedback, filter unclear, validate floors
│   └── reader.py                   # load_replay(version) → (X, y, metadata)
│
├── model_pipeline/                 # model module (package)
│   ├── __init__.py
│   ├── base_model.py               # abstract: fit, predict_proba, preprocess, save, load
│   ├── preprocessor.py             # StandardScaler wrapper
│   ├── config.py                   # config schema definition + validation
│   ├── validator.py                # threshold optimisation on val set → target recall
│   ├── evaluator.py                # test set + retrospective replay → JSON report + console
│   └── loader.py                   # load_model(version) → self-contained packed artifact
│
├── models/
│   └── v1/
│       ├── model.pkl               # trained weights
│       ├── base_model.py           # snapshot of BaseModel ABC at train time
│       ├── model_class.py          # snapshot of concrete model class at train time
│       ├── preprocessor.py         # snapshot of Preprocessor class at train time
│       ├── preprocessor.pkl        # fitted preprocessor state
│       ├── metadata.json           # threshold, features, schema_version, data_version
│       └── train_config.json       # full config snapshot for exact reproduction
│
├── reports/
│   └── eval_v1.json
│
├── tests/
│   └── test_promote.py             # 2 gate tests
│
├── config.json                     # default training config
├── data_registry.json              # data + replay version registry
├── model_registry.json             # model version registry
│
├── run_data_pipe.py                # CLI: --input --version
├── run_replay_pipe.py              # CLI: --predictions --feedback --version
├── train.py                        # CLI: --data-version --model-version [--config]
├── validate.py                     # CLI: --model-version
├── evaluate.py                     # CLI: --model-version --replay-version
├── promote.py                      # CLI: --candidate --replay-version [--promote]
└── predict.py                      # reads model_registry.json, loads active model
```

---

## 2. Registry Schemas

### `data_registry.json`

```json
{
  "latest": "v2",
  "versions": {
    "v1": {
      "version": "v1",
      "input_file": "data/raw/applications_v1.csv",
      "data_hash": "sha256:abc123...",
      "schema_version": "v1",
      "split_ratio": [0.7, 0.15, 0.15],
      "split_seed": 42,
      "splits": {
        "train": { "path": "data/v1/train.csv", "rows": 5600, "fraud_rate": 0.040 },
        "val":   { "path": "data/v1/val.csv",   "rows": 1200, "fraud_rate": 0.041 },
        "test":  { "path": "data/v1/test.csv",  "rows": 1200, "fraud_rate": 0.039 }
      },
      "created_at": "2025-08-14T10:00:00"
    }
  },
  "replay": {
    "latest": "v1",
    "versions": {
      "v1": {
        "path": "data/replay/v1/replay.csv",
        "rows": 1829,
        "fraud_rows": 145,
        "date_range": ["2025-09-01", "2025-11-28"],
        "weeks_spanned": 12.9,
        "sources": {
          "predictions_hash": "sha256:abc...",
          "feedback_hash": "sha256:def..."
        },
        "created_at": "2025-11-28T09:00:00"
      }
    }
  }
}
```

### `model_registry.json`

The registry is an index and status tracker only. Training hyperparameters, thresholds, and metrics live in the artifact and eval report — not here.

```json
{
  "active": "v1",
  "models": {
    "v1": {
      "version": "v1",
      "status": "active",
      "data_version": "v1",
      "artifact_path": "models/v1/",
      "eval_report_path": "reports/eval_v1.json",
      "created_at": "2025-08-14T10:00:00",
      "promoted_at": "2025-08-14T11:00:00",
      "promotion_gate": null
    },
    "v2": {
      "version": "v2",
      "status": "candidate",
      "data_version": "v2",
      "artifact_path": "models/v2/",
      "eval_report_path": null,
      "created_at": "2025-11-28T09:00:00",
      "promoted_at": null,
      "promotion_gate": null
    }
  },
  "history": []
}
```

**Model status lifecycle:** `candidate` → (validate) → (evaluate) → (promote) → `active` | `rejected`. When a new model is promoted, the previously active model moves to `history` with status `retired` and a `retired_at` timestamp. The `promotion_gate` field is written at gate time with the full result: pass/fail per condition, recall delta, FPR delta, p-value, and replay version used.

---

## 3. Data Pipeline Module (`data_pipeline/`)

### Schema versioning (`schema.py`)

Input schemas are versioned and immutable. Old schemas are never modified — a new field, removed field, or type change always produces a new schema version:

```python
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
```

`validate(df, schema_version)` collects **all** violations before raising — not fail-fast. Error format is a diff, not a version label:

```
Schema mismatch:
  Missing columns  : ['email_domain_risk_score']
  Unexpected columns: ['email_score']
  Type mismatches  : ['hour_of_day': expected int, got float]
```

This same diff-style error surface is used everywhere schema validation runs: `run_data_pipe.py`, `loader.py` at inference time, `predict.py`.

### Pipeline (`pipeline.py`)

1. Load raw CSV → validate schema (detect version from columns present, or `--schema-version`)
2. Resolve version: auto-increment from `data_registry.json` latest, or use `--version`
3. Stratified 70/15/15 split on `label`, fixed seed from config
4. Write `data/v{n}/train.csv`, `val.csv`, `test.csv`
5. Compute SHA-256 hash of input file
6. Append entry to `data_registry.json`

### Reader contract (`reader.py`)

```python
FEATURES = [
    "application_completion_seconds", "hour_of_day", "email_domain_risk_score",
    "account_age_days", "num_applications_last_24h", "ip_location_mismatch_km",
    "is_vpn_or_proxy", "profile_trust_score",
]

def load_train(version: str) -> tuple[pd.DataFrame, pd.Series]: ...
def load_val(version: str)   -> tuple[pd.DataFrame, pd.Series]: ...
def load_test(version: str)  -> tuple[pd.DataFrame, pd.Series]: ...
```

Returns `(X, y)` always — exactly the feature columns in `X`, `label` in `y`. Training module calls these and never touches CSVs directly. Any augmentation (class weights etc.) happens after this boundary, inside the training module.

---

## 4. Replay Pipeline Module (`replay_pipeline/`)

### Pipeline (`pipeline.py`)

1. Join `predictions.csv` + `feedback.csv` on `prediction_id`
2. Drop rows where `verdict == "unclear"`
3. Validate floor conditions: ≥200 rows, ≥4 weeks spanned — fail with clear message if not met
4. Write `data/replay/v{n}/replay.csv` + `metadata.json`
5. Hash both source files; append entry to `data_registry.json` under `"replay"`

### Reader (`reader.py`)

```python
def load_replay(version: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    # returns (X_features, y_true, metadata)
    # metadata includes row count, fraud count, date range, weeks spanned
```

`promote.py` calls `load_replay(version)` — never reads raw prediction/feedback files directly.

---

## 5. Model Pipeline Module (`model_pipeline/`)

### Base model contract (`base_model.py`)

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame: ...
    # concrete model decides: StandardScaler, identity, or anything else
    # callers never call preprocess() directly — only fit() and predict_proba()

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel": ...
```

### Concrete model (`model_class.py`, snapshotted into artifact)

`LogisticRegressionModel` owns its own `Preprocessor` instance — not the base class:

```python
class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def preprocess(self, X):
        return self.preprocessor.transform(X)

    def fit(self, X, y):
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocess(X))[:, 1]

    def save(self, path): ...   # saves model.pkl + preprocessor.pkl
    @classmethod
    def load(cls, path): ...    # loads both back in
```

A future model without preprocessing implements `preprocess()` as a no-op (`return X`). The abstraction holds regardless.

### Packed artifact (self-contained)

```
models/v2/
  model.pkl            # trained weights
  base_model.py        # snapshot of BaseModel ABC
  model_class.py       # snapshot of LogisticRegressionModel
  preprocessor.py      # snapshot of Preprocessor class
  preprocessor.pkl     # fitted preprocessor state
  metadata.json        # threshold, feature list, schema_version, data_version, created_at
  train_config.json    # full config for exact reproduction
```

Everything needed for inference lives in the artifact folder. `loader.py` loads `model_class.py` from the artifact — not from the current codebase — so changes to `model_pipeline/` after training never affect old artifacts.

### Config (`config.py` + `config.json`)

`config.json` is the default training config committed to the repo:

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
  "preprocessor": { "type": "StandardScaler" },
  "training": {
    "split_seed": 42,
    "target_recall": 0.80
  }
}
```

`config.py` validates any config against a locked schema before training starts — required fields, valid model types, valid preprocessor types, numeric ranges. `train.py` accepts `--config` to use a non-default config. The validated config is snapshotted into `models/v{n}/train_config.json` at pack time.

### Validator (`validator.py`)

Loads the candidate artifact via `loader.py`, scores the val set, sweeps thresholds 0.1–0.9, finds the lowest threshold that meets `target_recall` from config. Writes threshold into `models/v{n}/metadata.json`.

If no threshold meets target: `"No threshold achieves target recall 0.80 on val set. Best achievable: 0.74 at threshold 0.10"` — fail explicitly so the problem is diagnosable.

### Evaluator (`evaluator.py`)

Two-part report:

1. **Test set metrics** — loads `data/v{n}/test.csv` via `reader.py`, scores at artifact threshold, computes recall, FPR, AUC, F1
2. **Retrospective replay metrics** — loads replay version via `replay_pipeline.reader`, scores at artifact threshold, computes recall and FPR on labeled subset

Writes `reports/eval_v{n}.json` (machine-readable, for `promote.py`), prints human-readable table to console. Updates `eval_report_path` in `model_registry.json`.

### Loader (`loader.py`)

```python
def load_model(version: str) -> BaseModel:
    # reads models/v{n}/metadata.json
    # imports base_model.py and model_class.py from artifact folder
    # loads model.pkl and preprocessor.pkl via model_class.load()
    # validates incoming data schema on predict_proba calls (diff-style errors)
```

Single load interface for all consumers: `promote.py`, `evaluate.py`, `predict.py`.

---

## 6. Pipeline Scripts

### `run_data_pipe.py`
```bash
python run_data_pipe.py --input data/raw/applications_v2.csv --version v2
# auto-increments version if --version omitted
```

### `run_replay_pipe.py`
```bash
python run_replay_pipe.py --predictions data/predictions.csv --feedback data/feedback.csv
```

### `train.py`
```bash
python train.py --data-version v2 --model-version v2
python train.py --data-version v2 --model-version v2 --config configs/high_recall.json
python train.py --config models/v2/train_config.json   # exact reproduction
```

### `validate.py`
```bash
python validate.py --model-version v2
# reads data_version from model_registry, loads data/v2/val.csv
# writes threshold to models/v2/metadata.json only (not registry)
```

### `evaluate.py`
```bash
python evaluate.py --model-version v2 --replay-version v1
# writes reports/eval_v2.json + updates model_registry.json
```

### `promote.py`
```bash
python promote.py --candidate v2 --replay-version v1           # dry run, full report
python promote.py --candidate v2 --replay-version v1 --promote # execute if gate passes
```

Prerequisite check: `models/v2/metadata.json` must have `threshold` set (validate ran); `model_registry.json` entry for v2 must have `eval_report_path` set (evaluate ran). Fails with actionable message identifying which step was skipped.

Gate report format:
```
=== PROMOTION GATE REPORT ===
Replay dataset : v1 — 1829 rows | Sep 1 – Nov 28 2025 (12.9 weeks)

                  Active (v1)   Candidate (v2)   Delta
  Recall            0.76           0.84          +0.08
  FPR               0.04           0.09          +0.05

Gate conditions:
  [PASS] C1: candidate recall 0.84 >= active 0.76, above floor 0.80
  [PASS] C2: FPR increase 0.05 within 0.10 guardrail
  [PASS] C3: McNemar p=0.018 < 0.05 | 1829 examples | 47 disagreements

Verdict: PROMOTE — rerun with --promote to update registry
```

`promote.py` re-scores both models fresh on the replay data — it does not trust stored replay metrics. Active model's baseline is computed live on the same dataset so the comparison is always fair.

---

## 7. Promotion Gate Conditions

All three must pass. Any failure blocks auto-promotion; the full metric context is printed for human review.

| # | Condition | Threshold | Failure mode caught |
|---|-----------|-----------|---------------------|
| C1 | `candidate_recall >= active_recall` AND `candidate_recall >= 0.80` | Absolute floor 0.80 | Recall regression; ratchet degradation across cycles |
| C2 | `candidate_fpr - active_fpr <= 0.10` | 10pp guardrail | Indiscriminately aggressive model |
| C3 | McNemar p < 0.05, ≥200 examples, ≥30 disagreements | p < 0.05 | Noise mistaken for improvement |

---

## 8. Gate Tests (`tests/test_promote.py`)

**Test 1 — All conditions pass → PROMOTE**

Synthetic replay: 250 rows, 40 fraud. Active: recall 0.75, FPR 0.05. Candidate: recall 0.85, FPR 0.12 (7pp increase, within guardrail). McNemar p < 0.05, 35 disagreements. Expected verdict: PROMOTE.

**Test 2 — Recall improves but FPR guardrail exceeded → REJECT**

Same setup. Candidate: recall 0.88, FPR 0.18 (13pp increase, exceeds guardrail). C1 passes, C2 fails. Expected verdict: REJECT, with failure message identifying C2 specifically — not a generic rejection.

These two tests cover the primary promotion path and the key design tension: a model that boosts recall by flagging everything should be blocked.

---

## 9. Schema Evolution

Adding, removing, or renaming a feature field is a breaking change. The process:

1. Add new schema version to `data_pipeline/schema.py` — old schemas are immutable
2. Run `run_data_pipe.py` with new raw data containing the new field → new data version
3. Train a new model on the new data version
4. Old models continue to serve against old-schema inputs until the new model is promoted
5. On promotion, `predict.py` expects new-schema inputs — diff-style error if old-schema data arrives

Schema version is recorded in `data_registry.json` per data version and in `models/v{n}/metadata.json` per artifact.

# Job-Applicant Fraud Detection — Model Pipeline

**Video walkthrough:** _[Loom link — add before submitting]_

---

## What this is

A binary classifier that scores incoming job applications and flags fraudulent identities — bots, applicant farms, stolen or synthetic identities. The model scores each application at submission time; scores above the operating threshold trigger a block or escalation before the candidate advances in the hiring pipeline.

---

## What I changed from the starter, and why

The starter had three problems worth naming:

**1. Accuracy on training data is meaningless.** The original `train.py` evaluated the model on the same rows it trained on. That 99.94% accuracy figure in `registry.json` is noise. The fraud class is ~4% of the data — a model that predicts all-legit gets 96% accuracy without learning anything. I replaced accuracy with recall, FPR, AUC, and F1 on a held-out test split.

**2. The promotion gate compared apples to apples, then did nothing.** The original `promote.py` loaded both models, evaluated them both on the training data, printed PROMOTE or KEEP ACTIVE, and stopped. Nothing was written back to the registry. I replaced this with a gate that: (a) evaluates both models on the same retrospective replay dataset — real production predictions with delayed reviewer labels — rather than stale training data, (b) applies three conditions rather than one metric, and (c) actually updates the registry when promotion passes.

**3. A single mutable artifact.** The original pipeline overwrote `models/model.pkl` on every training run and edited `registry.json` by hand. There was no way to know what data a given model was trained on, or to roll back to a previous version. I replaced this with versioned artifact folders and two JSON registries that are only ever appended to.

---

## Repository layout

```
data/
  raw/                    # raw inputs — schema-validated on intake
  v1/ v2/                 # versioned train/val/test splits (70/15/15)
  replay/v1/              # versioned replay dataset (predictions + labels)

data_pipeline/            # schema validation, stratified split, reader contract
replay_pipeline/          # joins predictions.csv + feedback.csv → replay dataset
model_pipeline/           # BaseModel ABC, preprocessor, config, loader, validator, evaluator

train.py                  # trains, packs self-contained artifact, registers candidate
validate.py               # finds operating threshold that hits target recall on val set
evaluate.py               # test set + replay metrics → JSON report
promote.py                # promotion gate — dry run by default, --promote to execute
predict.py                # scores an application using the active model
run_data_pipe.py          # CLI for data_pipeline
run_replay_pipe.py        # CLI for replay_pipeline

config.json               # default training config (validated schema)
data_registry.json        # data and replay version index
model_registry.json       # model version index — status and pointers only
```

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## Running the pipeline

### Step 1 — Version the training data

```bash
python run_data_pipe.py --input data/raw/applications_v1.csv --version v1
python run_data_pipe.py --input data/raw/applications_v2.csv --version v2
```

Validates the input schema, creates a stratified 70/15/15 split, writes `data/v1/` and `data/v2/`, and records the SHA-256 hash of the source file in `data_registry.json`.

### Step 2 — Version the replay dataset

```bash
python run_replay_pipe.py --predictions data/predictions.csv --feedback data/feedback.csv
```

Joins the prediction log to reviewer feedback on `prediction_id`, drops `unclear` verdicts, validates the dataset meets the gate's floor conditions (≥200 rows, ≥4 weeks spanned), and writes `data/replay/v1/`.

### Step 3 — Train a candidate

```bash
python train.py --data-version v2 --model-version v2
```

Trains a `LogisticRegression` with `class_weight='balanced'` on `data/v2/train.csv`, packs a self-contained artifact into `models/v2/` (model weights, preprocessor state, class definition snapshots, config), and registers v2 as a candidate in `model_registry.json`. The artifact is self-contained: `loader.py` imports the model class from inside the artifact folder, not from the current codebase, so future code changes can't silently break old artifacts.

To reproduce a specific run exactly:

```bash
python train.py --config models/v2/train_config.json
```

### Step 4 — Find the operating threshold

```bash
python validate.py --model-version v2
```

Scores the validation set and sweeps thresholds from 0.05 to 0.95 to find the lowest threshold that achieves `target_recall` (default 0.80 from `config.json`). Writes the threshold into the artifact's `metadata.json`. This is separate from training because the threshold is a business decision — you can tighten or loosen it without retraining.

### Step 5 — Evaluate

```bash
python evaluate.py --model-version v2 --replay-version v1
```

Evaluates on two datasets: the held-out test split (for a clean offline metric) and the retrospective replay (for a metric grounded in what the model will actually see in production). Writes `reports/eval_v2.json` and prints a summary table.

### Step 6 — Run the promotion gate

```bash
# Dry run — shows gate report without updating registry
python promote.py --candidate v2 --replay-version v1

# Execute — updates registry if all conditions pass
python promote.py --candidate v2 --replay-version v1 --promote
```

### Score an application

```bash
python predict.py
```

Reads the active model version from `model_registry.json`, loads the artifact, and scores using the threshold set by `validate.py`.

### Run tests

```bash
.venv/bin/python -m pytest tests/ -v
```

---

## The promotion gate

The gate requires all three conditions. Meeting two of three is not sufficient — disagreement escalates to human review.

**Condition 1 — Recall** (primary signal)

The candidate's recall on the replay dataset must be ≥ the active model's recall, and must also clear an absolute floor of 0.80. The relative comparison alone has a slow-ratchet failure mode: if the active model has degraded to recall 0.62 and the candidate comes in at 0.64, it promotes, and 0.64 becomes the new baseline. Over several cycles this compounds. The absolute floor prevents this.

**Condition 2 — FPR guardrail** (safety check)

The candidate's FPR must not increase by more than 10 percentage points over the active model. The system accepts higher false-positive rates in exchange for higher recall — a genuine candidate who is incorrectly flagged can be reviewed and reinstated; a fraudulent identity that slips through may not be caught at all. But a model that improves recall by flagging everything provides no signal quality and erodes reviewer trust. The guardrail blocks this.

**Condition 3 — Statistical significance**

McNemar's test (with continuity correction) on the paired binary predictions, requiring p < 0.05, at least 200 labeled replay examples, and at least 30 disagreement cases between the two models. This prevents promoting a model whose apparent improvement is within the noise of a small labeled sample.

**Why recall-first, not accuracy or precision-first**

This model makes a binary decision — block or allow — before a candidate advances. A false positive (genuine candidate wrongly blocked) damages that person's job opportunity and is reversible if caught by a reviewer. A false negative (fraudulent identity wrongly passed) may go undetected through the entire hiring process. FN cost dominates. Accuracy hides this because the fraud class is rare (~4%); a model predicting all-legit gets 96% accuracy. Precision matters, but not at the cost of recall — we accept more false alarms in exchange for catching more fraud.

### What happened when we ran it

Both v1 and v2 achieve AUC 1.0 and recall 1.0 on their respective held-out test sets. This reflects the synthetic data — the feature distributions for fraud and legit are well-separated and the model has no trouble separating them on in-distribution holdout data.

The replay tells a different story. On the 1,829 labeled production predictions:

|         | Recall | FPR    | AUC    |
|---------|--------|--------|--------|
| v1 (active) | 0.4828 | 0.0042 | 0.7412 |
| v2 (candidate) | 0.4828 | 0.0048 | 0.7462 |

v2 was correctly rejected by the gate:

- **C1 FAIL**: candidate replay recall 0.4828 is below the 0.80 absolute floor
- **C3 FAIL**: only 1 disagreement between v1 and v2 on the replay data — not enough to run a meaningful McNemar test

This is the right result. Both models, despite perfect test-set performance, only catch about half the fraud cases in the production prediction log. The test set and the replay are measuring different things: the test set measures how well the model separates in-distribution examples; the replay measures how well it performs on the actual traffic the original model saw, which has survivorship bias baked in (only above-threshold predictions were reviewed, so the labeled set skews toward borderline cases the original model was uncertain about). The gap between test recall (1.0) and replay recall (0.48) is a signal worth investigating before promoting anything — it suggests the model hasn't learned to generalise to the harder edge cases in production traffic.

For the video, I'll demonstrate the full retrain → validate → evaluate → gate flow and walk through this gap in detail.

---

## Monitoring in production

Two layers, as described in the design doc:

**Layer 1 — Label-free, real-time (Evidently AI, 4-hour schedule)**

- **PSI on input features**: compare the rolling 7-day feature distribution against the training distribution frozen at the last deployment. Warning at PSI > 0.1, alert at > 0.2. On this model, `email_domain_risk_score`, `ip_location_mismatch_km`, and `is_vpn_or_proxy` are the features most likely to drift as fraud patterns change — watch these first.
- **PSI on score distribution**: the same thresholds. During the 48–72 hour burn-in window after any promotion, run this check at high frequency and trigger auto-rollback if PSI > 0.2. Post burn-in, alert for human review.
- **Request volume**: alert at ±40% of 30-day baseline. Volume shifts usually indicate infrastructure changes, not model degradation, but they're cheap to watch.

Layer 1 is a leading indicator. It fires early on distribution shift but cannot confirm model degradation without labels.

**Layer 2 — Labeled feedback, delayed (Prefect/Dagster, rolling window)**

- **Recall on labeled feedback**: trigger retraining if recall drops more than 5pp below the deployment baseline. For this model, the baseline replay recall is ~0.48 — already low, which means the retraining trigger will fire quickly. This is a healthy signal: the system will demand better models.
- **FPR on labeled feedback**: alert if FPR rises more than 10pp. This represents genuine candidates being over-flagged — it shows up as reviewer queue growth before it shows up in feedback labels.
- **Label arrival rate**: alert if no labels arrive for two consecutive weeks. Silence usually means the feedback pipeline is broken, not that the model is perfect.

**Distinguishing drift types**: if Layer 1 PSI fires but Layer 2 recall is stable, the input distribution changed but the model still works — investigate the data pipeline. If Layer 2 recall fires but PSI is stable, fraudsters adapted while keeping input features similar (concept drift without data drift) — retrain, prioritise recent labeled examples. If both fire, retrain and investigate the pipeline.

---

## From registry to serving traffic

`predict.py` already shows the path: read `model_registry.json`, load the active artifact via `model_pipeline.loader`, apply the threshold stored in the artifact's `metadata.json`. The production serving layer would wrap this in a Flask or FastAPI endpoint.

In a production deployment (not built here, described per the challenge instructions):

1. **Artifact storage**: `models/v{n}/` directories would live in S3 or GCS rather than on disk. `loader.py` would pull from there using the `artifact_path` pointer in the registry. The registry itself would move to a database or an object store with a single writer.

2. **Promotion without downtime**: the registry's `"active"` field is a single string key. Promotion updates this key atomically. The serving layer reads it on each request (or caches it with a short TTL), so traffic shifts to the new model without a restart.

3. **Rollback**: the previous active model is moved to `history` in `model_registry.json` but its artifact is never deleted. Rollback means writing the previous version's key back into `"active"`. If using MLflow (as designed), this is `client.set_registered_model_alias("fraud-detector", "production", previous_version)`.

4. **Burn-in**: the first 48–72 hours after promotion run Layer 1 PSI at high frequency (every 4 hours). If score distribution shifts sharply (PSI > 0.2), auto-rollback fires and reverts `"active"` to the previous version. This is the only automated rollback; post burn-in rollbacks are manual.

5. **Threshold as a versioned config**: the operating threshold lives in the artifact's `metadata.json`, not in the serving layer. To adjust the threshold without retraining, run `validate.py` with a different `target_recall`, which writes a new threshold into the artifact. The serving layer picks it up on the next request. Each threshold change is logged with the previous value and the recall check that justified it.

---

## Design decisions worth calling out

**Why retrospective replay over a fixed holdout set** — a fixed holdout is historical by definition. As the input distribution shifts over time, the holdout becomes less representative of what the model faces in production. Retrospective replay re-scores already-decided, already-labeled production predictions with both the active and candidate model, then compares them against the same ground truth. It's grounded in recent real-world data and doesn't expose live applicants to an unvalidated model.

**Why not A/B testing** — routing even a small fraction of live traffic to an unvalidated model means real hiring decisions are made by a model that hasn't passed the gate. A false positive on a genuine candidate has immediate consequences for that person. Retrospective replay evaluates candidates who already have known outcomes.

**Why not time-based retraining** — a scheduled retrain decouples the retraining cost from evidence that retraining is needed. A healthy model gets retrained unnecessarily; a degrading model waits for the next cycle. FPR/recall-triggered retraining ties compute cost to observed degradation.

**Survivorship bias in the replay data** — candidates who score below the operating threshold are never reviewed and never labeled. The feedback table is structurally blind to false negatives — fraudulent identities the model confidently cleared. This is a known limitation. The mitigation is to periodically surface a random sample of sub-threshold predictions for manual review. It doesn't need to be many; even 50–100 per month would let you audit whether the model's pass decisions are trustworthy.

**Registry design** — `model_registry.json` is an index and status tracker, not a metrics store. Metrics live in `reports/eval_{version}.json`; hyperparameters and training config live in the artifact's `train_config.json`. This keeps the registry auditable at a glance without it becoming a dump of every number the pipeline ever computed.

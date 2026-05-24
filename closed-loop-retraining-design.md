# Closed-Loop Model Retraining — Design Document

---

## 1. System Design

### Overview

This system is built for candidate identity fraud detection: a binary classifier scores incoming job applicants and classifies them as either a genuine candidate or a fraudulent identity — someone using a fabricated or stolen identity to apply for roles. A flagged applicant is blocked or escalated before advancing in the hiring pipeline, a decision that is difficult to reverse once made. The asymmetry matters: falsely flagging a genuine candidate damages their opportunity and the hiring company's talent pipeline, while a missed fraudulent identity advances through the process under a false persona. This shapes every design decision that follows.

The system closes the loop between live predictions, delayed consumer feedback, and periodic model retraining. The core principle is conservative by default: before triggering an expensive retraining run, the system checks whether a threshold adjustment alone resolves the degradation. Retraining is demand-driven, not scheduled.

**Design assumption — asymmetric cost:** This design assumes FN cost significantly exceeds FP cost. A missed fraudulent identity advances through the hiring pipeline under a false persona — potentially being placed in a role, accessing systems, or defrauding the hiring company before being caught. A genuine candidate who is falsely flagged can be reviewed and reinstated; a fraudulent identity that slips through may not be caught at all. Recall is therefore the primary monitoring signal and promotion gate criterion throughout. The system is calibrated to over-alert rather than under-alert: false positives create review burden; false negatives create fraud exposure. If this assumption is wrong — for example, if hiring companies report high review fatigue from excessive false positives — the balance between recall and precision should be revisited. This should be validated with the hiring companies Tofu serves before go-live and revisited as new customer segments are onboarded.

```
=== FEEDBACK PIPELINE ===

[Feedback Arrives]
       |
       +--[UNCLEAR]-----------> [Human Reviewer] ------+
       |                                               |
       +--[CORRECT / INCORRECT]                        |
                   |                                   |
                   v                                   |
       [Re-score with Live Model]                      |
                   |                                   |
                   v                                   |
       [Threshold shift fix FPR?]                      |
                   |                                   |
       YES (recall holds)  NO                          |
           |                |                          |
           v                +---------------------------+
   [Update Threshold]                                  |
       [DONE]                                          |
                                                       v
                                   [Data Cleaning + source=real_world tag]
                                                       |
                                                       v
                                   [Training / Validation / Test Data]
                                                       |
                                                       |
=== MONITORING ===                                     |
                                                       |
[Live Model] --> [Prediction Log]                      |
                       |                               |
          +------------+------------+                  |
          |                         |                  |
          v                         v                  |
  [Layer 1: Real-Time]     [Layer 2: FPR Monitor]      |
  No labels needed         Delayed, needs labels        |
                                    |                  |
  3 signals watched:        FPR > Baseline             |
  a) PSI on input features          |                  |
     warn >0.1, alert >0.2  [Trigger Retraining] <----+
  b) PSI on score dist.             |
     warn >0.1, alert >0.2          |
     (auto rollback if              |
      within burn-in window)        |
  c) Request volume                 |
     alert if +-40% baseline        |
          |                         |
       [Alert]                      |
      Investigate                   |
                                    |
=== RETRAINING PIPELINE ===         |
                                    v
                       [Train Candidate Model]
                                    |
                                    v
                       [Retrospective Replay]
                  re-score recent labeled predictions
                  with both live + candidate model
                                    |
                                    v
                          [Promotion Gate]
                     1. Candidate recall >= live recall
                     2. FPR increase < 10pp (guardrail)
                     3. McNemar's test p < 0.05
                                    |
                  +-----------------+-----------------+
                  |                                   |
            [ALL PASS]                    [SIGNALS DISAGREE]
                  |                                   |
                  v                                   v
          [Promote via MLflow]               [Human Review]
                  |                                   |
                  v                                   +---> back to gate
          [48-72hr Burn-in]
       high-frequency PSI watch
                  |
       +----------+----------+
       |                     |
 [Score Drift]            [Clean]
       |                     |
       v                     v
 [Auto Rollback]       [Live Model]
 prev. artifact        cycle continues
```

---

### Feedback Ingestion and Data Pipeline

Feedback is joined to predictions via `prediction_id`, a key written to both the prediction log and the feedback table at the time the prediction is served. This is the single join key that links a consumer's label back to the original input features and score. Predictions that never receive a label are retained in the prediction log for monitoring but excluded from training data.

**Survivorship bias:** Candidate profiles that score below the operating threshold are passed through without review and therefore never labeled. The feedback table is structurally blind to false negatives — fraudulent identities the model confidently cleared. This is especially dangerous in identity fraud: a synthetic identity that learns to mimic genuine candidates stays invisible to the retraining pipeline indefinitely. The mitigation is to periodically surface a random sample of sub-threshold applicants for manual review — not at scale, but enough to audit whether the model's pass decisions are trustworthy.

Incoming feedback from the feedback table is split by label type:

- **Correct / Incorrect:** The live model is re-run on the feedback batch and the score distribution is examined. A threshold sensitivity check is performed — if lowering the operating threshold recovers the missed fraudulent identities (false negatives) without driving FPR above the guardrail ceiling, only the threshold is updated and retraining is skipped entirely. If the threshold shift cannot recover recall, the examples are added to the training, validation, and test data pools.

  > **Example:** 25 applicant profiles arrive labeled incorrect — fraudulent identities the model passed through. The live model scored them between 0.42–0.49 against a threshold of 0.50. Lowering the threshold to 0.40 correctly catches all 25 while keeping FPR within the guardrail. Threshold is updated; no retraining run is triggered.

  **Threshold versioning and rollback:** The operating threshold is treated as a versioned config value, not a mutable setting. Every change is logged with the previous value, the timestamp, and the recall and FPR check results that justified it. If Layer 2 monitoring shows recall degrading further after a threshold change — in a part of the distribution not covered by the test set check — the previous threshold is restored from the log. Threshold rollback follows the same trigger as model rollback: recall drops more than 5pp from the pre-change baseline.

- **Unclear:** Routed to a human reviewer to obtain a definitive label. Once labeled, the example proceeds directly to data cleaning and the training data pool — it bypasses the threshold check, which applies only to batches of model-scored predictions.

All examples sourced from real-world feedback carry a `source=real_world` tag in the training dataset. This enables a **resilience metric** — AUC and FPR evaluated exclusively on the real-world subset — reported alongside overall metrics at every evaluation. A new model whose real-world subset FPR is worse than the live model's real-world FPR does not promote, even if aggregate metrics pass. This is enforced as an additional condition in the promotion gate alongside the three conditions in Section 2.

  > **Example:** After adding 600 real-world feedback examples, the model reports AUC 0.88 overall but AUC 0.79 on the `source=real_world` subset. The gap signals the model is overfitting to historical applicant patterns and underperforming on the identity fraud cases it actually encounters in production. A new model that closes this gap passes the check; one that widens it is blocked.

**Temporal splitting:** When constructing train/validation/test splits from the combined dataset (historical + real-world feedback), splits must be time-based — not random. Random splitting would allow feedback records from the future to appear in training, leaking outcome information that wouldn't have been available at prediction time. The split boundary is set by timestamp: training data precedes the validation cutoff, validation precedes the test cutoff.

---

### Retraining Trigger

Retraining is not time-based. It is triggered when **Layer 2 monitoring** (recall on labeled feedback) drops below the baseline recall established at the last deployment by **more than 5 absolute percentage points**. The 5pp margin is a fixed operational parameter — it represents the point at which recall degradation becomes consequential given the cost of missing fraudulent identities. The recall baseline itself is recorded at deployment time from the promoted model's performance on the retrospective replay dataset.

> **Example:** The model was deployed with a baseline recall of 0.87. Over the following three weeks, as 300 labeled applicant profiles accumulate, the observed recall on feedback drops to 0.79 — 8 points below baseline, crossing the 5-point margin. The retraining pipeline is triggered automatically.

**Training data window:** When retraining fires, the new model is trained on a **rolling window of the most recent 18 months of data**, not the full historical corpus. Training on all historical data means new feedback examples — the signal retraining is meant to respond to — are numerically drowned out by older records. A rolling window keeps the model responsive to emerging identity fraud patterns while retaining enough history for stability. The 18-month boundary is a starting parameter to be revisited once empirical data on model degradation rate is available.

**Class balance monitoring:** Fraudulent identities are a rare class relative to the volume of genuine candidates. The feedback loop adds labeled data only from above-threshold predictions — confirmed fraudulent identities and genuine candidates wrongly flagged. Genuine candidates who score well below the threshold are never reviewed and never enter training. Over multiple retraining cycles, the training data skews toward borderline cases and loses coverage of clearly genuine applicants. To counter this: monitor the positive/negative class ratio at the start of every retraining run and alert if it deviates more than 10% from the ratio at initial training. Use **class weights** in the training job to correct for imbalance — most frameworks support this as a single parameter, leaving the data pipeline untouched.

---

### Candidate Evaluation

Once a candidate is trained, it is evaluated via **retrospective replay**: recent labeled predictions — already stored in the prediction log and feedback table — are re-scored by the candidate. Both the live model's scores and the candidate's scores are compared against the same ground truth labels. This is fast, safe, and grounded in recent real-world data rather than stale historical holdout sets.

**Train/eval partition:** The recent labeled examples collected when retraining is triggered must be partitioned before any data flows anywhere. A held-out portion is reserved exclusively for retrospective replay evaluation and never enters the training data. Without this hard partition, the candidate is evaluated on examples it trained on, producing optimistically biased metrics and a gate that cannot be trusted.

> **Example:** 350 labeled applicant profiles from the past 6 weeks are pulled from the prediction log and feedback table. The live model scores these with FPR 0.17 and recall 0.81. The new model scores the same 350 profiles with FPR 0.10 and recall 0.80. McNemar's test on the disagreement cases returns p = 0.02. All promotion gate conditions are met.

---

### Inference Layer

The model artifact is packaged together with its inference code — feature computation, preprocessing, and postprocessing — as a single deployable unit using MLflow's `pyfunc` model flavor. Every consumer of the model — retrospective replay evaluation, burn-in monitoring, and production serving — calls through this same packaged artifact. There is no separate feature pipeline that runs at evaluation time vs. production time. This eliminates training-serving skew by construction: if evaluation passes, production sees identical inputs, because they are the same code path.

### Promotion

Promotion is handled through the **MLflow model registry**. The candidate is registered as a new version. If it passes the promotion gate (Section 2), the production alias is updated to point to the new version. The previous artifact is always retained and can be restored immediately.

---

### Rollback

Immediately after promotion, a **48–72 hour burn-in window** begins. During this period, Layer 1 monitoring (PSI on prediction score distribution) runs at high frequency. A sharp shift in score distribution triggers an **automatic rollback** — the production alias reverts to the previous artifact without human intervention.

After the burn-in window passes cleanly, rollback becomes manual: a human reviews sustained FPR degradation from Layer 2 monitoring and makes the call. Automated rollback is limited to the burn-in window because spurious PSI alerts — caused by transient traffic patterns rather than a bad model — are more tolerable during routine operation than immediately after a promotion swap.

> **Example:** Six hours after promoting a new candidate, the PSI on prediction score distribution spikes to 0.34. Investigation reveals a feature pipeline bug was sending null values for a key input feature, causing the model to score almost everything near 0.5. Auto rollback fires immediately, reverting to the previous artifact. The bug is fixed before re-attempting promotion.

---

## 2. Promotion Gate

The gate requires **all three conditions** to be satisfied. Meeting two of three is not sufficient for automatic promotion — disagreement between signals escalates to a human.

### Condition 1 — Recall at Operating Threshold
The candidate's recall at the operating decision threshold must be **≥ the live model's recall** on the retrospective replay dataset. This is the primary signal, directly tied to the downstream cost of missed fraudulent identities advancing through the hiring pipeline.

**Absolute recall floor:** The relative comparison alone has a failure mode: if the live model has degraded to recall 0.62 and the new model comes in at 0.64, it promotes — and 0.64 becomes the new baseline. Over several cycles this is a slow ratchet of compounding recall degradation, with progressively more fraudulent identities slipping through. To prevent this, an absolute floor of **recall ≥ 0.80** applies regardless of relative improvement. No model promotes below this floor. If no model can clear it, the system needs intervention beyond retraining — new features, more labeled data, or a fundamental model review.

### Condition 2 — FPR Must Not Increase Unboundedly (Guardrail)
FPR on the replay dataset must not increase by more than **10 percentage points** relative to the live model. The system accepts higher false positive rates in exchange for higher recall — a genuine candidate who is over-flagged can be reviewed and reinstated — but a model that becomes indiscriminately aggressive provides no signal quality and erodes reviewer trust. A candidate that improves recall by becoming excessively aggressive (flagging nearly everything) fails this check.

### Condition 3 — Statistical Significance
The difference in predictions between the two models must be statistically significant using **McNemar's test** (appropriate for paired binary classifiers evaluated on the same examples), requiring **p < 0.05**. This prevents promoting a model whose apparent improvement is within the noise of a small labeled sample.

**Minimum sample size:** The promotion gate requires at least **200 labeled examples** in the retrospective replay dataset, with at least **30 disagreement cases** between the live model and the candidate. This floor is set by the statistical power requirements of McNemar's test — it is independent of training dataset size, which is a red herring here. A model trained on 500K examples is not evaluated more rigorously by requiring 5,000 replay examples; what matters is whether the disagreement cases are sufficient to distinguish real improvement from noise.

**Representativeness:** To ensure the evaluation set covers meaningful variation rather than just the most recent batch of feedback, the replay dataset must span **at least 4 weeks of labeled predictions** across distinct time slices. This guards against the evaluation passing on a narrow, easy slice of the distribution while the model still fails on cases that arrive less frequently.

### When Signals Disagree

If the candidate improves recall but drives FPR beyond the 10-point guardrail, or if the improvement does not clear statistical significance, the gate does **not** auto-promote. A human reviews the full metric context — replay dataset size, recall delta, FPR delta, and p-value — and makes the final call. The goal is to make the default safe and require active human judgement to override it.

> **Example:** The new model improves recall from 0.79 to 0.87 (passes condition 1) but FPR rises from 0.12 to 0.24 — a 12-point increase exceeding the 10-point guardrail (fails condition 2). The model is becoming indiscriminately aggressive. A human reviewer examines score distributions by feature bucket, determines the model is systematically over-scoring a legitimate applicant segment, and rejects the model. The affected segment is flagged for targeted data collection in the next training run.

---

## 3. Drift and Monitoring

Two monitoring layers run continuously on the live model. Layer 1 is immediate and label-free. Layer 2 is authoritative but delayed.

### Layer 1 — Real-Time (No Labels Required)

| Signal | Warning | Alert | Action |
|---|---|---|---|
| PSI on input features (per feature) | 0.1 – 0.2 | > 0.2 | Warning: increase check frequency. Alert: investigate upstream data pipeline, flag for human review. |
| PSI on prediction score distribution | 0.1 – 0.2 | > 0.2 | Warning: watch closely. Alert: if within burn-in window → auto rollback. Post burn-in → human review. |
| Prediction request volume | ±20% of 30-day baseline | ±40% | Investigate upstream data pipeline — likely infrastructure, not model degradation. |

PSI thresholds follow the standard convention: < 0.1 stable, 0.1–0.2 moderate shift worth monitoring, > 0.2 significant change requiring action.

**PSI reference window:** PSI is computed by comparing the current production distribution (rolling 7-day window of live traffic) against the training distribution captured and frozen at the time of the last deployment. Using a rolling window on the current side catches gradual drift that would be invisible if comparing against a fixed recent snapshot. The reference distribution is re-anchored to training data every time a new model is promoted.

**Check frequency:** Layer 1 PSI runs on a **4-hour schedule** via Evidently AI. This ensures that during the 48–72 hour burn-in window, at least 12–18 checks occur before the window closes — enough resolution to catch a bad promotion within hours rather than the next morning.

> **Example:** A feature representing an applicant's employment history length has a training distribution concentrated in a certain range. Three weeks into deployment, a new pattern of synthetic identities with fabricated multi-decade histories starts appearing. PSI on that feature rises to 0.26, crossing the alert threshold. Layer 1 fires. Layer 2 FPR on labeled feedback is also rising from 0.10 to 0.16. Both signals agree: retraining is triggered.

Layer 1 is a **leading indicator**. It fires early but cannot confirm model degradation without labels. It raises the alert; Layer 2 confirms it.

**Distinguishing data drift from concept drift:** PSI on features alone cannot tell you which type of drift you are seeing, and the remediation paths are different. Use this decision matrix before acting on a Layer 1 alert:

| Layer 1 PSI | Layer 2 FPR | Interpretation | Action |
|---|---|---|---|
| Fires | Stable | Data drift or pipeline issue — profiles are changing but model still performs | Investigate upstream data pipeline. Do not retrain. |
| Stable | Fires | Concept drift — fraudsters adapted, inputs look the same but labels changed | Retrain. Prioritise recent labeled examples in training window. |
| Both fire | Both fire | Combined drift — new profile types and degraded model performance | Retrain. Also investigate pipeline for data quality issues. |
| Neither fires | Neither fires | Model is healthy | No action. |

In identity fraud, concept drift without data drift is a realistic and dangerous scenario: sophisticated fraudulent actors deliberately learn to mimic genuine candidate profiles while changing the underlying fraud patterns. PSI on features will not catch this — only rising FPR on labeled feedback will.

### Layer 2 — Delayed (Requires Labels)

| Signal | Threshold | Action |
|---|---|---|
| Recall on labeled feedback | Drop > 5pp from deployment baseline | Trigger retraining pipeline |
| FPR on labeled feedback | Rise > 10pp from deployment baseline | Escalate to human — model may be over-flagging, degrading reviewer trust |
| Label arrival rate | Zero labels for 2 consecutive weeks | Alert — feedback pipeline may be broken |

The recall baseline is recorded at deployment time from the promoted model's performance on the retrospective replay dataset. The alert margin — 5 absolute percentage points — is a fixed operational parameter applied consistently across all deployments.

### Tooling

- **Evidently AI** — computes PSI on features and score distribution on a schedule, generates drift reports, and fires alerts when thresholds are crossed.
- **Prefect / Dagster** — orchestrates the full pipeline: alert → threshold check → retraining → evaluation → promotion decision.
- **MLflow** — manages artifact versioning, model registry, and production alias updates for promotion and rollback.

---

## 4. Explicit Non-Choices

### OOD Detection as a Decision Gate

I considered flagging incorrect predictions as out-of-distribution before adding them to training data. If a prediction is OOD, the model's failure is distributional — simply adding the example to training may not fix the underlying problem.

**Rejected because:** OOD detection requires a well-calibrated reference distribution and produces weak, unreliable signals without one. Engineering a robust OOD detector adds significant complexity for a small team. The pragmatic alternative — threshold sensitivity check first, then add to training if that fails — captures the majority of the value with far less overhead. OOD detection is a candidate for a later iteration if systematic distributional failures become a recurring pattern.

---

### Shadow Mode Evaluation

I considered running the candidate model in parallel on live traffic before promotion. Shadow mode gives the most accurate signal of candidate behavior on real incoming data, including the current input distribution.

**Rejected because:** Shadow mode requires waiting weeks for labels to arrive on shadow predictions before a promotion decision can be made. That lag defeats the purpose of a system designed to respond to model degradation. Retrospective replay on already-labeled predictions achieves comparable signal without the delay.

---

### Holdout-Only Evaluation

I considered maintaining a fixed holdout set for all candidate evaluation. It is simple to implement and deterministic.

**Rejected because:** A fixed holdout set is historical by definition. As input distribution shifts over time, the holdout becomes less representative of what the model will face in production. Retrospective replay on recent labeled predictions is more expensive to set up but more honest about current performance.

---

### Time-Based Retraining Cadence

I considered retraining on a fixed schedule (e.g., every two weeks) regardless of model health. This is predictable and easy to operate.

**Rejected as the primary trigger because:** It decouples retraining from evidence that retraining is needed. A healthy model gets retrained unnecessarily; a degrading model waits for the next cycle. FPR-triggered retraining ties the compute cost directly to observed degradation.

---

### Champion/Challenger A/B Testing

I considered routing a small percentage of live traffic to the candidate model and comparing outcomes against the live model over time — a standard champion/challenger setup.

**Rejected because:** In identity fraud detection, a fraud flag blocks a candidate from advancing in the hiring process — a decision that is difficult to reverse. Exposing even a small fraction of live applicants to an unvalidated model means real hiring decisions are made by a model that hasn't passed the promotion gate. A false positive on a genuine candidate has immediate consequences for that person. Retrospective replay evaluates the new model on already-decided, already-labeled profiles — no live applicants are exposed to an unvalidated model before promotion.

---

### Online / Continuous Learning

I considered updating the model continuously as each feedback label arrives — adjusting weights incrementally rather than triggering a full retraining run. This would make the model respond to new fraud patterns faster than a batch retraining cycle allows.

**Rejected because:** Continuous learning introduces instability that is hard to monitor and harder to roll back. Each incremental update shifts the model in an uncontrolled direction; there is no clean artifact version to revert to if a bad batch of labels corrupts the weights. In a system where a false positive blocks a real person from a job, silent model drift from continuous updates is more dangerous than the lag of a scheduled retraining cycle. Batch retraining with a versioned artifact and an explicit promotion gate gives the team a clear intervention point — continuous learning removes it.

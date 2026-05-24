# Test Suite Design Spec
**Date:** 2026-05-24
**Project:** Job-Applicant Fraud Detection — Model Pipeline

---

## Approach

**Unit tests + integration tests with `tmp_path`.** Pure logic functions (metric computation, schema validation, gate conditions, threshold sweep) are tested in-memory with synthetic DataFrames. Pipeline functions that touch the filesystem use pytest's `tmp_path` fixture to write real files in a temp directory — no mocks on file I/O, no dependency on the real repo's `data/` or `models/` directories.

**No mocking of file I/O.** Mocking JSON reads/writes hides the bugs most likely to occur in pipeline code.

---

## File Structure

```
tests/
  conftest.py              # shared fixtures
  test_schema.py           # 7 tests
  test_data_pipeline.py    # 11 tests
  test_replay_pipeline.py  # 12 tests
  test_config.py           # 8 tests
  test_preprocessor.py     # 8 tests
  test_validator.py        # 7 tests
  test_evaluator.py        # 11 tests
  test_loader.py           # 7 tests
  test_promote.py          # 2 existing + 9 new = 11 tests
```

Total: ~55 tests across 9 test files + conftest.

---

## `tests/conftest.py` — Shared Fixtures

### `tiny_df`
100-row synthetic DataFrame with all 8 features + label (10% fraud). Pure in-memory. Used wherever sample data is needed.

```python
@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(0)
    n, n_fraud = 100, 10
    legit = {
        "application_completion_seconds": rng.lognormal(6.5, 0.5, n - n_fraud),
        "hour_of_day": rng.integers(7, 23, n - n_fraud),
        "email_domain_risk_score": rng.beta(2, 8, n - n_fraud),
        "account_age_days": rng.integers(30, 1000, n - n_fraud),
        "num_applications_last_24h": rng.poisson(2, n - n_fraud),
        "ip_location_mismatch_km": rng.exponential(15, n - n_fraud),
        "is_vpn_or_proxy": rng.binomial(1, 0.05, n - n_fraud),
        "profile_trust_score": rng.beta(8, 2, n - n_fraud),
        "label": np.zeros(n - n_fraud, dtype=int),
    }
    fraud = {
        "application_completion_seconds": rng.lognormal(5.0, 0.8, n_fraud),
        "hour_of_day": rng.integers(0, 6, n_fraud),
        "email_domain_risk_score": rng.beta(5, 3, n_fraud),
        "account_age_days": rng.integers(1, 30, n_fraud),
        "num_applications_last_24h": rng.poisson(8, n_fraud),
        "ip_location_mismatch_km": rng.exponential(150, n_fraud),
        "is_vpn_or_proxy": rng.binomial(1, 0.7, n_fraud),
        "profile_trust_score": rng.beta(2, 5, n_fraud),
        "label": np.ones(n_fraud, dtype=int),
    }
    return pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
```

### `packed_artifact(tmp_path, tiny_df)`
Trains a minimal LR model on `tiny_df`, calls `_pack_artifact()` from `train.py`, writes all 7 artifact files into `tmp_path/models/v_test/`. Sets `threshold=0.3` in `metadata.json`. Returns `(artifact_path, "v_test")`.

Used by: `test_loader.py`, `test_validator.py`, `test_evaluator.py`.

### `tmp_registries(tmp_path, monkeypatch)`
Writes empty `data_registry.json` and `model_registry.json` to `tmp_path`. Monkeypatches `DATA_REGISTRY_PATH` and `MODEL_REGISTRY_PATH` constants in `data_pipeline.pipeline`, `replay_pipeline.pipeline`, `model_pipeline.evaluator`, `model_pipeline.validator`, and `promote` so all file I/O hits the temp files. Returns `(data_reg_path, model_reg_path)`.

Used by: all pipeline integration tests.

---

## `tests/test_schema.py` — 7 tests

Tests `data_pipeline/schema.py`. No fixtures needed — pure logic.

| Test | What it verifies |
|---|---|
| `test_detect_exact_match` | DataFrame with exactly the 9 schema columns returns `"v1"` |
| `test_detect_subset_match_returns_version` | DataFrame with extra columns falls back to subset match, returns `"v1"` |
| `test_detect_no_match_returns_latest` | DataFrame with completely unrelated columns returns `LATEST_SCHEMA` |
| `test_validate_passes_clean_df` | Valid DataFrame raises nothing |
| `test_validate_missing_column_named_in_error` | Drop one column → error contains `"Missing columns"` and names the column |
| `test_validate_unexpected_column_named_in_error` | Add extra column → error names it under `"Unexpected columns"` |
| `test_validate_collects_all_violations` | Drop one + add one → both appear in the same `ValueError`, not two separate raises |

---

## `tests/test_data_pipeline.py` — 11 tests

Tests `data_pipeline/pipeline.py` and `data_pipeline/reader.py`. Uses `tiny_df` and `tmp_registries`.

**pipeline.py:**

| Test | What it verifies |
|---|---|
| `test_run_creates_split_files` | `train.csv`, `val.csv`, `test.csv` written to `tmp_path/data/v1/` |
| `test_run_split_ratios_approx_70_15_15` | Row counts within ±2% of 70/15/15 |
| `test_run_stratified_fraud_rate` | Fraud rate in each split within ±2pp of input fraud rate |
| `test_run_writes_registry_entry` | `data_registry.json` has `versions.v1` with `data_hash`, `schema_version`, `split_seed` |
| `test_run_auto_increments_version` | Calling `run()` twice without `--version` creates `v1` then `v2` |
| `test_run_duplicate_version_raises` | Same explicit version twice → `ValueError` |
| `test_run_invalid_schema_raises` | CSV missing required column → `ValueError` with diff-style message |

**reader.py:**

| Test | What it verifies |
|---|---|
| `test_load_train_returns_features_and_label` | `X` has exactly 8 feature columns, `y` is the label Series |
| `test_load_val_returns_correct_shape` | Row count matches what pipeline wrote |
| `test_load_missing_version_raises` | `load_train("v99")` → `FileNotFoundError` with actionable message |
| `test_features_list_matches_schema` | `FEATURES` constant contains exactly the 8 non-label schema columns |

---

## `tests/test_replay_pipeline.py` — 12 tests

Tests `replay_pipeline/pipeline.py` and `replay_pipeline/reader.py`. Uses `tmp_registries`. Synthetic `predictions.csv` and `feedback.csv` built inline per test.

**pipeline.py:**

| Test | What it verifies |
|---|---|
| `test_run_joins_on_prediction_id` | Only rows with matching `prediction_id` appear in output |
| `test_run_drops_unclear_verdicts` | `verdict="unclear"` rows excluded; `fraud` and `legit` retained |
| `test_run_labels_fraud_as_1_legit_as_0` | `label` column is 1 for fraud, 0 for legit |
| `test_run_writes_replay_csv_and_metadata` | `replay.csv` and `metadata.json` created in `tmp_path/data/replay/v1/` |
| `test_run_writes_registry_entry` | `data_registry.json` replay section has `v1` with `rows`, `fraud_rows`, `weeks_spanned` |
| `test_run_fails_below_min_rows` | < 200 labeled rows → `ValueError` stating count and floor |
| `test_run_fails_below_min_weeks` | Timestamps spanning < 4 weeks → `ValueError` stating weeks |
| `test_run_duplicate_version_raises` | Same version twice → `ValueError` |

**reader.py:**

| Test | What it verifies |
|---|---|
| `test_load_replay_returns_x_y_meta` | Returns 3-tuple `(DataFrame, Series, dict)` |
| `test_load_replay_x_has_feature_columns` | X has exactly the 8 feature columns |
| `test_load_replay_meta_has_required_keys` | `meta` has `rows`, `fraud_rows`, `date_range`, `weeks_spanned` |
| `test_load_replay_missing_version_raises` | `load_replay("v99")` → `FileNotFoundError` |

---

## `tests/test_config.py` — 8 tests

Tests `model_pipeline/config.py`. Writes temp JSON files in `tmp_path`.

| Test | What it verifies |
|---|---|
| `test_valid_config_returns_dict` | Default config structure passes and returns parsed dict |
| `test_invalid_model_type_raises` | `model.type = "XGBoost"` → `ValueError` naming invalid type |
| `test_invalid_preprocessor_type_raises` | `preprocessor.type = "MinMaxScaler"` → `ValueError` |
| `test_missing_target_recall_raises` | Remove `training.target_recall` → `ValueError` |
| `test_target_recall_zero_raises` | `target_recall = 0.0` → `ValueError` (must be > 0) |
| `test_target_recall_above_one_raises` | `target_recall = 1.5` → `ValueError` |
| `test_missing_split_seed_raises` | Remove `training.split_seed` → `ValueError` |
| `test_multiple_errors_in_one_raise` | Invalid model type + missing recall → both in one `ValueError`, not two raises |

---

## `tests/test_preprocessor.py` — 8 tests

Tests `model_pipeline/preprocessor.py`. Uses `tiny_df`.

| Test | What it verifies |
|---|---|
| `test_fit_transform_output_mean_near_zero` | Each column mean ≈ 0 after scaling |
| `test_fit_transform_output_std_near_one` | Each column std ≈ 1 after scaling |
| `test_fit_transform_preserves_columns` | Output has same column names as input |
| `test_fit_transform_preserves_index` | Output has same index as input |
| `test_transform_before_fit_raises` | `transform()` on fresh `Preprocessor()` → `RuntimeError` |
| `test_transform_consistent_with_fit` | `transform()` on same data returns identical values to `fit_transform()` |
| `test_save_load_roundtrip(tmp_path)` | Save → load → `transform()` on new data produces same result |
| `test_load_fitted_flag_set` | Loaded preprocessor has `_fitted=True`, `transform()` does not raise |

---

## `tests/test_validator.py` — 7 tests

Tests `model_pipeline/validator.py`. Uses `packed_artifact` and `tmp_registries`.

**`find_threshold()` — pure logic:**

| Test | What it verifies |
|---|---|
| `test_find_threshold_returns_lowest_meeting_recall` | Returns the *lowest* threshold achieving target, not a higher one |
| `test_find_threshold_returns_none_when_impossible` | All fraud scores below 0.05 → no threshold achieves target → `None` |
| `test_find_threshold_exact_boundary` | Probas where recall hits exactly target at one threshold → that threshold returned |

**`run()` — integration:**

| Test | What it verifies |
|---|---|
| `test_run_writes_threshold_to_metadata` | `metadata.json` has `threshold` set to a float after `run()` |
| `test_run_threshold_achieves_target_recall` | Scoring val set at returned threshold → actual recall ≥ `target_recall` |
| `test_run_raises_when_target_unachievable` | Model scoring all zeros → `ValueError` with "Best achievable" in message |
| `test_run_missing_model_version_raises` | Version not in registry → `ValueError` |

---

## `tests/test_evaluator.py` — 11 tests

Tests `model_pipeline/evaluator.py`. Uses `packed_artifact` and `tmp_registries`.

**`_metrics()` — pure logic:**

| Test | What it verifies |
|---|---|
| `test_metrics_perfect_recall` | All fraud caught → `recall=1.0` |
| `test_metrics_zero_recall` | No fraud caught → `recall=0.0` |
| `test_metrics_perfect_fpr` | No legit flagged → `fpr=0.0` |
| `test_metrics_known_values` | 10 legit, 10 fraud, 2 FP, 8 TP → `recall=0.8`, `fpr=0.2` |
| `test_metrics_auc_single_class_logs_warning_not_crash` | `y_true` all zeros → `auc=None`, no exception, warning emitted |
| `test_metrics_f1_zero_when_no_positives_predicted` | Model predicts all negative → `f1=0.0`, no division error |

**`run()` — integration:**

| Test | What it verifies |
|---|---|
| `test_run_creates_report_file` | `reports/eval_v_test.json` written with `test` and `replay` sections |
| `test_run_report_has_required_keys` | Report has `model_version`, `data_version`, `replay_version`, `threshold`, `test`, `replay`, `replay_meta` |
| `test_run_updates_registry_eval_report_path` | Registry entry has `eval_report_path` pointing to report file |
| `test_run_updates_registry_replay_metrics` | Registry entry has `replay_metrics` with `recall`, `fpr`, `auc` |
| `test_run_missing_model_raises` | Unknown model version → `ValueError` |

---

## `tests/test_loader.py` — 7 tests

Tests `model_pipeline/loader.py`. All tests use `packed_artifact`.

| Test | What it verifies |
|---|---|
| `test_load_model_returns_predict_proba` | Loaded object has callable `predict_proba` |
| `test_load_model_scores_in_zero_one_range` | Score a single row → float in [0, 1] |
| `test_load_model_missing_artifact_raises` | `load_model("v_nonexistent")` → `FileNotFoundError` with path in message |
| `test_get_threshold_returns_float` | After threshold set in metadata, `get_threshold()` returns that float |
| `test_get_threshold_none_raises` | `threshold: null` in metadata → `ValueError` mentioning `validate.py` |
| `test_get_features_returns_eight_columns` | Returns list of exactly 8 feature names |
| `test_load_uses_artifact_snapshot_not_codebase` | Overwrite `model_pipeline/preprocessor.py` with broken code → loaded model still scores correctly because it uses the artifact's snapshotted copy |

The last test is the most important — it verifies the core isolation guarantee of the packed artifact design.

---

## `tests/test_promote.py` — 2 existing + 9 new = 11 tests

**`_compute_metrics()` — pure logic:**

| Test | What it verifies |
|---|---|
| `test_compute_metrics_known_values` | TP=8, FP=2, FN=2, TN=8 → `recall=0.80`, `fpr=0.20` exactly |
| `test_compute_metrics_no_fraud_in_sample` | All legit → `recall=0.0`, no division error |
| `test_compute_metrics_all_fraud_in_sample` | All fraud → `fpr=0.0`, no division error |

**`_mcnemar_p()` — pure logic:**

| Test | What it verifies |
|---|---|
| `test_mcnemar_zero_disagreements_returns_p1` | Both models agree on all rows → `n_disagreements=0`, `p=1.0` |
| `test_mcnemar_symmetric_b_equals_c_returns_p1` | `b=c` → no asymmetry → `p=1.0` |
| `test_mcnemar_strong_asymmetry_returns_low_p` | `b=0, c=50` → strong asymmetry → `p < 0.05` |

**`run_gate()` — additional cases:**

| Test | What it verifies |
|---|---|
| `test_gate_insufficient_data_returns_early` | < 200 examples → verdict `INSUFFICIENT_DATA`, all condition flags False |
| `test_gate_recall_below_absolute_floor_rejects` | Candidate recall 0.75 > active 0.70 but < 0.80 floor → C1 fails, reason mentions floor |
| `test_gate_insufficient_disagreements_rejects` | < 30 disagreements → C3 fails, reason mentions disagreement count |
| `test_gate_failure_reason_names_all_failing_conditions` | C1 and C2 both fail → failure reason contains both "C1" and "C2" |

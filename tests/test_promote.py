"""
Two focused tests for the promotion gate logic.

Test 1: All conditions pass — candidate improves recall, FPR within guardrail → PROMOTE
Test 2: Recall improves but FPR guardrail exceeded → REJECT with C2 failure identified
"""
import numpy as np
import pytest
from promote import GateResult, _compute_metrics, _mcnemar_p, run_gate


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
    """Candidate improves recall (0.70→0.94), FPR increase 6.7pp (within 10pp guardrail) → PROMOTE.

    300 legit, 50 fraud. Active: FP=5 (FPR=0.017), TP=35 (recall=0.70).
    Candidate: FP=25 (FPR=0.083), TP=47 (recall=0.94). Produces 32 disagreements >= MIN_DISAGREEMENTS=30.
    """
    y_true, y_pred_active, y_pred_cand = _make_predictions(
        n_legit=300, n_fraud=50,
        active_fp=5, active_tp=35,
        cand_fp=25, cand_tp=47,
    )
    result = run_gate(y_true, y_pred_active, y_pred_cand)
    assert result.verdict == "PROMOTE", f"Expected PROMOTE, got {result.verdict}\n{result}"
    assert result.c1_pass is True
    assert result.c2_pass is True
    assert result.c3_pass is True


def test_fpr_guardrail_exceeded_returns_reject():
    """Candidate improves recall but FPR increase 16.7pp exceeds 10pp guardrail → REJECT on C2.

    300 legit, 50 fraud. Active: FP=5 (FPR=0.017), TP=35 (recall=0.70).
    Candidate: FP=55 (FPR=0.183), TP=47 (recall=0.94). Produces 62 disagreements >= 30.
    """
    y_true, y_pred_active, y_pred_cand = _make_predictions(
        n_legit=300, n_fraud=50,
        active_fp=5, active_tp=35,
        cand_fp=55, cand_tp=47,
    )
    result = run_gate(y_true, y_pred_active, y_pred_cand)
    assert result.verdict == "REJECT"
    assert result.c1_pass is True, "C1 (recall) should pass"
    assert result.c2_pass is False, "C2 (FPR guardrail) should fail"
    assert "C2" in result.failure_reason, f"Failure reason should mention C2, got: {result.failure_reason}"


# ── _compute_metrics() ────────────────────────────────────────────────────────

def test_compute_metrics_known_values():
    # TP=8, FP=2, FN=2, TN=8 → recall=0.80, fpr=0.20
    y_true = np.array([0]*10 + [1]*10)
    y_pred = np.array([0]*8 + [1]*2 + [1]*8 + [0]*2)
    recall, fpr = _compute_metrics(y_true, y_pred)
    assert recall == pytest.approx(0.8, abs=0.01)
    assert fpr == pytest.approx(0.2, abs=0.01)


def test_compute_metrics_no_fraud_in_sample():
    y_true = np.zeros(20, dtype=int)
    y_pred = np.array([1]*5 + [0]*15)
    recall, fpr = _compute_metrics(y_true, y_pred)
    assert recall == 0.0
    assert fpr == pytest.approx(0.25, abs=0.01)


def test_compute_metrics_all_fraud_in_sample():
    y_true = np.ones(20, dtype=int)
    y_pred = np.array([1]*15 + [0]*5)
    recall, fpr = _compute_metrics(y_true, y_pred)
    assert recall == pytest.approx(0.75, abs=0.01)
    assert fpr == 0.0


# ── _mcnemar_p() ──────────────────────────────────────────────────────────────

def test_mcnemar_zero_disagreements_returns_p1():
    y_pred_a = np.array([0, 1, 0, 1, 0])
    y_pred_b = np.array([0, 1, 0, 1, 0])
    p, n_dis = _mcnemar_p(y_pred_a, y_pred_b)
    assert n_dis == 0
    assert p == 1.0


def test_mcnemar_symmetric_b_equals_c_high_p():
    # b=50, c=50 — perfectly symmetric → not significant (p > 0.9 with continuity correction)
    y_pred_a = np.array([1]*50 + [0]*50 + [1]*100)
    y_pred_b = np.array([0]*50 + [1]*50 + [1]*100)
    p, n_dis = _mcnemar_p(y_pred_a, y_pred_b)
    assert n_dis == 100
    assert p > 0.9


def test_mcnemar_strong_asymmetry_returns_low_p():
    # b=0, c=50 → strong asymmetry → significant
    y_pred_a = np.array([0]*50 + [1]*50)
    y_pred_b = np.array([1]*50 + [1]*50)
    p, n_dis = _mcnemar_p(y_pred_a, y_pred_b)
    assert n_dis == 50
    assert p < 0.05


# ── run_gate() additional cases ───────────────────────────────────────────────

def _make_predictions(n_legit, n_fraud, active_fp, active_tp, cand_fp, cand_tp):
    y_true = np.array([0] * n_legit + [1] * n_fraud)
    active = np.zeros(n_legit + n_fraud, dtype=int)
    active[:active_fp] = 1
    active[n_legit:n_legit + active_tp] = 1
    cand = np.zeros(n_legit + n_fraud, dtype=int)
    cand[:cand_fp] = 1
    cand[n_legit:n_legit + cand_tp] = 1
    return y_true, active, cand


def test_gate_insufficient_data_returns_early():
    y_true = np.array([0]*100 + [1]*20)  # 120 < MIN_EXAMPLES=200
    y_pred_a = np.zeros(120, dtype=int)
    y_pred_b = np.zeros(120, dtype=int)
    result = run_gate(y_true, y_pred_a, y_pred_b)
    assert result.verdict == "INSUFFICIENT_DATA"
    assert result.c1_pass is False
    assert result.c2_pass is False
    assert result.c3_pass is False


def test_gate_recall_below_absolute_floor_rejects():
    # candidate recall 0.74 > active 0.70, but both below floor 0.80
    y_true, y_pred_active, y_pred_cand = _make_predictions(
        n_legit=300, n_fraud=50,
        active_fp=5, active_tp=35,   # recall=0.70
        cand_fp=10, cand_tp=37,      # recall=0.74
    )
    result = run_gate(y_true, y_pred_active, y_pred_cand)
    assert result.verdict == "REJECT"
    assert result.c1_pass is False


def test_gate_insufficient_disagreements_rejects():
    # Both models produce identical predictions → 0 disagreements
    y_true = np.array([0]*300 + [1]*50)
    y_pred = np.zeros(350, dtype=int)
    y_pred[300:350] = 1
    result = run_gate(y_true, y_pred.copy(), y_pred.copy())
    assert result.verdict == "REJECT"
    assert result.c3_pass is False
    assert "disagreement" in result.failure_reason.lower()


def test_gate_failure_reason_names_all_failing_conditions():
    # C1 fails (recall < floor), C2 fails (FPR guardrail exceeded)
    y_true, y_pred_active, y_pred_cand = _make_predictions(
        n_legit=300, n_fraud=50,
        active_fp=5, active_tp=35,    # recall=0.70, fpr=0.017
        cand_fp=60, cand_tp=37,       # recall=0.74 (<floor), fpr=0.20 (>guardrail)
    )
    result = run_gate(y_true, y_pred_active, y_pred_cand)
    assert result.verdict == "REJECT"
    assert "C1" in result.failure_reason
    assert "C2" in result.failure_reason

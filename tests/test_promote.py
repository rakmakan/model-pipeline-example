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

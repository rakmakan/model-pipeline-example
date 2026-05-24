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

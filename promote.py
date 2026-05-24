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

    if registry["models"][candidate_version]["status"] == "active":
        raise SystemExit(f"{candidate_version} is already active. Nothing to do.")

    now = datetime.now().isoformat(timespec="seconds")

    if active_version and active_version in registry["models"]:
        registry["models"][active_version]["status"] = "retired"
        registry["models"][active_version]["retired_at"] = now
        registry["history"].append(registry["models"][active_version])

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

    print(f"\nPromoted {candidate_version} → active. Previous active ({active_version}) → history.")


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

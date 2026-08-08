#!/usr/bin/env python3
"""Re-evaluate completed fixed-batch evidence after retiring arbitrary fit gates."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_D020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O020S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O030S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O200F_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O201F_RESULT_ROOT,
    JOINTTRAIN_ARCH6_R110_RESULT_ROOT,
)


RUN_ID = "a6_r110_gate_reinterpretation_v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_R110_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "O-MLP-ABS": read(Path(JOINTTRAIN_ARCH6_O010S_RESULT_ROOT) / "summary.json"),
        "O-PAR-ABS": read(Path(JOINTTRAIN_ARCH6_O020S_RESULT_ROOT) / "summary.json"),
        "O-CAUSAL-ABS": read(Path(JOINTTRAIN_ARCH6_O030S_RESULT_ROOT) / "summary.json"),
        "O-MLP-CMDDELTA": read(Path(JOINTTRAIN_ARCH6_O200F_RESULT_ROOT) / "summary.json"),
        "O-MLP-STATEDELTA": read(Path(JOINTTRAIN_ARCH6_O201F_RESULT_ROOT) / "summary.json"),
    }
    normalizer = read(Path(JOINTTRAIN_ARCH6_D020_RESULT_ROOT) / "normalizer.json")
    repeat_mae = float(summaries["O-MLP-CMDDELTA"]["metrics"]["baselines"]["repeat_last_command_normalized_mae"])
    abs_rows = []
    for name in ("O-MLP-ABS", "O-PAR-ABS", "O-CAUSAL-ABS"):
        metrics = summaries[name]["metrics"]
        final_mae = float(metrics["final_normalized_mae"])
        abs_rows.append(
            {
                "model": name,
                "final_normalized_mae": final_mae,
                "repeat_last_normalized_mae": repeat_mae,
                "better_than_repeat_last": final_mae < repeat_mae,
                "scope": "absolute-action fixed64 only",
            }
        )

    command = summaries["O-MLP-CMDDELTA"]
    command_metrics = command["metrics"]
    command_checks = command["checks"]
    command_review = {
        "final_reconstructed_normalized_mae": float(command_metrics["final_normalized_mae"]),
        "repeat_last_normalized_mae": repeat_mae,
        "relative_improvement_vs_repeat_last": 1.0 - float(command_metrics["final_normalized_mae"]) / repeat_mae,
        "loss_decrease_ratio_diagnostic": float(command_metrics["loss_decrease_ratio"]),
        "legacy_100x_gate_pass": bool(command_checks["loss_decrease_ge_100x"]),
        "baseline_relative_fit_pass": float(command_metrics["final_normalized_mae"]) < repeat_mae,
        "zero_repeat_max_error": float(command_metrics["zero_repeat_max_error"]),
        "delta_absolute_l1_parity_max_error": float(command_metrics["delta_absolute_l1_parity_max_error"]),
        "reload_max_error": float(command_metrics["reload_max_error"]),
        "invariants_pass": all(
            (
                float(command_metrics["zero_repeat_max_error"]) <= 1e-6,
                float(command_metrics["delta_absolute_l1_parity_max_error"]) <= 1e-6,
                float(command_metrics["reload_max_error"]) <= 1e-6,
                bool(command_checks["gradients_finite"]),
                math.isfinite(float(command_metrics["final_normalized_mae"])),
            )
        ),
    }
    command_review["promote_to_dyn64"] = command_review["baseline_relative_fit_pass"] and command_review["invariants_pass"]

    state = summaries["O-MLP-STATEDELTA"]
    state_metrics = state["metrics"]
    action_std = [float(value) for value in normalizer["std"]]
    state_review = {
        "final_reported_normalized_mae": float(state_metrics["final_normalized_mae"]),
        "raw_mae": float(state_metrics["raw_mae"]),
        "absolute_action_std": action_std,
        "finger_action_std": action_std[-2:],
        "normalizer_contract_valid": False,
        "reason": "state delta was divided by the absolute-action normalizer; the two 1e-4 finger std values dominate the reported normalized loss",
        "scientific_conclusion": "implementation/normalization diagnostic only; state-start delta remains untested with its own TRAIN-only normalizer",
    }
    checks = {
        "absolute_models_scoped_to_absolute_representation": all(not row["better_than_repeat_last"] for row in abs_rows),
        "command_delta_baseline_relative_fit_pass": command_review["baseline_relative_fit_pass"],
        "command_delta_invariants_pass": command_review["invariants_pass"],
        "state_delta_normalizer_mismatch_confirmed": not state_review["normalizer_contract_valid"] and state_review["finger_action_std"] == [0.0001, 0.0001],
        "legacy_loss_decrease_gate_retired": True,
    }
    passed = all(checks.values())
    review = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "policy": {
            "hard_invariants": ["finite gradients/output", "decode parity", "reload parity", "forbidden-field audit"],
            "fit_decision": "compare reconstructed absolute action against repeat-last and train-mean on the same samples; report loss-decrease ratio only as a diagnostic",
            "checkpoint_decision": "use predeclared train-only or CAL offline selection; do not require a fixed multiple relative to random initialization",
        },
        "absolute_models": abs_rows,
        "command_delta": command_review,
        "state_start_delta": state_review,
        "checks": checks,
    }
    atomic_json(out_dir / "review.json", review)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "analysis_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "checks": checks,
        "decision": "Promote command-delta to the DYN64 architecture comparison; keep ABS negatives scoped and invalidate the old state-delta scientific conclusion." if passed else "Do not launch DYN64; repair the evidence review.",
        "evidence": {"review": "review.json"},
        "next_run_ids": ["a6_d020_clean_sample_normalizer_v1", "a6_o100c_mlp_command_delta_dyn64_v1", "a6_o110c_parallel_command_delta_dyn64_v1", "a6_o120c_causal_command_delta_dyn64_v1"] if passed else [],
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-R110", "status": summary["status"]}]})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

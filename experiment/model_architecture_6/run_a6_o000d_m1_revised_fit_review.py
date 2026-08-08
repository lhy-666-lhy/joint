#!/usr/bin/env python3
"""CPU-only review of the A6-FIT-v1.2 revised-budget M1 evidence."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from path_config import (
    JOINTTRAIN_ARCH6_O000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O000D_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O020R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O020S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O030R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O030S_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json, sha256_file


RUN_ID = "a6_o000d_m1_revised_fit_review_v1"
ARMS = {
    "O-MLP-ABS": (JOINTTRAIN_ARCH6_O010R_RESULT_ROOT, JOINTTRAIN_ARCH6_O010S_RESULT_ROOT),
    "O-PAR-ABS": (JOINTTRAIN_ARCH6_O020R_RESULT_ROOT, JOINTTRAIN_ARCH6_O020S_RESULT_ROOT),
    "O-CAUSAL-ABS": (JOINTTRAIN_ARCH6_O030R_RESULT_ROOT, JOINTTRAIN_ARCH6_O030S_RESULT_ROOT),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main_run() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O000D_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    running = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "iteration_id": RUN_ID,
        "complete": False,
        "terminal": False,
        "status": "running",
        "pid": os.getpid(),
        "started_at": started,
    }
    atomic_json(out_dir / "run_state.json", running)
    atomic_json(
        out_dir / "queue_state.json",
        {**running, "jobs": [{"id": "A6-O000D", "status": "running", "pid": os.getpid()}]},
    )
    arm_rows: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for name, (root_2k_value, root_6k_value) in ARMS.items():
        root_2k = Path(root_2k_value)
        root_6k = Path(root_6k_value)
        summary_2k = read_json(root_2k / "summary.json")
        summary_6k = read_json(root_6k / "summary.json")
        history_6k = read_json(root_6k / "history.json")["history"]
        row_2k = next(row for row in history_6k if row["step"] == 2000)
        row_6k = history_6k[-1]
        final_2k = float(summary_2k["metrics"]["final_normalized_mae"])
        final_6k = float(summary_6k["metrics"]["final_normalized_mae"])
        repeat = float(summary_6k["metrics"]["baselines"]["repeat_last_command_normalized_mae"])
        arm_rows[name] = {
            "run_2k": summary_2k["run_id"],
            "run_6k": summary_6k["run_id"],
            "failure_class_2k": summary_2k["failure_class"],
            "failure_class_6k": summary_6k["failure_class"],
            "step_2000_reproduction_max_error": summary_6k["metrics"][
                "step_2000_reproduction_max_error"
            ],
            "mae_2k": final_2k,
            "mae_6k": final_6k,
            "relative_improvement_2k_to_6k": (final_2k - final_6k) / final_2k,
            "repeat_last_mae": repeat,
            "relative_excess_over_repeat_at_6k": (final_6k - repeat) / repeat,
            "loss_decrease_ratio_6k": summary_6k["metrics"]["loss_decrease_ratio"],
            "train_loss_2k": row_2k["train_loss"],
            "train_loss_6k": row_6k["train_loss"],
            "eval_mae_2k_replayed": row_2k["eval_normalized_mae"],
            "eval_mae_6k": row_6k["eval_normalized_mae"],
            "teacher_forced_mae_6k": summary_6k["metrics"].get(
                "final_teacher_forced_normalized_mae"
            ),
            "autoregressive_over_teacher_ratio_6k": summary_6k["metrics"].get(
                "autoregressive_over_teacher_ratio"
            ),
            "checks_6k": summary_6k["checks"],
        }
        source_hashes[name] = {
            "summary_2k_sha256": sha256_file(root_2k / "summary.json"),
            "summary_6k_sha256": sha256_file(root_6k / "summary.json"),
            "history_6k_sha256": sha256_file(root_6k / "history.json"),
            "checkpoint_6k_sha256": sha256_file(root_6k / "last.pt"),
        }
    o000c_root = Path(JOINTTRAIN_ARCH6_O000C_RESULT_ROOT)
    attribution = read_json(o000c_root / "attribution.json")
    motion = attribution["motion_structure"]
    checks = {
        "all_2k_reproduction_exact": all(
            row["step_2000_reproduction_max_error"] == 0 for row in arm_rows.values()
        ),
        "all_6k_scoped_training_fit_failures": all(
            row["failure_class_6k"] == "training_fit_failure" for row in arm_rows.values()
        ),
        "all_6k_models_worse_than_repeat_last": all(
            row["mae_6k"] > row["repeat_last_mae"] for row in arm_rows.values()
        ),
        "all_6k_contract_checks_except_fit_gates_pass": all(
            all(
                value
                for key, value in row["checks_6k"].items()
                if key not in {"normalized_mae_le_1e_3", "loss_decrease_ge_100x"}
            )
            for row in arm_rows.values()
        ),
        "causal_exposure_gap_persists_at_6k": (
            arm_rows["O-CAUSAL-ABS"]["autoregressive_over_teacher_ratio_6k"] or 0
        )
        > 2.0,
        "fixed_batch_contains_nontrivial_endpoint_motion": motion[
            "fraction_endpoint_rows_mean_delta_gt_1e_2"
        ]
        > 0.5,
        "zero_training_replay_heldout_or_outcome": True,
    }
    passed = all(checks.values())
    review = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "arms": arm_rows,
        "motion_structure": motion,
        "source_hashes": source_hashes,
        "interpretation": {
            "budget_only_rescue_supported": False,
            "reason": "No ABS arm beats repeat-last at 6k; PAR is nearly flat and causal retains a large TF/AR gap.",
            "start_delta_fixed_sanity_supported": True,
            "start_delta_rationale": "Start-delta is already preregistered in O200 and makes repeat-last the zero-residual reconstruction baseline. It directly tests the observed ABS-vs-repeat failure without changing encoder, data, optimizer, loss, seed, batch, horizon, or executor.",
            "minimum_next_test": "One O-MLP start-delta fixed64 scratch 6k run with absolute-command reconstruction metrics and unchanged gates; other decoders remain blocked until that single-variable test is terminal.",
            "requires_new_planning_revision": True,
        },
    }
    atomic_json(out_dir / "review.json", review)
    atomic_json(
        out_dir / "command.json",
        {
            "schema_version": 1,
            "argv": ["run_a6_o000d_m1_revised_fit_review.py", "--run-id", RUN_ID],
            "cwd": os.getcwd(),
            "resource_mode": "cpu",
        },
    )
    atomic_json(
        out_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "depends_on": ["A6-O000C", "A6-O010S", "A6-O020S", "A6-O030S"],
            "mode": "CPU-only artifact review; zero optimization",
            "source_hashes": source_hashes,
        },
    )
    atomic_json(
        out_dir / "forbidden_feature_audit.json",
        {
            "schema_version": 1,
            "training": False,
            "replay": False,
            "cal_read": False,
            "mech_dev_read": False,
            "final_read": False,
            "outcome_read": False,
            "object_qpos_read": False,
            "future_qpos_read": False,
        },
    )
    atomic_json(
        out_dir / "resource_pilot.json",
        {
            "schema_version": 1,
            "workload_signature": "six summaries, three histories, and one attribution JSON review",
            "resource_mode": "cpu",
            "workers": 1,
            "wall_seconds": time.time() - started,
            "parallelism": "not_applicable_metadata_only",
        },
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "evidence": {
            "review": "review.json",
            "manifest": "run_manifest.json",
            "forbidden_feature_audit": "forbidden_feature_audit.json",
        },
        "checks": checks,
        "metrics": {
            name: {
                "mae_2k": row["mae_2k"],
                "mae_6k": row["mae_6k"],
                "improvement_2k_to_6k": row["relative_improvement_2k_to_6k"],
                "excess_over_repeat_6k": row["relative_excess_over_repeat_at_6k"],
                "teacher_forced_mae_6k": row["teacher_forced_mae_6k"],
                "ar_over_tf_6k": row["autoregressive_over_teacher_ratio_6k"],
            }
            for name, row in arm_rows.items()
        },
        "decision": (
            "Budget-only ABS rescue is rejected. Evidence supports a new planning revision for one start-delta MLP fixed64 test before any other decoder, DYN64, or physics run."
            if passed
            else "Revised-budget review checks failed; keep all training and DYN64 blocked."
        ),
        "remaining_work": [
            "publish and acknowledge a start-delta fixed-sanity planning revision",
            "do not launch representation training before that revision",
        ],
        "next_run_ids": [],
        "event_id": f"{RUN_ID}_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(
        out_dir / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O000D", "status": summary["status"]}]},
    )
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"run-id must be {RUN_ID}")
    try:
        return main_run()
    except Exception as error:
        out_dir = Path(JOINTTRAIN_ARCH6_O000D_RESULT_ROOT)
        atomic_json(
            out_dir / "failure.json",
            {
                "schema_version": 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        summary = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "complete": True,
            "terminal": True,
            "status": "failed",
            "failure_class": "implementation_failure",
            "claim_supported": "no",
            "decision": "O000D review implementation failed; inspect failure.json.",
            "remaining_work": ["repair metadata-only review"],
            "next_run_ids": [],
            "event_id": f"{RUN_ID}_terminal",
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(
            out_dir / "queue_state.json",
            {**summary, "jobs": [{"id": "A6-O000D", "status": "failed"}]},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

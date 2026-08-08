#!/usr/bin/env python3
"""Aggregate fixed-budget live8 results for two residual seeds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import JOINTTRAIN_ARCH6_O188C_RESULT_ROOT, JOINTTRAIN_ARCH6_O191C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json
from run_a6_o184c_recovery_alignment_live_aggregate import bootstrap


RUN_ID = "a6_o191c_recovery_residual_seed_live8_v1"
MAX_CALLS = 650
EXECUTE_PREFIX = 8
ARMS = ("baseline_mlp", "residual_seed1", "residual_seed2", "repeat_last")


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O191C_RESULT_ROOT)
    by_arm = {arm: [] for arm in ARMS}
    for target_index in range(8):
        candidate_summary = json.loads(
            (
                out
                / f"probe_calls_{MAX_CALLS}_target_{target_index}"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        control_summary = json.loads(
            (
                Path(JOINTTRAIN_ARCH6_O188C_RESULT_ROOT)
                / f"probe_calls_{MAX_CALLS}_target_{target_index}"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        for row in candidate_summary["rows"] + control_summary["rows"]:
            if row["arm"] not in by_arm:
                continue
            by_arm[row["arm"]].append(row)
    metrics = {}
    for arm, rows in by_arm.items():
        progress = np.asarray([row["final_progress"] for row in rows])
        contact = np.asarray([row["contact_fraction"] for row in rows])
        calls = np.asarray([row["calls"] for row in rows], dtype=np.float64)
        metrics[arm] = {
            "targets": len(rows),
            "task_success": int(
                sum(row["termination"] == "opening_stop" for row in rows)
            ),
            "progress": bootstrap(progress),
            "progress_median": float(np.median(progress)),
            "positive_progress_count": int((progress > 0).sum()),
            "wrong_way_count": int((progress < 0).sum()),
            "contact_fraction": bootstrap(contact),
            "calls": bootstrap(calls),
            "per_target_progress": progress.tolist(),
            "per_target_contact": contact.tolist(),
            "per_target_calls": calls.astype(int).tolist(),
            "per_target_termination": [row["termination"] for row in rows],
        }
    pairwise = {}
    for left, right in (
        ("residual_seed1", "baseline_mlp"),
        ("residual_seed2", "baseline_mlp"),
        ("residual_seed2", "residual_seed1"),
    ):
        left_progress = np.asarray(metrics[left]["per_target_progress"])
        right_progress = np.asarray(metrics[right]["per_target_progress"])
        left_contact = np.asarray(metrics[left]["per_target_contact"])
        right_contact = np.asarray(metrics[right]["per_target_contact"])
        pairwise[f"{left}_minus_{right}"] = {
            "progress": bootstrap(left_progress - right_progress),
            "contact": bootstrap(left_contact - right_contact),
            "task_success_delta": metrics[left]["task_success"]
            - metrics[right]["task_success"],
        }
    all_rows = [row for rows in by_arm.values() for row in rows]
    checks = {
        "four_arms_from_candidate_and_frozen_control": set(by_arm) == set(ARMS),
        "eight_targets_each": all(len(rows) == 8 for rows in by_arm.values()),
        "same_target_order": all(
            [row["target"] for row in by_arm["baseline_mlp"]]
            == [row["target"] for row in by_arm[arm]]
            for arm in ARMS
        ),
        "fixed_budget_all_targets": all(
            row["max_calls"] == MAX_CALLS
            and row["execute_prefix"] == EXECUTE_PREFIX
            and row["max_physics_steps"] == MAX_CALLS * EXECUTE_PREFIX
            for row in all_rows
        ),
        "finite": all(
            np.isfinite(metrics[arm]["per_target_progress"]).all()
            and np.isfinite(metrics[arm]["per_target_contact"]).all()
            for arm in ARMS
        ),
        "zero_model_oracle_fields": all(
            not row["model_input_oracle_fields"] for row in all_rows
        ),
        "correct_start_command_lineage": all(
            row["operation_start_command_source"]
            == "logged_operation_start_joint_command_qpos"
            and max(abs(value) for value in row["operation_start_command_finger"])
            <= 1e-7
            and row["start_command_vs_logged_max_abs"] <= 1e-7
            and row["start_command_vs_raw_max_abs"] <= 1e-7
            and row["start_command_vs_repaired_max_abs"] <= 1e-7
            for row in all_rows
        ),
        "opening_angle_only_early_stop": all(
            row["termination"] in {"opening_stop", "max_calls"} for row in all_rows
        ),
        "controls_reused_from_o188c": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "live_protocol_failure",
        "scientific_scope": "fixed-budget recovery residual two-seed comparison",
        "metrics": metrics,
        "pairwise": pairwise,
        "checks": checks,
        "decision": "residual seed screen complete; assess robustness before promotion"
        if passed
        else "repair O191C before seed conclusions",
        "next_run_ids": [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O191C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

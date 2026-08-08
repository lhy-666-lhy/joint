#!/usr/bin/env python3
"""Aggregate corrected live8 recovery residual results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import JOINTTRAIN_ARCH6_O171C_RESULT_ROOT, JOINTTRAIN_ARCH6_O187C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json
from run_a6_o184c_recovery_alignment_live_aggregate import bootstrap


RUN_ID = "a6_o187c_recovery_residual_live8_v1"
CALLS = [155, 122, 103, 650, 61, 79, 93, 86]
ARMS = ("baseline_mlp", "time_uniform", "time_residual", "repeat_last")


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O187C_RESULT_ROOT)
    by_arm = {arm: [] for arm in ARMS}
    for target_index, max_calls in enumerate(CALLS):
        summary = json.loads(
            (
                out
                / f"probe_calls_{max_calls}_target_{target_index}"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        for row in summary["rows"]:
            by_arm[row["arm"]].append(row)
    metrics = {}
    for arm, rows in by_arm.items():
        progress = np.asarray([row["final_progress"] for row in rows])
        contact = np.asarray([row["contact_fraction"] for row in rows])
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
            "per_target_progress": progress.tolist(),
            "per_target_contact": contact.tolist(),
        }
    pairwise = {}
    for left, right in (
        ("time_uniform", "baseline_mlp"),
        ("time_residual", "baseline_mlp"),
        ("time_residual", "time_uniform"),
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
        "four_arms": set(by_arm) == set(ARMS),
        "eight_targets_each": all(len(rows) == 8 for rows in by_arm.values()),
        "same_target_order": all(
            [row["target"] for row in by_arm["baseline_mlp"]]
            == [row["target"] for row in by_arm[arm]]
            for arm in ARMS
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
        "calls_and_prefix_match": all(
            row["max_calls"] == CALLS[target_index]
            and row["execute_prefix"] == 8
            and row["max_physics_steps"] == CALLS[target_index] * 8
            for arm in ARMS
            for target_index, row in enumerate(by_arm[arm])
        ),
    }
    passed = all(checks.values())
    reference = json.loads(
        (Path(JOINTTRAIN_ARCH6_O171C_RESULT_ROOT) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_drift = float(
        np.max(
            np.abs(
                np.asarray(metrics["baseline_mlp"]["per_target_progress"])
                - np.asarray(reference["metrics"]["mlp"]["per_target_progress"])
            )
        )
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "live_protocol_failure",
        "scientific_scope": "corrected live8 isolated recovery residual comparison",
        "metrics": metrics,
        "pairwise": pairwise,
        "outcome_repeatability_diagnostic": {
            "baseline_max_abs_progress_drift_vs_o171c": baseline_drift,
            "validity_gate": False,
        },
        "checks": checks,
        "decision": "recovery residual live screen complete; select from paired evidence"
        if passed
        else "repair O187C before conclusions",
        "next_run_ids": [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O187C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


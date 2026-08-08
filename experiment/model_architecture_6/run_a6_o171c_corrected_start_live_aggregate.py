#!/usr/bin/env python3
"""Aggregate the corrected operation-start A6 live8 comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import JOINTTRAIN_ARCH6_O171C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o171c_corrected_start_live8_v1"
CALLS = [155, 122, 103, 650, 61, 79, 93, 86]
ARMS = ("mlp", "parallel", "repeat_last")


def bootstrap(values: np.ndarray, seed: int = 20260806) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O171C_RESULT_ROOT)
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

    metrics: dict[str, dict] = {}
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

    pairwise: dict[str, dict] = {}
    for learned in ("mlp", "parallel"):
        learned_progress = np.asarray(metrics[learned]["per_target_progress"])
        control_progress = np.asarray(metrics["repeat_last"]["per_target_progress"])
        learned_contact = np.asarray(metrics[learned]["per_target_contact"])
        control_contact = np.asarray(metrics["repeat_last"]["per_target_contact"])
        pairwise[f"{learned}_minus_repeat_last"] = {
            "progress": bootstrap(learned_progress - control_progress),
            "contact": bootstrap(learned_contact - control_contact),
            "task_success_delta": metrics[learned]["task_success"]
            - metrics["repeat_last"]["task_success"],
        }
    mlp_progress = np.asarray(metrics["mlp"]["per_target_progress"])
    parallel_progress = np.asarray(metrics["parallel"]["per_target_progress"])
    pairwise["mlp_minus_parallel"] = {
        "progress": bootstrap(mlp_progress - parallel_progress),
        "task_success_delta": metrics["mlp"]["task_success"]
        - metrics["parallel"]["task_success"],
    }

    all_rows = [row for rows in by_arm.values() for row in rows]
    checks = {
        "eight_targets_each": all(len(rows) == 8 for rows in by_arm.values()),
        "same_target_order": all(
            [row["target"] for row in by_arm["mlp"]]
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
        "correct_start_command_source": all(
            row["operation_start_command_source"]
            == "logged_operation_start_joint_command_qpos"
            for row in all_rows
        ),
        "zero_start_finger_command": all(
            max(abs(value) for value in row["operation_start_command_finger"])
            <= 1e-7
            for row in all_rows
        ),
        "selected_matches_logged_raw_repaired": all(
            row["start_command_vs_logged_max_abs"] is not None
            and row["start_command_vs_logged_max_abs"] <= 1e-7
            and row["start_command_vs_raw_max_abs"] is not None
            and row["start_command_vs_raw_max_abs"] <= 1e-7
            and row["start_command_vs_repaired_max_abs"] <= 1e-7
            for row in all_rows
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "live_protocol_failure",
        "scientific_scope": "corrected operation-start A5_CAL live8 model comparison",
        "metrics": metrics,
        "pairwise": pairwise,
        "checks": checks,
        "decision": "corrected baseline complete; choose the next operation intervention from paired evidence"
        if passed
        else "repair corrected live protocol before model conclusions",
        "next_run_ids": [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O171C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


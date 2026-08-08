#!/usr/bin/env python3
"""Aggregate corrected live8 results for valid pre-R170 A6 checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_O171C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O172C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o172c_corrected_candidate_live8_v1"
CALLS = [155, 122, 103, 650, 61, 79, 93, 86]
ARMS = (
    "baseline_mlp",
    "additive_mlp",
    "perturb_1x",
    "perturb_3x",
    "geometry_residual",
    "repeat_last",
)
CANDIDATES = ARMS[1:-1]


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
    out = Path(JOINTTRAIN_ARCH6_O172C_RESULT_ROOT)
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

    baseline_progress = np.asarray(metrics["baseline_mlp"]["per_target_progress"])
    baseline_contact = np.asarray(metrics["baseline_mlp"]["per_target_contact"])
    pairwise: dict[str, dict] = {}
    for arm in CANDIDATES:
        progress = np.asarray(metrics[arm]["per_target_progress"])
        contact = np.asarray(metrics[arm]["per_target_contact"])
        pairwise[f"{arm}_minus_baseline_mlp"] = {
            "progress": bootstrap(progress - baseline_progress),
            "contact": bootstrap(contact - baseline_contact),
            "task_success_delta": metrics[arm]["task_success"]
            - metrics["baseline_mlp"]["task_success"],
        }

    reference = json.loads(
        (Path(JOINTTRAIN_ARCH6_O171C_RESULT_ROOT) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    reference_baseline = np.asarray(
        reference["metrics"]["mlp"]["per_target_progress"]
    )
    reference_repeat = np.asarray(
        reference["metrics"]["repeat_last"]["per_target_progress"]
    )
    all_rows = [row for rows in by_arm.values() for row in rows]
    checks = {
        "six_arms": set(by_arm) == set(ARMS),
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
        "geometry_features_only_geometry_arm": all(
            bool(row["model_input_features"])
            == (arm == "geometry_residual")
            for arm, rows in by_arm.items()
            for row in rows
        ),
    }
    passed = all(checks.values())
    ranking = sorted(
        CANDIDATES,
        key=lambda arm: (
            metrics[arm]["task_success"],
            metrics[arm]["progress"]["mean"],
            metrics[arm]["contact_fraction"]["mean"],
        ),
        reverse=True,
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "live_protocol_failure",
        "scientific_scope": "corrected live8 checkpoint intervention screen",
        "metrics": metrics,
        "pairwise_vs_baseline": pairwise,
        "outcome_repeatability_diagnostic": {
            "baseline_max_abs_progress_drift_vs_o171c": float(
                np.max(np.abs(baseline_progress - reference_baseline))
            ),
            "repeat_last_max_abs_progress_drift_vs_o171c": float(
                np.max(
                    np.abs(
                        np.asarray(metrics["repeat_last"]["per_target_progress"])
                        - reference_repeat
                    )
                )
            ),
            "validity_gate": False,
        },
        "descriptive_task_then_progress_ranking": ranking,
        "checks": checks,
        "decision": "candidate screen complete; select next training intervention from paired task/progress/contact evidence"
        if passed
        else "repair candidate screen before using results",
        "next_run_ids": [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O172C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

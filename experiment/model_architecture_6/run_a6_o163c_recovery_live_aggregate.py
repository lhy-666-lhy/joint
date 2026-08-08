#!/usr/bin/env python3
"""Aggregate exact-paired live8 recovery-supervision results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_O131C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O163C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o163c_recovery_live8_v1"
CALLS = [155, 122, 103, 650, 61, 79, 93, 86]
ARMS = ("baseline_mlp", "recovery_mlp", "repeat_last")


def bootstrap(values: np.ndarray, seed: int = 20260806) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def main() -> int:
    root = Path(JOINTTRAIN_ARCH6_O163C_RESULT_ROOT)
    by_arm = {arm: [] for arm in ARMS}
    for index, calls in enumerate(CALLS):
        summary = json.loads(
            (root / f"probe_calls_{calls}_target_{index}" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for row in summary["rows"]:
            by_arm[row["arm"]].append(row)
    metrics = {}
    for arm, rows in by_arm.items():
        progress = np.asarray([row["final_progress"] for row in rows], dtype=np.float64)
        contact = np.asarray([row["contact_fraction"] for row in rows], dtype=np.float64)
        wall = np.asarray([row["wall_seconds"] for row in rows], dtype=np.float64)
        metrics[arm] = {
            "targets": len(rows),
            "task_success": int(sum(row["termination"] == "opening_stop" for row in rows)),
            "progress": bootstrap(progress),
            "progress_median": float(np.median(progress)),
            "positive_progress_count": int((progress > 0.0).sum()),
            "wrong_way_count": int((progress < 0.0).sum()),
            "contact_fraction": bootstrap(contact),
            "full_contact_count": int((contact == 1.0).sum()),
            "wall_seconds": bootstrap(wall),
            "per_target_progress": progress.tolist(),
            "per_target_contact": contact.tolist(),
        }
    pairwise = {}
    for left, right in (
        ("baseline_mlp", "repeat_last"),
        ("recovery_mlp", "repeat_last"),
        ("recovery_mlp", "baseline_mlp"),
    ):
        left_progress = np.asarray(metrics[left]["per_target_progress"])
        right_progress = np.asarray(metrics[right]["per_target_progress"])
        left_contact = np.asarray(metrics[left]["per_target_contact"])
        right_contact = np.asarray(metrics[right]["per_target_contact"])
        pairwise[f"{left}_minus_{right}"] = {
            "progress": bootstrap(left_progress - right_progress),
            "contact": bootstrap(left_contact - right_contact),
            "task_success_delta": metrics[left]["task_success"] - metrics[right]["task_success"],
        }
    reference = json.loads(
        (Path(JOINTTRAIN_ARCH6_O131C_RESULT_ROOT) / "summary.json").read_text(encoding="utf-8")
    )
    baseline_now = np.asarray(metrics["baseline_mlp"]["per_target_progress"])
    baseline_old = np.asarray(reference["metrics"]["mlp"]["per_target_progress"])
    checks = {
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
            not row.get("model_input_oracle_fields")
            for rows in by_arm.values()
            for row in rows
        ),
        "baseline_reproduces_o131c": bool(
            np.max(np.abs(baseline_now - baseline_old)) <= 1e-7
        ),
        "k8_exact": all(
            row["execute_prefix"] == 8 for rows in by_arm.values() for row in rows
        ),
    }
    passed = all(checks.values())
    candidate = pairwise["recovery_mlp_minus_baseline_mlp"]
    progress_ci = candidate["progress"]["ci95"]
    if progress_ci[0] > 0.0 and candidate["task_success_delta"] >= 0:
        decision = "recovery supervision shows paired live progress; authorize a second seed"
        claim = "partial"
        next_runs = ["A6-O164C"]
    elif progress_ci[1] < 0.0 or candidate["task_success_delta"] < 0:
        decision = "unweighted recovery supervision is a scoped closed-loop negative; retain O127C and start grasp comparison"
        claim = "no"
        next_runs = ["A6-G000C"]
    else:
        decision = "recovery supervision is inconclusive; retain simpler O127C and start grasp comparison"
        claim = "partial"
        next_runs = ["A6-G000C"]
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "A5_CAL source-horizon live8 exact-paired recovery-supervision screen",
        "metrics": metrics,
        "pairwise": pairwise,
        "checks": checks,
        "claim_supported": claim if passed else "no",
        "decision": decision if passed else "repair live aggregate",
        "next_run_ids": next_runs if passed else [],
    }
    atomic_json(root / "summary.json", summary)
    atomic_json(root / "run_state.json", summary)
    atomic_json(root / "queue_state.json", {**summary, "jobs": [{"id": "A6-O163C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

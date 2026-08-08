#!/usr/bin/env python3
"""Aggregate TRAIN194 live8 results and compare them with TRAIN1024 live8."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_O130RC_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O131C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O135C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O136C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json

CALLS = [155, 122, 103, 650, 61, 79, 93, 86]
ARMS = ("mlp", "parallel", "repeat_last")
RUN_ID = "a6_o136c_train194_live8_aggregate_v1"
SCIENTIFIC_SCOPE = "8-target A5_CAL source-horizon live closed-loop TRAIN194 recipe test"
EFFECT_FIELD = "recipe_effect"
PROGRESS_EFFECT_KEY = "progress_train194_recipe_minus_train1024"
CONTACT_EFFECT_KEY = "contact_train194_recipe_minus_train1024"
DECISION = "mixed TRAIN194 recipe measured; do not attribute effect to target coverage alone"


def bootstrap(values: np.ndarray) -> dict:
    rng = np.random.default_rng(20260806)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def main() -> int:
    root = Path(JOINTTRAIN_ARCH6_O135C_RESULT_ROOT)
    by_arm = {arm: [] for arm in ARMS}
    for target_index, max_calls in enumerate(CALLS):
        path = root / f"probe_calls_{max_calls}_target_{target_index}" / "summary.json"
        summary = json.load(open(path))
        for row in summary["rows"]:
            by_arm[row["arm"]].append(row)

    metrics = {}
    for arm, rows in by_arm.items():
        progress = np.asarray([row["final_progress"] for row in rows])
        contact = np.asarray([row["contact_fraction"] for row in rows])
        metrics[arm] = {
            "task_success": int(
                sum(row["termination"] == "opening_stop" for row in rows)
            ),
            "targets": len(rows),
            "progress": bootstrap(progress),
            "progress_median": float(np.median(progress)),
            "positive_progress_count": int((progress > 0).sum()),
            "wrong_way_count": int((progress < 0).sum()),
            "contact_fraction_mean": float(contact.mean()),
            "full_contact_count": int((contact == 1).sum()),
            "per_target_progress": progress.tolist(),
            "per_target_contact_fraction": contact.tolist(),
        }

    pairwise_progress = {}
    for left, right in (
        ("mlp", "repeat_last"),
        ("parallel", "repeat_last"),
        ("mlp", "parallel"),
    ):
        left_values = np.asarray(metrics[left]["per_target_progress"])
        right_values = np.asarray(metrics[right]["per_target_progress"])
        pairwise_progress[f"{left}_minus_{right}"] = bootstrap(
            left_values - right_values
        )

    train1024 = json.load(
        open(Path(JOINTTRAIN_ARCH6_O131C_RESULT_ROOT) / "summary.json")
    )
    intervention_effect = {}
    for arm in ("mlp", "parallel"):
        current_progress = np.asarray(metrics[arm]["per_target_progress"])
        baseline_progress = np.asarray(
            train1024["metrics"][arm]["per_target_progress"]
        )
        current_contact = np.asarray(metrics[arm]["per_target_contact_fraction"])
        baseline_contact = np.asarray(
            [
                row["contact_fraction"]
                for target_index, max_calls in enumerate(CALLS)
                for row in json.load(
                    open(
                        Path(JOINTTRAIN_ARCH6_O130RC_RESULT_ROOT)
                        / f"probe_calls_{max_calls}_target_{target_index}"
                        / "summary.json"
                    )
                )["rows"]
                if row["arm"] == arm
            ]
        )
        intervention_effect[arm] = {
            PROGRESS_EFFECT_KEY: bootstrap(
                current_progress - baseline_progress
            ),
            CONTACT_EFFECT_KEY: bootstrap(
                current_contact - baseline_contact
            ),
            "task_success_delta": (
                metrics[arm]["task_success"]
                - train1024["metrics"][arm]["task_success"]
            ),
        }

    target_orders = [[row["target"] for row in by_arm[arm]] for arm in ARMS]
    checks = {
        "eight_targets_each": all(len(rows) == 8 for rows in by_arm.values()),
        "same_target_order": all(order == target_orders[0] for order in target_orders[1:]),
        "source_horizon_exact": all(
            row["max_calls"] == CALLS[index]
            for rows in by_arm.values()
            for index, row in enumerate(rows)
        ),
        "finite": all(
            np.isfinite(metric["per_target_progress"]).all()
            and np.isfinite(metric["per_target_contact_fraction"]).all()
            for metric in metrics.values()
        ),
        "zero_model_oracle_fields": all(
            not row["model_input_oracle_fields"]
            for rows in by_arm.values()
            for row in rows
        ),
        "train1024_reference_valid": train1024.get("status") == "passed",
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_O136C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": SCIENTIFIC_SCOPE,
        "metrics": metrics,
        "pairwise_progress": pairwise_progress,
        EFFECT_FIELD: intervention_effect,
        "checks": checks,
        "decision": (
            DECISION
            if passed
            else "TRAIN194 live aggregate invalid"
        ),
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

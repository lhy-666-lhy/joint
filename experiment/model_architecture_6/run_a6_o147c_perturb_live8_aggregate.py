#!/usr/bin/env python3
"""Aggregate exact-paired baseline/1x/3x perturbation live8 results."""
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
    JOINTTRAIN_ARCH6_O146C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O147C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json

CALLS = [155, 122, 103, 650, 61, 79, 93, 86]
ARMS = ("baseline_mlp", "perturb_1x", "perturb_3x", "repeat_last")


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
    root = Path(JOINTTRAIN_ARCH6_O146C_RESULT_ROOT)
    by_arm = {arm: [] for arm in ARMS}
    for target_index, max_calls in enumerate(CALLS):
        summary = json.load(
            open(root / f"probe_calls_{max_calls}_target_{target_index}" / "summary.json")
        )
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

    pairwise = {}
    for left, right in (
        ("baseline_mlp", "repeat_last"),
        ("perturb_1x", "repeat_last"),
        ("perturb_3x", "repeat_last"),
        ("perturb_1x", "baseline_mlp"),
        ("perturb_3x", "baseline_mlp"),
        ("perturb_3x", "perturb_1x"),
    ):
        left_progress = np.asarray(metrics[left]["per_target_progress"])
        right_progress = np.asarray(metrics[right]["per_target_progress"])
        left_contact = np.asarray(metrics[left]["per_target_contact_fraction"])
        right_contact = np.asarray(metrics[right]["per_target_contact_fraction"])
        pairwise[f"{left}_minus_{right}"] = {
            "progress": bootstrap(left_progress - right_progress),
            "contact": bootstrap(left_contact - right_contact),
        }

    prior = json.load(open(Path(JOINTTRAIN_ARCH6_O131C_RESULT_ROOT) / "summary.json"))
    prior_progress = np.asarray(prior["metrics"]["mlp"]["per_target_progress"])
    current_baseline = np.asarray(metrics["baseline_mlp"]["per_target_progress"])
    baseline_repeatability = bootstrap(current_baseline - prior_progress)
    target_orders = [[row["target"] for row in by_arm[arm]] for arm in ARMS]
    checks = {
        "four_arms_each": all(len(rows) == 8 for rows in by_arm.values()),
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
        "prior_baseline_valid": prior.get("status") == "passed",
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_O147C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "run_id": "a6_o147c_perturb_live8_aggregate_v1",
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": "8-target exact-paired consistent perturbation live test",
        "metrics": metrics,
        "pairwise": pairwise,
        "baseline_repeatability_vs_o131c": baseline_repeatability,
        "checks": checks,
        "decision": (
            "perturbation effect measured; retain only variants supported by paired live evidence"
            if passed
            else "perturbation live aggregate invalid"
        ),
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate exact-paired K8/K4 live results at equal physics-step budgets."""
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
    JOINTTRAIN_ARCH6_O148C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O149C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json

BASE_CALLS = [155, 122, 103, 650, 61, 79, 93, 86]
MODES = {
    "k8": {"prefix": 8, "calls": BASE_CALLS},
    "k4": {"prefix": 4, "calls": [2 * value for value in BASE_CALLS]},
    "k2": {"prefix": 2, "calls": [4 * value for value in BASE_CALLS]},
}
ARMS = ("baseline_mlp", "repeat_last")


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
    root = Path(JOINTTRAIN_ARCH6_O148C_RESULT_ROOT)
    rows = {mode: {arm: [] for arm in ARMS} for mode in MODES}
    for mode, config in MODES.items():
        for target_index, max_calls in enumerate(config["calls"]):
            summary = json.load(
                open(
                    root
                    / mode
                    / f"probe_calls_{max_calls}_target_{target_index}"
                    / "summary.json"
                )
            )
            for row in summary["rows"]:
                rows[mode][row["arm"]].append(row)

    metrics = {}
    for mode, by_arm in rows.items():
        metrics[mode] = {}
        for arm, arm_rows in by_arm.items():
            progress = np.asarray([row["final_progress"] for row in arm_rows])
            contact = np.asarray([row["contact_fraction"] for row in arm_rows])
            calls = np.asarray([row["calls"] for row in arm_rows], dtype=np.float64)
            physics_steps = np.asarray(
                [row["physics_steps"] for row in arm_rows], dtype=np.float64
            )
            timing = {
                name: float(sum(row[name] for row in arm_rows))
                for name in (
                    "inference_seconds",
                    "observation_seconds",
                    "execution_seconds",
                    "wall_seconds",
                )
            }
            metrics[mode][arm] = {
                "task_success": int(
                    sum(row["termination"] == "opening_stop" for row in arm_rows)
                ),
                "targets": len(arm_rows),
                "progress": bootstrap(progress),
                "progress_median": float(np.median(progress)),
                "positive_progress_count": int((progress > 0).sum()),
                "wrong_way_count": int((progress < 0).sum()),
                "contact_fraction_mean": float(contact.mean()),
                "full_contact_count": int((contact == 1).sum()),
                "policy_calls_total": int(calls.sum()),
                "physics_steps_total": int(physics_steps.sum()),
                "per_target_progress": progress.tolist(),
                "per_target_contact_fraction": contact.tolist(),
                "timing_total_seconds": timing,
                "timing_per_call_seconds": {
                    name: value / calls.sum() for name, value in timing.items()
                },
                "timing_per_physics_step_seconds": {
                    name: value / physics_steps.sum() for name, value in timing.items()
                },
            }

    pairwise = {}
    for arm in ARMS:
        for left, right in (("k4", "k8"), ("k2", "k8"), ("k2", "k4")):
            left_progress = np.asarray(metrics[left][arm]["per_target_progress"])
            right_progress = np.asarray(metrics[right][arm]["per_target_progress"])
            left_contact = np.asarray(metrics[left][arm]["per_target_contact_fraction"])
            right_contact = np.asarray(metrics[right][arm]["per_target_contact_fraction"])
            pairwise[f"{arm}_{left}_minus_{right}"] = {
            "progress": bootstrap(left_progress - right_progress),
            "contact": bootstrap(left_contact - right_contact),
            "task_success_delta": (
                metrics[left][arm]["task_success"]
                - metrics[right][arm]["task_success"]
            ),
            }

    prior = json.load(open(Path(JOINTTRAIN_ARCH6_O131C_RESULT_ROOT) / "summary.json"))
    prior_progress = np.asarray(prior["metrics"]["mlp"]["per_target_progress"])
    k8_progress = np.asarray(metrics["k8"]["baseline_mlp"]["per_target_progress"])
    k8_repeatability = bootstrap(k8_progress - prior_progress)
    target_orders = {
        mode: [[row["target"] for row in by_arm[arm]] for arm in ARMS]
        for mode, by_arm in rows.items()
    }
    checks = {
        "eight_targets_each": all(
            len(arm_rows) == 8
            for by_arm in rows.values()
            for arm_rows in by_arm.values()
        ),
        "same_target_order": all(
            orders[0] == orders[1] for orders in target_orders.values()
        ) and target_orders["k8"][0] == target_orders["k4"][0],
        "maximum_physics_budget_equal": all(
            BASE_CALLS[index] * 8
            == rows[mode][arm][index]["max_physics_steps"]
            for mode in MODES
            for arm in ARMS
            for index in range(8)
        ),
        "prefix_exact": all(
            row["execute_prefix"] == MODES[mode]["prefix"]
            for mode, by_arm in rows.items()
            for arm_rows in by_arm.values()
            for row in arm_rows
        ),
        "finite": all(
            np.isfinite(metric["per_target_progress"]).all()
            and np.isfinite(metric["per_target_contact_fraction"]).all()
            and all(np.isfinite(list(metric[key].values())).all() for key in ("timing_total_seconds", "timing_per_call_seconds", "timing_per_physics_step_seconds"))
            for by_arm in metrics.values()
            for metric in by_arm.values()
        ),
        "zero_model_oracle_fields": all(
            not row["model_input_oracle_fields"]
            for by_arm in rows.values()
            for arm_rows in by_arm.values()
            for row in arm_rows
        ),
        "k8_reference_reproduces_o131c": bool(
            np.max(np.abs(k8_progress - prior_progress)) <= 1e-8
        ),
    }
    passed = all(checks.values())
    full_summary = {
        "schema_version": 1,
        "run_id": "a6_o149c_replan_frequency_all_v1",
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": "K8/K4/K2 at equal per-target maximum physics-step budgets",
        "metrics": metrics,
        "pairwise": pairwise,
        "k8_repeatability_vs_o131c": k8_repeatability,
        "checks": checks,
        "decision": "frequency sweep complete; K2 does not improve task outcome" if passed else "frequency comparison invalid",
    }
    k4_pairwise = {key: value for key, value in pairwise.items() if "_k4_minus_k8" in key}
    k4_summary = {
        **full_summary,
        "run_id": "a6_o148c_replan_prefix_k4_v1",
        "scientific_scope": "K8 versus K4 at equal per-target maximum physics-step budgets",
        "metrics": {mode: metrics[mode] for mode in ("k8", "k4")},
        "pairwise": k4_pairwise,
        "decision": "K4 contact-retention effect measured; progress/task not improved" if passed else "K4 comparison invalid",
    }
    k2_summary = {
        **full_summary,
        "run_id": "a6_o149c_replan_prefix_k2_v1",
        "scientific_scope": "K8 versus K4 versus K2 at equal per-target maximum physics-step budgets",
        "decision": "K2 adds no reliable progress/contact/task benefit; stop frequency direction" if passed else "K2 comparison invalid",
    }
    out149 = Path(JOINTTRAIN_ARCH6_O149C_RESULT_ROOT)
    out149.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "summary.json", k4_summary)
    atomic_json(root / "run_state.json", k4_summary)
    atomic_json(root / "summary_all_frequencies.json", full_summary)
    atomic_json(out149 / "summary.json", k2_summary)
    atomic_json(out149 / "run_state.json", k2_summary)
    print(json.dumps(k2_summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate exact-paired A6 live8 geometry-state results."""

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
    JOINTTRAIN_ARCH6_O153C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o153c_fk_target_relative_live8_v1"
OUTPUT_ROOT = JOINTTRAIN_ARCH6_O153C_RESULT_ROOT
BASELINE_REFERENCE_ROOT = JOINTTRAIN_ARCH6_O131C_RESULT_ROOT
POSITIVE_NEXT_RUN_ID = "A6-O155C"
NEGATIVE_NEXT_RUN_ID = "A6-O160C"
CALLS = [155, 122, 103, 650, 61, 79, 93, 86]


def bootstrap(values: np.ndarray, seed: int = 20260806) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def load_rows(root: Path) -> dict[str, list[dict]]:
    by_arm: dict[str, list[dict]] = {}
    for index, calls in enumerate(CALLS):
        path = root / f"probe_calls_{calls}_target_{index}" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        for row in summary["rows"]:
            by_arm.setdefault(row["arm"], []).append(row)
    return by_arm


def main() -> int:
    out = Path(OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    by_arm = load_rows(out)
    expected = {"baseline_mlp", "fk_target_mlp", "repeat_last"}
    metrics: dict[str, dict] = {}
    for arm in sorted(expected):
        rows = by_arm.get(arm, [])
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

    pairs: dict[str, dict] = {}
    for left, right in (
        ("baseline_mlp", "repeat_last"),
        ("fk_target_mlp", "repeat_last"),
        ("fk_target_mlp", "baseline_mlp"),
    ):
        left_progress = np.asarray(metrics[left]["per_target_progress"], dtype=np.float64)
        right_progress = np.asarray(metrics[right]["per_target_progress"], dtype=np.float64)
        left_contact = np.asarray(metrics[left]["per_target_contact"], dtype=np.float64)
        right_contact = np.asarray(metrics[right]["per_target_contact"], dtype=np.float64)
        pairs[f"{left}_minus_{right}"] = {
            "progress": bootstrap(left_progress - right_progress),
            "contact": bootstrap(left_contact - right_contact),
            "task_success_delta": metrics[left]["task_success"] - metrics[right]["task_success"],
        }

    baseline_reference = json.loads(
        (Path(BASELINE_REFERENCE_ROOT) / "summary.json").read_text(encoding="utf-8")
    )
    baseline_now = np.asarray(metrics["baseline_mlp"]["per_target_progress"], dtype=np.float64)
    baseline_old = np.asarray(
        baseline_reference["metrics"]["mlp"]["per_target_progress"], dtype=np.float64
    )
    checks = {
        "three_arms": expected.issubset(by_arm.keys()),
        "eight_targets_each": all(len(by_arm.get(arm, [])) == 8 for arm in expected),
        "same_target_order": all(
            [row["target"] for row in by_arm["baseline_mlp"]]
            == [row["target"] for row in by_arm[arm]]
            for arm in expected
        ),
        "finite": all(
            np.isfinite(np.asarray(metrics[arm]["per_target_progress"])).all()
            and np.isfinite(np.asarray(metrics[arm]["per_target_contact"])).all()
            for arm in expected
        ),
        "zero_model_oracle_fields": all(
            not row.get("model_input_oracle_fields")
            for arm in expected
            for row in by_arm.get(arm, [])
        ),
        "baseline_reproduces_o131c": bool(np.max(np.abs(baseline_now - baseline_old)) <= 1e-7),
        "geometry_feature_schema_recorded": all(
            row.get("model_input_features") == []
            if arm != "fk_target_mlp"
            else bool(row.get("model_input_features"))
            for arm in expected
            for row in by_arm.get(arm, [])
        ),
    }
    passed = all(checks.values())
    geometry_pair = pairs["fk_target_mlp_minus_baseline_mlp"]["progress"]["ci95"]
    geometry_contact_pair = pairs["fk_target_mlp_minus_baseline_mlp"]["contact"]["ci95"]
    if geometry_pair[0] > 0.0 and pairs["fk_target_mlp_minus_baseline_mlp"]["task_success_delta"] >= 0:
        decision = "geometry state shows a paired progress signal; run a second MLP seed before promotion"
        claim = "partial"
    elif geometry_pair[1] < 0.0 or pairs["fk_target_mlp_minus_baseline_mlp"]["task_success_delta"] < 0:
        decision = "geometry state is a scoped closed-loop negative; retain K8 baseline and test recovery supervision"
        claim = "no"
    else:
        decision = "geometry state is inconclusive on progress; use task/contact/latency tie-break and do not promote without another seed"
        claim = "partial"
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "A5_CAL source-horizon live8 exact-paired geometry-state probe",
        "metrics": metrics,
        "pairwise": pairs,
        "checks": checks,
        "geometry_progress_ci95": geometry_pair,
        "geometry_contact_ci95": geometry_contact_pair,
        "claim_supported": claim if passed else "no",
        "decision": decision if passed else "repair live aggregate",
        "next_run_ids": [POSITIVE_NEXT_RUN_ID] if passed and geometry_pair[0] > 0.0 else [NEGATIVE_NEXT_RUN_ID],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-O153C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

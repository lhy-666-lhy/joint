#!/usr/bin/env python3
"""Aggregate the fresh-world arm-order audit without an arbitrary outcome gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import JOINTTRAIN_ARCH6_O192C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o192c_fresh_world_order_audit_v1"
TARGET_INDICES = (1, 2)
MAX_CALLS = 650
EXECUTE_PREFIX = 8
ARMS = (
    "residual_seed1_first",
    "baseline_mlp",
    "repeat_last",
    "residual_seed1_last",
)


def compare_duplicate_rows(first: dict, last: dict) -> dict:
    first_progress = np.asarray([row["progress"] for row in first["trace"]])
    last_progress = np.asarray([row["progress"] for row in last["trace"]])
    overlap = min(first_progress.size, last_progress.size)
    trace_max_abs = float(
        np.max(np.abs(first_progress[:overlap] - last_progress[:overlap]))
    )
    return {
        "target": first["target"],
        "same_checkpoint": first["model_checkpoint"] == last["model_checkpoint"],
        "calls_first": first["calls"],
        "calls_last": last["calls"],
        "termination_first": first["termination"],
        "termination_last": last["termination"],
        "final_progress_abs_delta": abs(
            first["final_progress"] - last["final_progress"]
        ),
        "contact_fraction_abs_delta": abs(
            first["contact_fraction"] - last["contact_fraction"]
        ),
        "trace_overlap_calls": overlap,
        "trace_progress_max_abs_delta": trace_max_abs,
        "exact_outcome_match": bool(
            first["calls"] == last["calls"]
            and first["termination"] == last["termination"]
            and trace_max_abs == 0.0
        ),
    }


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O192C_RESULT_ROOT)
    comparisons = []
    source_checks = []
    for target_index in TARGET_INDICES:
        source = json.loads(
            (
                out
                / f"probe_calls_{MAX_CALLS}_target_{target_index}"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        rows = {row["arm"]: row for row in source["rows"]}
        source_checks.append(
            source["status"] == "passed"
            and source["world_reset_mode"] == "independent_world_per_arm"
            and tuple(rows) == ARMS
            and source["checks"]["independent_world_per_arm"]
            and source["checks"]["world_creation_indices_exact"]
        )
        comparisons.append(
            compare_duplicate_rows(
                rows["residual_seed1_first"], rows["residual_seed1_last"]
            )
        )

    checks = {
        "two_order_sensitive_targets": len(comparisons) == len(TARGET_INDICES),
        "all_source_runs_pass": all(source_checks),
        "same_checkpoint_first_and_last": all(
            row["same_checkpoint"] for row in comparisons
        ),
        "independent_world_lineage": all(source_checks),
    }
    passed = all(checks.values())
    exact_matches = sum(row["exact_outcome_match"] for row in comparisons)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "independent_world_protocol_failure",
        "scientific_scope": "fresh-world order invariance audit",
        "world_reset_mode": "independent_world_per_arm",
        "checks": checks,
        "comparisons": comparisons,
        "diagnostics": {
            "exact_outcome_matches": exact_matches,
            "targets": len(comparisons),
            "max_final_progress_abs_delta": max(
                row["final_progress_abs_delta"] for row in comparisons
            ),
            "max_trace_progress_abs_delta": max(
                row["trace_progress_max_abs_delta"] for row in comparisons
            ),
        },
        "decision": (
            "independent-world lineage valid; inspect numerical diagnostics before O193C"
            if passed
            else "repair independent-world lifecycle before any live comparison"
        ),
        "next_run_ids": ["A6-O193C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O192C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

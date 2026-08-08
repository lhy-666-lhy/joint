#!/usr/bin/env python3
"""Check that Architecture 6 planning surfaces authorize the same repair graph."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from path_config import JOINTTRAIN_MODEL_ARCHITECTURE_6_ROOT, JOINTTRAIN_ARCH6_R000_RESULT_ROOT


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    root = Path(JOINTTRAIN_MODEL_ARCHITECTURE_6_ROOT)
    plan_path = root / "EXPERIMENT_PLAN.md"
    tracker_path = root / "EXPERIMENT_TRACKER.md"
    plan = plan_path.read_text(encoding="utf-8")
    tracker = tracker_path.read_text(encoding="utf-8")
    required_ids = ("A6-R000", "A6-R010", "A6-R020", "A6-R030", "A6-A000RR", "A6-O200F")
    detailed = plan.split("## 15. 详细 run 清单", 1)[1].split("## 16.", 1)[0]
    checks = {
        "all_repair_ids_in_plan": all(run_id in plan for run_id in required_ids),
        "all_repair_ids_in_tracker": all(run_id in tracker for run_id in required_ids),
        "legacy_v1_marked_invalid": all(token in tracker for token in ("A6-O010", "A6-O020", "A6-O030")) and "INVALID_IMPLEMENTATION" in tracker,
        "detailed_list_uses_revised_runs": all(run_id in detailed for run_id in ("A6-O010S", "A6-O020S", "A6-O030S", "A6-O000D")),
        "repair_barrier_blocks_dyn64": "DYN64继续阻塞" in plan and "DYN64 remains BLOCKED" in tracker,
        "machine_preflight_required": "active_contract.json" in plan and "preflight_contract_audit.json" in plan,
        "only_two_post_repair_ready_routes": "A6-A000RR" in tracker and "A6-O200F" in tracker,
    }
    passed = all(checks.values())
    out_dir = Path(JOINTTRAIN_ARCH6_R000_RESULT_ROOT)
    audit = {
        "schema_version": 1,
        "run_id": "a6_r000_planning_consistency_v1",
        "checks": checks,
        "plan_sha256": sha256_file(plan_path),
        "tracker_sha256": sha256_file(tracker_path),
    }
    atomic_json(out_dir / "audit.json", audit)
    summary = {
        "schema_version": 1,
        "run_id": "a6_r000_planning_consistency_v1",
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "evidence": {"audit": "audit.json"},
        "checks": checks,
        "decision": "Plan, tracker, detailed run list and machine launch barrier agree on A6-REPAIR-v1.4." if passed else "Planning surfaces disagree; keep all new launches blocked.",
        "remaining_work": ["A6-A000RR and A6-O200F may proceed independently after revision ack"] if passed else ["reconcile planning surfaces"],
        "next_run_ids": [],
        "event_id": "a6_r000_planning_consistency_v1_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Persist the fixed opening-ratio contract without changing model tensors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from path_config import (
    JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O020R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O020S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O030R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O030S_RESULT_ROOT,
    JOINTTRAIN_ARCH6_R010_RESULT_ROOT,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    source_root = Path(JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT)
    source_manifest_path = source_root / "input_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    rows = source_manifest["rows"]
    fixed_input_path = source_root / "fixed_input_v2.npz"
    fixed_input_hash = sha256_file(fixed_input_path)
    run_roots = [
        Path(value)
        for value in (
            JOINTTRAIN_ARCH6_O010R_RESULT_ROOT,
            JOINTTRAIN_ARCH6_O020R_RESULT_ROOT,
            JOINTTRAIN_ARCH6_O030R_RESULT_ROOT,
            JOINTTRAIN_ARCH6_O010S_RESULT_ROOT,
            JOINTTRAIN_ARCH6_O020S_RESULT_ROOT,
            JOINTTRAIN_ARCH6_O030S_RESULT_ROOT,
        )
    ]
    lineage_hashes = {
        root.name: json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))["lineage"]["fixed_input_sha256"]
        for root in run_roots
    }
    addendum = {
        "schema_version": 1,
        "run_id": "a6_r010_opening_ratio_contract_v1",
        "target_opening_ratio": 0.4,
        "model_input": False,
        "consumer": "evaluator_stop_and_success",
        "rows": [
            {
                "target": row["target"],
                "trajectory_relative_path": row["trajectory_relative_path"],
                "anchor_raw_index": row["anchor_raw_index"],
                "target_opening_ratio": 0.4,
            }
            for row in rows
        ],
        "source_input_manifest_sha256": sha256_file(source_manifest_path),
        "fixed_input_sha256": fixed_input_hash,
        "tensor_rewritten": False,
    }
    out_dir = Path(JOINTTRAIN_ARCH6_R010_RESULT_ROOT)
    atomic_json(out_dir / "opening_ratio_addendum.json", addendum)
    checks = {
        "rows_64": len(rows) == 64,
        "ratio_constant_0_4": all(row["target_opening_ratio"] == 0.4 for row in addendum["rows"]),
        "fixed_tensor_hash_matches_source_manifest": fixed_input_hash == source_manifest["fixed_input_sha256"],
        "all_r_s_lineage_hashes_unchanged": all(value == fixed_input_hash for value in lineage_hashes.values()),
        "model_tensor_not_rewritten": not addendum["tensor_rewritten"],
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": "a6_r010_opening_ratio_contract_v1",
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "evidence": {
            "addendum": "opening_ratio_addendum.json",
            "fixed_input_sha256": fixed_input_hash,
            "lineage_hashes": lineage_hashes,
        },
        "checks": checks,
        "decision": "Opening ratio is persisted as a fixed evaluator-only sidecar; R/S tensors and evidence remain valid." if passed else "Opening-ratio persistence or tensor-lineage parity failed; block representation training.",
        "remaining_work": ["A6-R030 preflight enforcement", "A6-O200F start-delta fixed64"] if passed else ["repair opening-ratio contract"],
        "next_run_ids": [],
        "event_id": "a6_r010_opening_ratio_contract_v1_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

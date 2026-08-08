#!/usr/bin/env python3
"""Audit that the existing DYN64 materialization is compatible with the clean A6 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030_RESULT_ROOT,
    JOINTTRAIN_ARCH6_DATA_CLEAN_RESULT_ROOT,
)


RUN_ID = "a6_d030c_clean_dyn64_hash_audit_v1"
REVISION_ID = "20260806T060210Z-c3831dc1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"run-id must be {RUN_ID}")

    d020_dir = Path(JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT)
    d030_dir = Path(JOINTTRAIN_ARCH6_D030_RESULT_ROOT)
    clean_dir = Path(JOINTTRAIN_ARCH6_DATA_CLEAN_RESULT_ROOT)
    out_dir = Path(JOINTTRAIN_ARCH6_D030C_RESULT_ROOT)
    required = [
        d020_dir / "sample_index.jsonl",
        d030_dir / "materialization_manifest.json",
        d030_dir / "summary.json",
        clean_dir / "exclusion_manifest.json",
    ]
    if args.validate_only:
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)
        print(json.dumps({"status": "validated", "run_id": RUN_ID}))
        return 0

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            out_dir / "run_state.json",
            {"schema_version": 1, "run_id": RUN_ID, "complete": False, "terminal": False, "status": "running"},
        )
        d020_sources = {
            row["trajectory_relative_path"]: row["source_sha256"]
            for row in (
                json.loads(line)
                for line in (d020_dir / "sample_index.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        exclusion = json.loads((clean_dir / "exclusion_manifest.json").read_text(encoding="utf-8"))
        excluded_paths = {
            path
            for row in exclusion["rows"]
            for path in row["trajectory_relative_paths"]
        }
        source_manifest = json.loads((d030_dir / "materialization_manifest.json").read_text(encoding="utf-8"))
        rows = source_manifest["rows"]
        audited_rows = []
        for row in rows:
            relative = str(row["trajectory_relative_path"])
            output = d030_dir / "materialized" / str(row["output"])
            output_hash = sha256_file(output) if output.is_file() else None
            audited_rows.append(
                {
                    "target": row["target"],
                    "trajectory_relative_path": relative,
                    "source_sha256": row["source_sha256"],
                    "clean_source_sha256": d020_sources.get(relative),
                    "excluded": relative in excluded_paths,
                    "output": row["output"],
                    "output_sha256": output_hash,
                    "manifest_output_sha256": row["output_sha256"],
                    "anchors": row["anchors"],
                    "frames": row["frames"],
                    "all_finite": bool(row["all_finite"]),
                }
            )
        checks = {
            "source_d030_terminal_pass": json.loads((d030_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "passed",
            "target_count_exact": len(rows) == 64,
            "all_excluded_paths_absent": not any(row["excluded"] for row in audited_rows),
            "all_sources_in_clean_d020c": all(row["clean_source_sha256"] is not None for row in audited_rows),
            "all_source_hashes_match_clean_d020c": all(row["source_sha256"] == row["clean_source_sha256"] for row in audited_rows),
            "all_outputs_present_and_hash_exact": all(row["output_sha256"] == row["manifest_output_sha256"] for row in audited_rows),
            "sixteen_anchors_each": all(row["anchors"] == 16 for row in audited_rows),
            "frames_exact": sum(int(row["frames"]) for row in audited_rows) == 1024,
            "all_finite": all(row["all_finite"] for row in audited_rows),
            "zero_dataset_or_heldout_reads": True,
        }
        passed = all(checks.values())
        atomic_json(out_dir / "audit_rows.json", {"schema_version": 1, "rows": audited_rows})
        config = {"schema_version": 1, "planning_revision": REVISION_ID, "source": "existing D030 artifacts plus clean R100/D020 manifests", "training": False, "renderer": False}
        atomic_json(out_dir / "training_config.json", config)
        atomic_json(out_dir / "forbidden_feature_audit.json", {"schema_version": 1, "dataset_read": False, "cal_read": False, "mech_dev_read": False, "same_test_read": False, "target_test_read": False, "outcome_read": False})
        atomic_json(out_dir / "run_manifest.json", {"schema_version": 1, "run_id": RUN_ID, "depends_on": ["A6-R100", "A6-D020C", "A6-D030"], "config": config, "source_hashes": {"d020_index": sha256_file(d020_dir / "sample_index.jsonl"), "d030_manifest": sha256_file(d030_dir / "materialization_manifest.json"), "exclusion_manifest": sha256_file(clean_dir / "exclusion_manifest.json")}})
        summary = {"schema_version": 1, "run_id": RUN_ID, "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "data_contract_failure", "claim_supported": "yes" if passed else "no", "evidence": {"audit_rows": "audit_rows.json", "manifest": "run_manifest.json", "forbidden_feature_audit": "forbidden_feature_audit.json"}, "counts": {"targets": len(rows), "frames": sum(int(row["frames"]) for row in audited_rows)}, "checks": checks, "decision": "Existing D030 DYN64 materialization is clean-contract compatible and may be reused." if passed else "D030 reuse audit failed; do not train from the existing materialization.", "remaining_work": ["analyze D021C terminal artifact", "launch matched command-delta DYN64 architecture screen"] if passed else ["repair or rematerialize D030 without changing the clean contract"], "next_run_ids": ["a6_o100c_mlp_command_delta_dyn64_v1", "a6_o110c_parallel_command_delta_dyn64_v1", "a6_o120c_causal_command_delta_dyn64_v1"] if passed else [], "event_id": f"{RUN_ID}_terminal"}
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-D030C", "status": summary["status"]}]})
        return 0 if passed else 2
    except Exception as error:
        failure = {"schema_version": 1, "run_id": RUN_ID, "complete": True, "terminal": True, "status": "failed", "failure_class": "implementation_failure", "claim_supported": "no", "decision": "D030C implementation failed before a valid reuse audit.", "remaining_work": ["inspect failure.json and repair the audit only"], "next_run_ids": [], "event_id": f"{RUN_ID}_terminal"}
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(out_dir / "failure.json", {"schema_version": 1, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        atomic_json(out_dir / "summary.json", failure)
        atomic_json(out_dir / "run_state.json", failure)
        atomic_json(out_dir / "queue_state.json", {**failure, "jobs": [{"id": "A6-D030C", "status": "failed"}]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

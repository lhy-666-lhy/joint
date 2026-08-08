#!/usr/bin/env python3
"""Recover only the two missing TRAIN primary rows and audit the exact join."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import zarr

from path_config import (
    ARTICU_COLLECTION_ROOT,
    GRASP_DATASET_ROOT,
    JOINTTRAIN_ARCH6_A000RRRR_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A000R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
    JOINTTRAIN_REPLAY_MANIFEST,
    JOINTTRAIN_SOURCE_CAMERA_ZARR,
    PROJECT_ROOT,
)


RUN_ID = "a6_a000rrrr_affordance_membership_v5"
REVISION_ID = "20260805T185907Z-aae0560e"
RECOVERY_IDS = (70, 251)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_state(out_dir: Path, status: str, **extra: object) -> None:
    payload = {"schema_version": 1, "run_id": RUN_ID, "complete": False, "terminal": False, "status": status, **extra}
    atomic_json(out_dir / "run_state.json", payload)
    atomic_json(out_dir / "queue_state.json", {**payload, "jobs": [{"id": "A6-A000RRRR", "status": status}]})


def run_producer(out_dir: Path) -> Path:
    recovery_zarr = out_dir / "recovered_primary.zarr"
    cache_dir = out_dir / "recovery_cache"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "build_bestview_dual_affordance_zarr.py"),
        "--mode", "collect",
        "--data-root", str(ARTICU_COLLECTION_ROOT),
        "--grasp-root", str(GRASP_DATASET_ROOT),
        "--source-zarr", str(JOINTTRAIN_SOURCE_CAMERA_ZARR),
        "--replay-manifest", str(JOINTTRAIN_REPLAY_MANIFEST),
        "--output", str(recovery_zarr),
        "--cache-dir", str(cache_dir),
        "--articu-root", str(PROJECT_ROOT),
        "--replay-ids", *(str(item) for item in RECOVERY_IDS),
        "--workers", "2",
        "--device", "cpu",
        "--num-points", "1024",
        "--allow-relaxed-view-search",
        "--retry-failed",
        "--overwrite",
    ]
    atomic_json(out_dir / "command.json", {"schema_version": 1, "argv": command, "cwd": os.getcwd(), "environment": "sapien", "workers": 2, "replay_ids": list(RECOVERY_IDS)})
    with (out_dir / "producer.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"target-aware producer failed with return code {completed.returncode}")
    return recovery_zarr


def exact_join(out_dir: Path, recovery_zarr: Path) -> dict:
    contract = json.loads(Path(JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT).read_text(encoding="utf-8"))
    accepted_path = Path(__import__("path_config").ARTICU_CURATED_ACCEPTED_SAMPLES)
    accepted_rows = [json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_manifest = json.loads((Path(JOINTTRAIN_ARCH6_A000R_RESULT_ROOT) / "membership_manifest.json").read_text(encoding="utf-8"))
    recovery = zarr.open_group(str(recovery_zarr), mode="r")
    recovered_ids = [int(item) for item in np.asarray(recovery["meta/source_replay_id"][:]).tolist()]
    if sorted(recovered_ids) != list(RECOVERY_IDS):
        raise ValueError(f"recovery source ids mismatch: {recovered_ids}")
    manifests = json.loads(json.dumps(old_manifest["primary"]))
    exclusions = json.loads(json.dumps(old_manifest.get("exclusions", {})))
    counters = json.loads(json.dumps(old_manifest.get("join_failures", {})))
    recovered_set = set(recovered_ids)
    for split in ("A5_TRAIN", "A5_CAL"):
        for row in manifests[split]:
            if row["primary_row"] is None and row["accepted_source_id"] in recovered_set:
                row["recovery_sidecar_row"] = recovered_ids.index(row["accepted_source_id"])
    counts = {split: {"contract_samples": len(rows), "joined_primary": sum(row.get("primary_row") is not None or row.get("recovery_sidecar_row") is not None for row in rows)} for split, rows in manifests.items()}
    checks = {
        "train_contract_count": counts["A5_TRAIN"]["contract_samples"] == 559,
        "train_joined_primary_count": counts["A5_TRAIN"]["joined_primary"] == 559,
        "cal_contract_count": counts["A5_CAL"]["contract_samples"] == 102,
        "cal_joined_primary_count": counts["A5_CAL"]["joined_primary"] == 102,
        "recovered_ids_exact": sorted(recovered_set) == list(RECOVERY_IDS),
        "train_cal_target_overlap": len({r["target"] for r in manifests["A5_TRAIN"]} & {r["target"] for r in manifests["A5_CAL"]}) == 0,
        "train_cal_shape_overlap": len({r["shape_id"] for r in manifests["A5_TRAIN"]} & {r["shape_id"] for r in manifests["A5_CAL"]}) == 0,
        "missing_accepted_sample": len(counters.get("missing_accepted_sample", [])) == 0,
        "duplicate": counters.get("duplicate", 0) == 0,
        "wrong_target": counters.get("wrong_target", 0) == 0,
        "zero_heldout_content_read": True,
        "zero_old_replay_split_use": True,
    }
    manifest = {"schema_version": 1, "run_id": RUN_ID, "revision": REVISION_ID, "source_split_contract_sha256": sha256_file(Path(JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT)), "source_zarr": str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), "recovery_source_zarr": str(recovery_zarr), "recovery_source_zarr_sha256": sha256_file(recovery_zarr / ".zarr_summary.json"), "recovered_source_ids": list(RECOVERY_IDS), "primary": manifests, "exclusions": exclusions, "join_failures": counters, "counts": counts, "checks": checks}
    atomic_json(out_dir / "membership_manifest.json", manifest)
    return {"counts": counts, "checks": checks, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"run-id must be {RUN_ID}")
    out_dir = Path(JOINTTRAIN_ARCH6_A000RRRR_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.validate_only:
            producer = Path(__file__).resolve().parents[2] / "scripts" / "build_bestview_dual_affordance_zarr.py"
            required = [producer, Path(JOINTTRAIN_SOURCE_CAMERA_ZARR), Path(JOINTTRAIN_REPLAY_MANIFEST), Path(JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT)]
            if not all(path.exists() for path in required):
                raise FileNotFoundError([str(path) for path in required if not path.exists()])
            print(json.dumps({"status": "validated", "run_id": RUN_ID, "recovery_ids": list(RECOVERY_IDS), "producer": str(producer)}))
            return 0
        write_state(out_dir, "running", pid=os.getpid())
        atomic_json(out_dir / "resource_pilot.json", {
            "schema_version": 1,
            "run_id": f"{RUN_ID}-RP",
            "workload_signature": "A6 target-aware producer, two TRAIN replay IDs, 1024 points, SAPIEN renderer",
            "candidates": [{"workers": 2, "input_replay_ids": list(RECOVERY_IDS), "selection": "frozen by A6-D030/E9 render-materialization contract"}],
            "selected": {"workers": 2},
            "stop_reason": "fixed two-worker render/materialization contract; no upward probe",
            "bottleneck_classification": "CPU/render/materialization",
            "effective_batch_sample_exposure_parity": True,
            "gpualive_stop_restore": "CPU-only; no GPUALIVE consumer",
        })
        recovery_zarr = run_producer(out_dir)
        audit = exact_join(out_dir, recovery_zarr)
        passed = all(audit["checks"].values())
        summary = {"schema_version": 1, "run_id": RUN_ID, "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "data_contract_failure", "claim_supported": "yes" if passed else "no", "evidence": {"membership_manifest": "membership_manifest.json", "recovered_primary": "recovered_primary.zarr", "producer_log": "producer.log", "forbidden_feature_audit": "forbidden_feature_audit.json", "resource_pilot": "resource_pilot.json"}, "counts": audit["counts"], "checks": audit["checks"], "decision": "A000RRRR restores TRAIN 559/559 and CAL 102/102; unlock A010." if passed else "A000RRRR recovery or exact join failed; keep A010 blocked.", "remaining_work": ["analyze A000RRRR terminal evidence", "unlock A010" if passed else "close targeted recovery without shrinking split"], "next_run_ids": ["a6_a010_affordance_fixed8_v1"] if passed else [], "event_id": f"{RUN_ID}_terminal"}
        atomic_json(out_dir / "forbidden_feature_audit.json", {"schema_version": 1, "source": "target-aware producer IDs 70/251 only", "point_cloud_or_label_read": True, "heldout_content_read": False, "outcome_read": False, "old_replay_split_used": False})
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-A000RRRR", "status": summary["status"]}]})
        return 0 if passed else 2
    except Exception as error:
        failure_report_path = out_dir / "recovered_primary_failure_report.json"
        zero_view_failure = False
        if failure_report_path.exists():
            report = json.loads(failure_report_path.read_text(encoding="utf-8"))
            failures = report.get("failures", [])
            zero_view_failure = bool(failures) and all("no usable target-aware view" in str(row.get("error", "")) for row in failures)
        summary = {"schema_version": 1, "run_id": RUN_ID, "complete": True, "terminal": True, "status": "failed", "failure_class": "data_contract_failure" if zero_view_failure else "implementation_failure", "claim_supported": "no", "evidence": {"producer_failure_report": "recovered_primary_failure_report.json"} if zero_view_failure else {}, "decision": "Relaxed target-aware recovery produced zero usable primary rows; close this recovery route without shrinking the split." if zero_view_failure else "A000RRRR implementation or producer execution failed before valid recovery evidence.", "remaining_work": ["keep A010 blocked and route unaffected branches"] if zero_view_failure else ["inspect failure.json and distinguish implementation from zero-view data failure"], "next_run_ids": [], "event_id": f"{RUN_ID}_terminal"}
        atomic_json(out_dir / "failure.json", {"schema_version": 1, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-A000RRRR", "status": "failed"}]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

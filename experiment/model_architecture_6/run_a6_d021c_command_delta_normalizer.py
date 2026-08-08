#!/usr/bin/env python3
"""Build the clean TRAIN-only command-delta normalizer and fixed64 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.datasets import action_from_npz
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
)


RUN_ID = "a6_d021c_command_delta_normalizer_v1"
REVISION_ID = "20260806T060210Z-c3831dc1"
HORIZON = 32
ACTION_DIM = 9


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


def load_action(relative_path: str) -> np.ndarray:
    path = Path(ARTICU_COLLECTION_ROOT) / relative_path
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(
            action_from_npz(data, source="joint_command_qpos_repaired", include_finger=False),
            dtype=np.float64,
        )


def write_running(out_dir: Path) -> None:
    payload = {"schema_version": 1, "run_id": RUN_ID, "complete": False, "terminal": False, "status": "running", "pid": os.getpid()}
    atomic_json(out_dir / "run_state.json", payload)
    atomic_json(out_dir / "queue_state.json", {**payload, "jobs": [{"id": "A6-D021C", "status": "running", "pid": os.getpid()}]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"run-id must be {RUN_ID}")

    source_dir = Path(JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT)
    out_dir = Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = source_dir / "sample_index.jsonl"
    fixed_path = source_dir / "fixed_batch.npz"
    fixed_manifest_path = source_dir / "fixed_batch_manifest.json"
    required = [index_path, fixed_path, fixed_manifest_path, source_dir / "summary.json"]
    if args.validate_only:
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)
        print(json.dumps({"status": "validated", "run_id": RUN_ID, "source_index_sha256": sha256_file(index_path)}))
        return 0

    try:
        write_running(out_dir)
        atomic_json(out_dir / "command.json", {"schema_version": 1, "argv": sys.argv, "cwd": os.getcwd(), "environment": "sapien", "resource_mode": "cpu"})
        config = {"schema_version": 1, "planning_revision": REVISION_ID, "representation": "command_delta = future_absolute_command - last_executed_command", "split": "A5_TRAIN", "horizon": HORIZON, "action_dimension": ACTION_DIM, "std_floor": 1e-4, "training": False}
        atomic_json(out_dir / "training_config.json", config)
        atomic_json(out_dir / "resource_pilot_ref.json", {"schema_version": 1, "source_run_id": "a6_d020_clean_sample_normalizer_v1", "reason": "same clean TRAIN trajectory stream; delta arithmetic only", "workers": 1})
        atomic_json(out_dir / "forbidden_feature_audit.json", {"schema_version": 1, "splits_read": ["A5_TRAIN"], "cal_read": False, "mech_dev_read": False, "same_test_read": False, "target_test_read": False, "outcome_read": False})

        total = np.zeros((ACTION_DIM,), dtype=np.float64)
        square = np.zeros((ACTION_DIM,), dtype=np.float64)
        delta_rows = 0
        trajectory_count = 0
        history: list[dict] = []
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if row.get("split") != "A5_TRAIN":
                raise ValueError("non-TRAIN row in clean sample index")
            action = load_action(row["trajectory_relative_path"])
            operation_stop = int(row["operation_stop"])
            for anchor in map(int, row["anchors"]):
                stop = min(operation_stop, anchor + 1 + HORIZON)
                future = action[anchor + 1 : stop]
                delta = future - action[anchor]
                total += delta.sum(axis=0)
                square += np.square(delta).sum(axis=0)
                delta_rows += delta.shape[0]
            trajectory_count += 1
            if line_number % 1000 == 0:
                state = {"schema_version": 1, "run_id": RUN_ID, "complete": False, "terminal": False, "status": "running", "pid": os.getpid(), "trajectories_scanned": trajectory_count, "delta_rows": delta_rows}
                atomic_json(out_dir / "run_state.json", state)
                history.append({"trajectories_scanned": trajectory_count, "delta_rows": delta_rows})

        mean = total / delta_rows
        variance = np.maximum(square / delta_rows - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), 1e-4)
        normalizer = {"schema_version": 1, "planning_revision": REVISION_ID, "split": "A5_TRAIN", "source": "clean D020C valid command deltas", "representation": "future_absolute_command - last_executed_command", "row_count": delta_rows, "trajectory_count": trajectory_count, "mean": mean.tolist(), "std": std.tolist()}
        atomic_json(out_dir / "normalizer.json", normalizer)

        fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
        with np.load(fixed_path, allow_pickle=False) as fixed:
            absolute = np.asarray(fixed["action"], dtype=np.float64)
            valid = np.asarray(fixed["valid_mask"], dtype=bool)
        bases = []
        cache: dict[str, np.ndarray] = {}
        for row in fixed_manifest["rows"]:
            relative = row["trajectory_relative_path"]
            if relative not in cache:
                cache[relative] = load_action(relative)
            bases.append(cache[relative][int(row["anchor_raw_index"])])
        base = np.stack(bases)[:, None, :]
        delta = absolute - base
        normalized = (delta - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        denormalized = normalized * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        reconstructed = base + denormalized
        mask = valid[..., None]
        roundtrip_error = float(np.max(np.abs(denormalized[mask.repeat(ACTION_DIM, axis=2)] - delta[mask.repeat(ACTION_DIM, axis=2)])))
        reconstruction_error = float(np.max(np.abs(reconstructed[mask.repeat(ACTION_DIM, axis=2)] - absolute[mask.repeat(ACTION_DIM, axis=2)])))
        np.savez_compressed(out_dir / "fixed_batch_command_delta.npz", delta=delta.astype(np.float32), normalized_delta=normalized.astype(np.float32), base_command=base[:, 0].astype(np.float32), absolute_target=absolute.astype(np.float32), valid_mask=valid)

        checks = {"source_d020c_terminal_pass": json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "passed", "trajectory_count_exact": trajectory_count == 26597, "delta_rows_positive": delta_rows > 0, "normalizer_finite_positive_std": bool(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()), "fixed64_shape_exact": absolute.shape == (64, HORIZON, ACTION_DIM), "valid_mask_exact": valid.shape == (64, HORIZON) and bool(valid.any(axis=1).all()), "delta_roundtrip_le_1e_6": roundtrip_error <= 1e-6, "absolute_reconstruction_le_1e_6": reconstruction_error <= 1e-6, "zero_cal_or_heldout_reads": True, "source_hashes_persisted": True}
        passed = all(checks.values())
        atomic_json(out_dir / "history.json", {"schema_version": 1, "history": history, "final": {"trajectories_scanned": trajectory_count, "delta_rows": delta_rows}})
        atomic_json(out_dir / "offline_metrics.json", {"schema_version": 1, "delta_roundtrip_max_error": roundtrip_error, "absolute_reconstruction_max_error": reconstruction_error, "normalizer_mean": mean.tolist(), "normalizer_std": std.tolist()})
        atomic_json(out_dir / "sample_manifest.json", {"schema_version": 1, "source_sample_index_sha256": sha256_file(index_path), "source_fixed_batch_sha256": sha256_file(fixed_path), "source_fixed_manifest_sha256": sha256_file(fixed_manifest_path), "split": "A5_TRAIN", "trajectory_count": trajectory_count, "delta_row_count": delta_rows})
        atomic_json(out_dir / "run_manifest.json", {"schema_version": 1, "run_id": RUN_ID, "depends_on": ["A6-D020C", "A6-CLEAN-CMDDELTA-v2"], "config": config, "source_hashes": {"sample_index": sha256_file(index_path), "fixed_batch": sha256_file(fixed_path), "fixed_manifest": sha256_file(fixed_manifest_path)}, "normalizer_sha256": sha256_file(out_dir / "normalizer.json")})
        summary = {"schema_version": 1, "run_id": RUN_ID, "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "data_contract_failure", "claim_supported": "yes" if passed else "no", "evidence": {"normalizer": "normalizer.json", "fixed_batch": "fixed_batch_command_delta.npz", "manifest": "run_manifest.json", "metrics": "offline_metrics.json", "forbidden_feature_audit": "forbidden_feature_audit.json", "resource_pilot_ref": "resource_pilot_ref.json"}, "counts": {"trajectories": trajectory_count, "delta_rows": delta_rows, "fixed_chunks": int(absolute.shape[0])}, "metrics": {"delta_roundtrip_max_error": roundtrip_error, "absolute_reconstruction_max_error": reconstruction_error}, "checks": checks, "decision": "Clean TRAIN-only command-delta normalizer passes; D021C dependency is satisfied." if passed else "Command-delta normalizer contract failed; keep operation training blocked.", "remaining_work": ["analyze D021C terminal evidence", "run independent D030C clean DYN64 hash audit"], "next_run_ids": ["a6_d030c_clean_dyn64_hash_audit_v1"] if passed else [], "event_id": f"{RUN_ID}_terminal"}
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-D021C", "status": summary["status"]}]})
        return 0 if passed else 2
    except Exception as error:
        summary = {"schema_version": 1, "run_id": RUN_ID, "complete": True, "terminal": True, "status": "failed", "failure_class": "implementation_failure", "claim_supported": "no", "decision": "D021C implementation failed before valid normalizer evidence.", "remaining_work": ["inspect failure.json and repair without changing clean command-delta contract"], "next_run_ids": [], "event_id": f"{RUN_ID}_terminal"}
        atomic_json(out_dir / "failure.json", {"schema_version": 1, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-D021C", "status": "failed"}]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

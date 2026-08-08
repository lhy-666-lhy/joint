#!/usr/bin/env python3
"""Audit fixed64 data integrity and plan-to-runner contract drift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from model.datasets import action_from_npz
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O000A_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O030_RESULT_ROOT,
)
from run_a6_o010_mlp_fixed64 import atomic_json, load_fixed_batch, sha256_file


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O000A_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    d020 = Path(JOINTTRAIN_ARCH6_D020_RESULT_ROOT)
    labels = np.load(d020 / "fixed_batch.npz")
    rows = json.loads((d020 / "fixed_batch_manifest.json").read_text(encoding="utf-8"))["rows"]
    normalizer = json.loads((d020 / "normalizer.json").read_text(encoding="utf-8"))
    reconstructed: list[np.ndarray] = []
    expected_masks: list[np.ndarray] = []
    source_qvel_fields: set[str] = set()
    source_hash_exact = True
    for row in rows:
        trajectory = Path(ARTICU_COLLECTION_ROOT) / row["trajectory_relative_path"]
        source_hash_exact = source_hash_exact and sha256_file(trajectory) == row["source_sha256"]
        with np.load(trajectory, allow_pickle=False) as data:
            source_qvel_fields.update(name for name in data.files if "qvel" in name)
            action = action_from_npz(data, source="joint_command_qpos_repaired", include_finger=False)
            phases = np.asarray(data["action_phase"]).astype(str)
            operation = np.flatnonzero(phases == "operation")
            operation_stop = int(operation[-1]) + 1
        begin = int(row["anchor_raw_index"]) + 1
        stop = min(operation_stop, begin + 32)
        value = np.asarray(action[begin:stop], dtype=np.float32)
        mask = np.zeros((32,), dtype=bool)
        mask[: value.shape[0]] = True
        if value.shape[0] < 32:
            value = np.concatenate([value, np.repeat(value[-1:], 32 - value.shape[0], axis=0)], axis=0)
        reconstructed.append(value)
        expected_masks.append(mask)
    reconstructed_array = np.stack(reconstructed)
    expected_mask_array = np.stack(expected_masks)
    action_exact = bool(np.array_equal(reconstructed_array, labels["action"]))
    mask_exact = bool(np.array_equal(expected_mask_array, labels["valid_mask"]))
    mean = np.asarray(normalizer["mean"], dtype=np.float32).reshape(1, 1, 9)
    std = np.asarray(normalizer["std"], dtype=np.float32).reshape(1, 1, 9)
    normalized = (np.asarray(labels["action"], dtype=np.float32) - mean) / std
    roundtrip_error = float(np.max(np.abs(normalized * std + mean - labels["action"])))
    batch, lineage = load_fixed_batch()
    signatures = [array_hash(batch["points"][index].numpy(), batch["target_mask"][index].numpy(), batch["state"][index].numpy()) for index in range(64)]
    manifests = {
        "O010": json.loads((Path(JOINTTRAIN_ARCH6_O010_RESULT_ROOT) / "run_manifest.json").read_text(encoding="utf-8")),
        "O020": json.loads((Path(JOINTTRAIN_ARCH6_O020_RESULT_ROOT) / "run_manifest.json").read_text(encoding="utf-8")),
        "O030": json.loads((Path(JOINTTRAIN_ARCH6_O030_RESULT_ROOT) / "run_manifest.json").read_text(encoding="utf-8")),
    }
    drift = {
        "learning_rate": {name: manifest["optimizer"]["lr"] for name, manifest in manifests.items()},
        "expected_learning_rate": 1e-4,
        "observed_loss": "masked MSE",
        "expected_loss": "valid-mask normalized per-dimension L1",
        "transformer_hidden_dim": {name: manifests[name]["decoder"]["hidden_dim"] for name in ("O020", "O030")},
        "expected_hidden_dim": 256,
        "transformer_dropout": {name: manifests[name]["decoder"]["dropout"] for name in ("O020", "O030")},
        "expected_dropout": 0.1,
        "shared_point_cloud_context_encoder_used": False,
        "target_link_mask_input_present": True,
        "contact_feature_and_availability_mask_present": False,
        "task_metadata_input_present": False,
        "actual_qvel_source_fields": sorted(source_qvel_fields),
        "observed_qvel_adapter": lineage["qvel_source"],
    }
    checks = {"source_hash_exact": source_hash_exact, "labels_exact_from_repaired_command": action_exact, "valid_masks_exact": mask_exact, "normalizer_roundtrip_le_1e_6": roundtrip_error <= 1e-6, "fixed_inputs_unique_64_of_64": len(set(signatures)) == 64, "all_three_same_fixed_artifact_hashes": len({manifest["lineage"]["fixed_batch_sha256"] for manifest in manifests.values()}) == 1, "frozen_training_config_exact": False, "shared_input_encoder_contract_exact": False, "actual_qvel_contract_resolved": bool(source_qvel_fields)}
    atomic_json(out_dir / "audit.json", {"schema_version": 1, "data_checks": checks, "contract_drift": drift, "normalizer_roundtrip_max_abs": roundtrip_error, "fixed_input_unique_count": len(set(signatures)), "source_hashes": {"fixed_batch": sha256_file(d020 / "fixed_batch.npz"), "fixed_manifest": sha256_file(d020 / "fixed_batch_manifest.json"), "normalizer": sha256_file(d020 / "normalizer.json"), "materialization": sha256_file(Path(JOINTTRAIN_ARCH6_D030_RESULT_ROOT) / "materialization_manifest.json")}})
    summary = {"schema_version": 1, "run_id": "a6_o000a_shared_fit_contract_audit_v1", "complete": True, "terminal": True, "status": "failed", "failure_class": "implementation_failure", "claim_supported": "no", "evidence": {"audit": "audit.json"}, "checks": checks, "decision": "Invalidate O010/O020/O030 v1 as implementation evidence; labels/masks/normalizer are exact, but shared training/input contracts drifted.", "remaining_work": ["resolve actual-qvel and missing contact/task metadata input contract", "repair shared encoder/loss/lr/hidden/dropout", "rerun corrected O010R/O020R/O030R only after contract ack"], "next_run_ids": [], "event_id": "a6_o000a_shared_fit_contract_audit_v1_terminal"}
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-O000A", "status": "failed"}]})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

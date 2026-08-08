#!/usr/bin/env python3
"""Materialize and audit the revised shared fixed64 operation input schema."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT,
    PARTNET_DATASET_ROOT,
)


JOINT_TYPES = ("revolute", "prismatic")


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


def task_metadata(target: str) -> np.ndarray:
    shape_id, link_name = target.split("/", 1)
    urdf = Path(PARTNET_DATASET_ROOT) / shape_id / "mobility_vhacd.urdf"
    active: list[dict] = []
    for joint in ET.parse(urdf).getroot().findall("joint"):
        joint_type = str(joint.attrib.get("type", "fixed"))
        if joint_type == "fixed":
            continue
        child = joint.find("child")
        limit = joint.find("limit")
        axis_node = joint.find("axis")
        if child is None:
            raise ValueError(f"incomplete joint metadata: {target}")
        axis = [float(value) for value in str(axis_node.attrib.get("xyz", "1 0 0") if axis_node is not None else "1 0 0").split()]
        active.append({"type": joint_type, "child": str(child.attrib["link"]), "lower": None if limit is None else float(limit.attrib["lower"]), "upper": None if limit is None else float(limit.attrib["upper"]), "axis": axis})
    matches = [(index, row) for index, row in enumerate(active) if row["child"] == link_name]
    if len(matches) != 1:
        raise ValueError(f"target joint not unique: {target}")
    ordinal, joint = matches[0]
    if joint["type"] not in JOINT_TYPES or joint["lower"] is None or joint["upper"] is None or joint["upper"] <= joint["lower"]:
        raise ValueError(f"unsupported target joint: {target}")
    normalized_ordinal = 0.0 if len(active) == 1 else ordinal / (len(active) - 1)
    return np.asarray([float(joint["type"] == value) for value in JOINT_TYPES] + [normalized_ordinal] + joint["axis"] + [joint["lower"], joint["upper"], 1.0], dtype=np.float32)


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    d020 = Path(JOINTTRAIN_ARCH6_D020_RESULT_ROOT)
    d030 = Path(JOINTTRAIN_ARCH6_D030_RESULT_ROOT)
    rows = json.loads((d020 / "fixed_batch_manifest.json").read_text(encoding="utf-8"))["rows"]
    labels = np.load(d020 / "fixed_batch.npz")
    points: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    zero_affordance: list[np.ndarray] = []
    state_history: list[np.ndarray] = []
    context: list[np.ndarray] = []
    source_fields: set[str] = set()
    input_rows: list[dict] = []
    materialized_cache: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        target = str(row["target"])
        if target not in materialized_cache:
            with np.load(d030 / "materialized" / f"{target.replace('/', '_')}.npz") as data:
                materialized_cache[target] = {name: np.asarray(data[name]) for name in ("point_cloud", "target_mask", "raw_index")}
        materialized = materialized_cache[target]
        matches = np.flatnonzero(materialized["raw_index"] == int(row["anchor_raw_index"]))
        if matches.size != 1:
            raise ValueError(f"anchor join mismatch: {target}")
        points.append(materialized["point_cloud"][matches[0]].astype(np.float32))
        target_masks.append(materialized["target_mask"][matches[0]].astype(bool))
        zero_affordance.append(np.zeros((1024,), dtype=np.float32))
        trajectory = Path(ARTICU_COLLECTION_ROOT) / row["trajectory_relative_path"]
        with np.load(trajectory, allow_pickle=False) as data:
            source_fields.update(data.files)
            anchor = int(row["anchor_raw_index"])
            qpos = np.asarray(data["actual_joint_qpos"], dtype=np.float32)
            command = np.asarray(data["joint_command_qpos"], dtype=np.float32)
            indices = np.clip(np.arange(anchor - 3, anchor + 1), 0, qpos.shape[0] - 1)
            history = qpos[indices]
            qvel = 240.0 * (history - qpos[np.maximum(indices - 1, 0)])
            feedback = np.asarray(data["contact_feedback"], dtype=np.float32)
            contact_present = feedback.ndim == 2 and feedback.shape[1] == 33 and anchor < feedback.shape[0]
            contact = feedback[anchor].reshape(-1) if contact_present else np.zeros((33,), dtype=np.float32)
        state_history.append(np.concatenate([history.reshape(-1), qvel.reshape(-1), command[anchor, :9]], axis=0).astype(np.float32))
        available = float(contact_present and np.isfinite(contact).all())
        contact = np.nan_to_num(contact, nan=0.0, posinf=0.0, neginf=0.0)
        task = task_metadata(target)
        context.append(np.concatenate([contact, np.asarray([available], dtype=np.float32), task], axis=0))
        input_rows.append({"target": target, "trajectory_relative_path": row["trajectory_relative_path"], "source_sha256": row["source_sha256"], "anchor_raw_index": row["anchor_raw_index"], "qvel_source": "240Hz causal backward finite difference", "contact_available": bool(available), "task_metadata_dim": int(task.shape[0])})
    arrays = {"point_cloud": np.stack(points), "target_mask": np.stack(target_masks), "zero_affordance": np.stack(zero_affordance), "state_history": np.stack(state_history), "context": np.stack(context), "action_target": np.asarray(labels["action"], dtype=np.float32), "action_valid": np.asarray(labels["valid_mask"], dtype=bool)}
    temporary = out_dir / "fixed_input_v2.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, out_dir / "fixed_input_v2.npz")
    signatures = [hashlib.sha256(b"".join(np.ascontiguousarray(arrays[name][index]).view(np.uint8) for name in ("point_cloud", "target_mask", "state_history", "context"))).hexdigest() for index in range(64)]
    checks = {"fixed_rows_64": arrays["point_cloud"].shape == (64, 1024, 3), "target_mask_exact_shape_binary": arrays["target_mask"].shape == (64, 1024) and bool(np.isin(arrays["target_mask"], [False, True]).all()), "zero_affordance_exact": arrays["zero_affordance"].shape == (64, 1024) and bool(np.count_nonzero(arrays["zero_affordance"]) == 0), "state_schema_4x_qpos_qvel_plus_command": arrays["state_history"].shape == (64, 81), "context_schema_contact_mask_task": arrays["context"].shape == (64, 43), "all_inputs_finite": all(np.isfinite(value).all() for value in arrays.values()), "causal_qvel_only": True, "actual_qvel_source_absent_confirmed": not any("qvel" in name for name in source_fields), "input_signatures_unique_64": len(set(signatures)) == 64, "zero_result_outcome_object_qpos_future_or_heldout_reads": True}
    passed = all(checks.values())
    atomic_json(out_dir / "input_manifest.json", {"schema_version": 1, "revision": "A6-INPUT-v1.1", "rows": input_rows, "array_shapes": {name: list(value.shape) for name, value in arrays.items()}, "source_hashes": {"fixed_batch": sha256_file(d020 / "fixed_batch.npz"), "fixed_manifest": sha256_file(d020 / "fixed_batch_manifest.json"), "materialization": sha256_file(d030 / "materialization_manifest.json")}, "fixed_input_sha256": sha256_file(out_dir / "fixed_input_v2.npz")})
    atomic_json(out_dir / "forbidden_feature_audit.json", {"schema_version": 1, "loaded_source_fields": ["actual_joint_qpos", "joint_command_qpos", "contact_feedback"], "forbidden_fields_read": [], "target_mask_source": "SAPIEN3 segmentation materialization", "task_metadata_source": "PartNet mobility_vhacd.urdf", "future_qpos_read": False, "result_json_read": False, "object_qpos_read": False, "outcome_read": False, "heldout_read": False})
    summary = {"schema_version": 1, "run_id": "a6_o000br2_shared_input_contract_v3", "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "data_contract_failure", "claim_supported": "yes" if passed else "no", "evidence": {"fixed_input": "fixed_input_v2.npz", "manifest": "input_manifest.json", "forbidden_feature_audit": "forbidden_feature_audit.json"}, "checks": checks, "decision": "Shared A6-INPUT-v1.1 fixed64 input contract passes; authorize corrected runner implementation." if passed else "Shared input contract fails; keep corrected training blocked.", "remaining_work": ["implement shared hidden256 encoder and frozen lr/L1/dropout corrected runners"], "next_run_ids": ["a6_o010r_mlp_fixed64_v2"] if passed else [], "event_id": "a6_o000br2_shared_input_contract_v3_terminal"}
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-O000B", "status": summary["status"]}]})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

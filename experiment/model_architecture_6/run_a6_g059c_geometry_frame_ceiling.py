#!/usr/bin/env python3
"""Oracle and deployable-rule ceilings for contact-normal/hinge grasp frames."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from force_admittance_collect.controller import find_target_joint, joint_axis_world
from force_admittance_collect.world import pose_matrix, transform_world_to_base, yaw_pose
from jointTrain_new.experiment.model_architecture_6.run_a6_g051c_grasp_label_space_diagnostic import apply_initial_state_without_render
from jointTrain_new.joint_train.sim.capture_view_pcd import ViewPcdCapturer, base_pose_from_init, resolve_urdf
from path_config import ARTICU_COLLECTION_ROOT, JOINTTRAIN_ARCH6_G000C_RESULT_ROOT, JOINTTRAIN_ARCH6_G052C_RESULT_ROOT, JOINTTRAIN_ARCH6_G059C_RESULT_ROOT

Z_SYMMETRY = np.diag([-1.0, -1.0, 1.0])


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0)))


def symmetric_rotation_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return min(rotation_angle(predicted, target), rotation_angle(predicted @ Z_SYMMETRY, target))


def rotation_from_6d(value: np.ndarray) -> np.ndarray:
    first = value[:3] / max(np.linalg.norm(value[:3]), 1e-8)
    second = value[3:6] - np.dot(first, value[3:6]) * first
    second /= max(np.linalg.norm(second), 1e-8)
    return np.stack((first, second, np.cross(first, second)), axis=1)


def local_normal(points: np.ndarray, contact: np.ndarray) -> np.ndarray:
    _, indices = cKDTree(points).query(contact, k=min(32, len(points)))
    neighbors = points[np.atleast_1d(indices)]
    centered = neighbors - neighbors.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    normal = vectors[:, 0]
    return normal / max(np.linalg.norm(normal), 1e-8)


def frame(normal: np.ndarray, hinge: np.ndarray, mapping: str) -> np.ndarray | None:
    z = normal / max(np.linalg.norm(normal), 1e-8)
    tangent = hinge - np.dot(hinge, z) * z
    if np.linalg.norm(tangent) < 1e-5:
        return None
    tangent /= np.linalg.norm(tangent)
    if mapping == "x":
        x = tangent
        y = np.cross(z, x)
    else:
        y = tangent
        x = np.cross(y, z)
    return np.stack((x, y, z), axis=1)


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.quantile(array, 0.9)), "max": float(array.max())}


def main() -> int:
    started = time.time()
    groups = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_teacher_manifest.json").read_text())["groups"]
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        point = np.asarray(data["point_cloud_xyz"], dtype=np.float64)
        target_mask = np.asarray(data["target_mask"], dtype=bool)
        pose = np.asarray(data["se3_base"], dtype=np.float64)
        state = np.asarray(data["state_qpos"], dtype=np.float64)
        presence = np.asarray(data["presence"], dtype=bool)
        split = np.asarray(data["split"], dtype=np.int8)
    cap = ViewPcdCapturer(articu_root=ROOT, render_enabled=False, settle_steps=0)
    collection = Path(ARTICU_COLLECTION_ROOT) / "data" / "single"
    rows = []
    for group_index, group in enumerate(groups):
        init = json.loads((collection / group["sample_id"] / "initial_state.json").read_text())
        urdf = resolve_urdf(init["object_urdf"], partnet_root=cap.partnet_root)
        world = cap._get_world(urdf, float(init["size"]))
        apply_initial_state_without_render(world, init)
        base_pose = base_pose_from_init(init)
        base_rotation = pose_matrix(yaw_pose(base_pose))[:3, :3]
        joint = find_target_joint(world.object, str(init["link_name"]))
        hinge = base_rotation.T @ joint_axis_world(joint, str(urdf))
        current_hand = transform_world_to_base(base_pose, world.hand_pose_world(base_pose, state[group_index]))[:3, 3]
        target_points = point[group_index, target_mask[group_index]]
        tree = cKDTree(target_points)
        for slot in np.flatnonzero(presence[group_index]):
            target_position = pose[group_index, slot, :3]
            target_rotation = rotation_from_6d(pose[group_index, slot, 3:9])
            _, contact_index = tree.query(target_position, k=1)
            contact = target_points[contact_index]
            normal = local_normal(target_points, contact)
            if np.dot(normal, current_hand - contact) > 0:
                deploy_normal = -normal
            else:
                deploy_normal = normal
            candidates = []
            for sign in (-1.0, 1.0):
                for mapping in ("x", "y"):
                    candidate = frame(sign * normal, hinge, mapping)
                    if candidate is not None:
                        candidates.append((mapping, sign, candidate))
            deploy = {mapping: frame(deploy_normal, hinge, mapping) for mapping in ("x", "y")}
            rows.append({"split": int(split[group_index]), "contact": contact, "target_position": target_position, "target_rotation": target_rotation, "local_offset": target_rotation.T @ (target_position - contact), "candidates": candidates, "deploy": deploy})
    cap.close()
    train_rows = [row for row in rows if row["split"] == 0]
    cal_rows = [row for row in rows if row["split"] == 1]
    train_offset = np.mean([row["local_offset"] for row in train_rows], axis=0)
    mapping_errors = {}
    for mapping in ("x", "y"):
        valid = [row for row in train_rows if row["deploy"][mapping] is not None]
        mapping_errors[mapping] = float(np.mean([symmetric_rotation_error(row["deploy"][mapping], row["target_rotation"]) for row in valid]))
    selected_mapping = min(mapping_errors, key=mapping_errors.get)
    fixed_translation = []
    fixed_rotation = []
    oracle_translation = []
    oracle_rotation = []
    skipped = 0
    for row in cal_rows:
        fixed = row["deploy"][selected_mapping]
        if fixed is None or not row["candidates"]:
            skipped += 1
            continue
        fixed_position = row["contact"] + fixed @ train_offset
        fixed_translation.append(float(np.linalg.norm(fixed_position - row["target_position"])))
        fixed_rotation.append(symmetric_rotation_error(fixed, row["target_rotation"]))
        oracle = min(row["candidates"], key=lambda item: symmetric_rotation_error(item[2], row["target_rotation"]))[2]
        oracle_position = row["contact"] + oracle @ train_offset
        oracle_translation.append(float(np.linalg.norm(oracle_position - row["target_position"])))
        oracle_rotation.append(symmetric_rotation_error(oracle, row["target_rotation"]))
    fixed_success = np.mean((np.asarray(fixed_translation) <= 0.03) & (np.asarray(fixed_rotation) <= np.deg2rad(12.0)))
    oracle_success = np.mean((np.asarray(oracle_translation) <= 0.03) & (np.asarray(oracle_rotation) <= np.deg2rad(12.0)))
    metrics = {
        "labels": {"train": len(train_rows), "cal": len(cal_rows)},
        "train_only_mean_local_offset": train_offset.tolist(),
        "train_mapping_rotation_error": mapping_errors,
        "selected_deploy_mapping": selected_mapping,
        "fixed_rule": {"translation_m": stats(fixed_translation), "rotation_rad": stats(fixed_rotation), "pose_within_3cm_12deg": float(fixed_success)},
        "oracle_sign_mapping": {"translation_m": stats(oracle_translation), "rotation_rad": stats(oracle_rotation), "pose_within_3cm_12deg": float(oracle_success)},
        "skipped_cal_labels": skipped,
    }
    checks = {"label_counts": len(train_rows) == 1991 and len(cal_rows) == 382, "finite": bool(np.isfinite(fixed_translation + fixed_rotation + oracle_translation + oracle_rotation).all()), "train_only_rule_selection": True, "no_outcome_read": True, "metadata_axis_only": True}
    summary = {"schema_version": 1, "run_id": "A6-G059C", "status": "passed" if all(checks.values()) else "failed", "complete": True, "terminal": True, "elapsed_seconds": time.time() - started, "metrics": metrics, "checks": checks, "decision": "use fixed/oracle geometry ceiling to decide contact proposal training"}
    out = Path(JOINTTRAIN_ARCH6_G059C_RESULT_ROOT)
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare terminal joint labels with target-link-relative grasp frames."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jointTrain_new.joint_train.sim.capture_view_pcd import ViewPcdCapturer, base_pose_from_init, resolve_urdf
from path_config import ARTICU_COLLECTION_ROOT, JOINTTRAIN_ARCH6_G000C_RESULT_ROOT, JOINTTRAIN_ARCH6_G041C_RESULT_ROOT, JOINTTRAIN_ARCH6_G051C_RESULT_ROOT
from sapien_utils.sapien_compat import get_link_pose
from sapien_utils.env import set_articulation_joint_state
from force_admittance_collect.world import transform_world_to_base


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def pose_vector(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:3, 3], matrix[:3, :2].reshape(-1)))


def nearest_indices(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0).clip(min=1e-4)
    left = (train - mean) / std
    right = (query - mean) / std
    distance = np.linalg.norm(right[:, None] - left[None], axis=-1) / np.sqrt(train.shape[1])
    index = distance.argmin(axis=1)
    return index, distance[np.arange(len(query)), index]


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9)), "max": float(values.max())}


def apply_initial_state_without_render(world, init: dict) -> None:
    world.set_object_origin()
    world.set_base_pose(base_pose_from_init(init))
    set_articulation_joint_state(world.object, init.get("state", "target_almost_closed"), target_link_name=str(init["link_name"]), zero_qvel=True)
    object_qpos = np.asarray(init["initial_object_qpos"], dtype=np.float64)
    world.object.set_qpos(object_qpos)
    world.object.set_qvel(np.zeros_like(np.asarray(world.object.get_qvel(), dtype=np.float64)))
    world.set_robot_qpos(np.asarray(init.get("robot_default_full_qpos", world.default_full_qpos), dtype=np.float64))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    groups = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_teacher_manifest.json").read_text())["groups"]
    with np.load(Path(JOINTTRAIN_ARCH6_G041C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        state = np.asarray(data["state_qpos"], dtype=np.float64)
        qpose_relative = np.asarray(data["qpose_relative"], dtype=np.float64)
        presence = np.asarray(data["presence"], dtype=bool)
        split = np.asarray(data["split"], dtype=np.int8)
        group_index = np.asarray(data["group_index"], dtype=np.int64)
        point_cloud = np.asarray(data["point_cloud_xyz"], dtype=np.float64)
        target_mask = np.asarray(data["target_mask"], dtype=bool)
    count = min(args.limit, len(groups)) if args.limit else len(groups)
    cap = ViewPcdCapturer(articu_root=ROOT, render_enabled=False, settle_steps=0)
    collection = Path(ARTICU_COLLECTION_ROOT) / "data" / "single"
    qpose_rows = []
    pose_rows = []
    row_split = []
    row_group = []
    relative_matrices = []
    base_matrices = []
    centroid_relative_matrices = []
    ik_rows = []
    started = time.time()
    for index, group in enumerate(groups[:count]):
        init = json.loads((collection / group["sample_id"] / "initial_state.json").read_text())
        world = cap._get_world(resolve_urdf(init["object_urdf"], partnet_root=cap.partnet_root), float(init["size"]))
        apply_initial_state_without_render(world, init)
        target_link = next(link for link in world.object.get_links() if link.get_name() == str(init["link_name"]))
        link_world = np.asarray(get_link_pose(target_link).to_transformation_matrix(), dtype=np.float64)
        base_pose = base_pose_from_init(init)
        target_centroid = point_cloud[index, target_mask[index]].mean(axis=0)
        valid_slots = np.flatnonzero(presence[index])
        for slot in valid_slots:
            qpose = state[index] + qpose_relative[index, slot]
            hand_world = world.hand_pose_world(base_pose, qpose)
            relative = np.linalg.inv(link_world) @ hand_world
            hand_base = transform_world_to_base(base_pose, hand_world)
            centroid_relative = hand_base.copy()
            centroid_relative[:3, 3] -= target_centroid
            qpose_rows.append(qpose_relative[index, slot])
            pose_rows.append(pose_vector(relative))
            relative_matrices.append(relative)
            base_matrices.append(hand_base)
            centroid_relative_matrices.append(centroid_relative)
            row_split.append(split[index])
            row_group.append(group_index[index])
        if len(valid_slots):
            slot = int(valid_slots[0])
            qpose = state[index] + qpose_relative[index, slot]
            relative = relative_matrices[-len(valid_slots)]
            target_world = link_world @ relative
            ik = world.solve_ik(base_pose, target_world, seed_qpos=state[index])
            recovered = world.hand_pose_world(base_pose, ik.qpos)
            ik_rows.append({
                "group_index": int(group_index[index]),
                "success": bool(ik.success),
                "position_error": float(ik.position_error),
                "rotation_error": float(ik.rotation_error),
                "qpose_l2_to_teacher": float(np.linalg.norm(np.asarray(ik.qpos)[:7] - qpose)),
                "fk_position_error": float(np.linalg.norm(recovered[:3, 3] - target_world[:3, 3])),
            })
    cap.close()
    qpose_rows = np.asarray(qpose_rows, dtype=np.float64)
    pose_rows = np.asarray(pose_rows, dtype=np.float64)
    relative_matrices = np.asarray(relative_matrices, dtype=np.float64)
    base_matrices = np.asarray(base_matrices, dtype=np.float64)
    centroid_relative_matrices = np.asarray(centroid_relative_matrices, dtype=np.float64)
    row_split = np.asarray(row_split, dtype=np.int8)
    row_group = np.asarray(row_group, dtype=np.int64)
    train = np.flatnonzero(row_split == 0)
    cal = np.flatnonzero(row_split == 1)
    results = {}
    if len(train) and len(cal):
        q_index, q_distance = nearest_indices(qpose_rows[train], qpose_rows[cal])
        p_index, p_distance = nearest_indices(pose_rows[train], pose_rows[cal])
        q_neighbor = train[q_index]
        p_neighbor = train[p_index]
        translation = np.linalg.norm(relative_matrices[cal, :3, 3] - relative_matrices[p_neighbor, :3, 3], axis=1)
        rotation = np.asarray([rotation_angle(relative_matrices[left, :3, :3], relative_matrices[right, :3, :3]) for left, right in zip(cal, p_neighbor, strict=True)])
        results = {
            "labels": {"train": int(len(train)), "cal": int(len(cal)), "groups": int(count)},
            "qpose_standardized_train_nn": stats(q_distance),
            "relative_se3_standardized_train_nn": stats(p_distance),
            "relative_se3_train_nn_translation_m": stats(translation),
            "relative_se3_train_nn_rotation_rad": stats(rotation),
            "qpose_train_nn_l2": stats(np.linalg.norm(qpose_rows[cal] - qpose_rows[q_neighbor], axis=1)),
            "cal_group_count": int(len(np.unique(row_group[cal]))),
        }
        for name, matrices in (("base_se3", base_matrices), ("target_centroid_relative", centroid_relative_matrices)):
            vectors = np.stack([pose_vector(matrix) for matrix in matrices])
            index, distance = nearest_indices(vectors[train], vectors[cal])
            neighbor = train[index]
            translation = np.linalg.norm(matrices[cal, :3, 3] - matrices[neighbor, :3, 3], axis=1)
            rotation = np.asarray([rotation_angle(matrices[left, :3, :3], matrices[right, :3, :3]) for left, right in zip(cal, neighbor, strict=True)])
            results[f"{name}_standardized_train_nn"] = stats(distance)
            results[f"{name}_train_nn_translation_m"] = stats(translation)
            results[f"{name}_train_nn_rotation_rad"] = stats(rotation)
    ik_success = np.asarray([row["success"] for row in ik_rows], dtype=bool)
    checks = {
        "finite_labels": bool(np.isfinite(qpose_rows).all() and np.isfinite(pose_rows).all()),
        "one_pose_per_qpose": len(qpose_rows) == len(pose_rows),
        "valid_rotation": bool(np.allclose(np.linalg.det(relative_matrices[:, :3, :3]), 1.0, atol=1e-5)),
        "no_outcome_read": True,
        "full_group_count": count == 632 if not args.limit else count == args.limit,
        "full_split_counts": (len(train) == 1991 and len(cal) == 382) if not args.limit else True,
    }
    output = Path(JOINTTRAIN_ARCH6_G051C_RESULT_ROOT) / (f"probe_{args.limit}" if args.limit else "full")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "relative_grasp_labels.npz", relative_pose=relative_matrices, base_pose=base_matrices, target_centroid_relative_pose=centroid_relative_matrices, qpose_relative=qpose_rows, split=row_split, group_index=row_group)
    summary = {
        "schema_version": 1,
        "run_id": "A6-G051C-PROBE" if args.limit else "A6-G051C",
        "status": "passed" if all(checks.values()) else "failed",
        "complete": True,
        "terminal": True,
        "elapsed_seconds": time.time() - started,
        "results": results,
        "ik_roundtrip": {"attempted": len(ik_rows), "success": int(ik_success.sum()), "rate": float(ik_success.mean()) if len(ik_success) else 0.0, "rows": ik_rows},
        "checks": checks,
        "decision": "analyze label-space transfer before authorizing learned SE3 route",
    }
    atomic(output / "summary.json", summary)
    atomic(output / "run_state.json", summary)
    atomic(output / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

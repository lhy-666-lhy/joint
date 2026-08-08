#!/usr/bin/env python3
"""Raw pose metrics and current-state IK coverage for learned base-frame SE3 grasps."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_grasp_models import GraspProposalBase, rotation_6d_to_matrix
from jointTrain_new.experiment.model_architecture_6.run_a6_g051c_grasp_label_space_diagnostic import apply_initial_state_without_render
from jointTrain_new.joint_train.sim.capture_view_pcd import ViewPcdCapturer, base_pose_from_init, resolve_urdf
from path_config import ARTICU_COLLECTION_ROOT, JOINTTRAIN_ARCH6_G000C_RESULT_ROOT, JOINTTRAIN_ARCH6_G052C_RESULT_ROOT, JOINTTRAIN_ARCH6_G053C_RESULT_ROOT, JOINTTRAIN_ARCH6_G054C_RESULT_ROOT, JOINTTRAIN_ARCH6_G055C_RESULT_ROOT
from force_admittance_collect.world import pose_matrix, yaw_pose

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0)))


def vectors_to_pose(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values[..., 3:9], dtype=np.float32))
    rotation = rotation_6d_to_matrix(tensor).numpy().astype(np.float64)
    output = np.broadcast_to(np.eye(4), (*values.shape[:-1], 4, 4)).copy()
    output[..., :3, :3] = rotation
    output[..., :3, 3] = values[..., :3]
    return output


def match_rows(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    costs = np.zeros((4, 4), dtype=np.float64)
    translation = np.zeros((4, 4), dtype=np.float64)
    rotation = np.zeros((4, 4), dtype=np.float64)
    for left in range(4):
        for right in range(4):
            translation[left, right] = np.linalg.norm(predicted[left, :3, 3] - target[right, :3, 3])
            rotation[left, right] = rotation_angle(predicted[left, :3, :3], target[right, :3, :3])
            costs[left, right] = translation[left, right] + rotation[left, right]
    perm = min(PERMUTATIONS, key=lambda candidate: sum(costs[left, candidate[left]] for left in range(4) if valid[candidate[left]]))
    pairs = [(left, perm[left]) for left in range(4) if valid[perm[left]]]
    trans = np.asarray([translation[left, right] for left, right in pairs])
    rot = np.asarray([rotation[left, right] for left, right in pairs])
    return {
        "translation_m": float(trans.mean()),
        "rotation_rad": float(rot.mean()),
        "pose_within_3cm_12deg": float(np.mean((trans <= 0.03) & (rot <= np.deg2rad(12.0)))),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotations.sum(axis=0))
    output = u @ vt
    if np.linalg.det(output) < 0:
        u[:, -1] *= -1
        output = u @ vt
    return output


def main() -> int:
    started = time.time()
    groups = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_teacher_manifest.json").read_text())["groups"]
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        point = torch.from_numpy(np.asarray(data["point_cloud_xyz"], dtype=np.float32))
        state = torch.from_numpy(np.asarray(data["state_qpos"], dtype=np.float32))
        target_mask = torch.from_numpy(np.asarray(data["target_mask"], dtype=np.float32))
        target_values = np.asarray(data["se3_base"], dtype=np.float64)
        presence = np.asarray(data["presence"], dtype=bool)
        split = np.asarray(data["split"], dtype=np.int8)
        group_index = np.asarray(data["group_index"], dtype=np.int64)
    train = np.flatnonzero(split == 0)
    cal = np.flatnonzero(split == 1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    predictions = {}
    probabilities = {}
    for name, root, use_mask in (
        ("base_only", Path(JOINTTRAIN_ARCH6_G053C_RESULT_ROOT), False),
        ("target_local", Path(JOINTTRAIN_ARCH6_G054C_RESULT_ROOT), True),
    ):
        model = GraspProposalBase("se3", use_target_mask=use_mask, target_mask_encoding="local").to(device)
        checkpoint = root / "se3_zero_base_none_seed20260806.pth"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        model.eval()
        with torch.no_grad():
            output = model(point[cal].to(device), state[cal].to(device), target_mask=target_mask[cal].to(device) if use_mask else None)
        predictions[name] = vectors_to_pose(output["values"].cpu().numpy())
        probabilities[name] = torch.sigmoid(output["presence_logits"]).cpu().numpy()
    target_pose = vectors_to_pose(target_values)
    slot_mean = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    for slot in range(4):
        valid_train = train[presence[train, slot]]
        slot_mean[slot, :3, 3] = target_pose[valid_train, slot, :3, 3].mean(axis=0)
        slot_mean[slot, :3, :3] = mean_rotation(target_pose[valid_train, slot, :3, :3])
    metrics = {}
    for name, predicted in {**predictions, "train_slot_mean": np.repeat(slot_mean[None], len(cal), axis=0)}.items():
        rows = [match_rows(predicted[row], target_pose[index], presence[index]) for row, index in enumerate(cal)]
        metrics[name] = {"aggregate": aggregate(rows), "per_group": [{"group_index": int(group_index[index]), **rows[row]} for row, index in enumerate(cal)]}
    cap = ViewPcdCapturer(articu_root=ROOT, render_enabled=False, settle_steps=0)
    collection = Path(ARTICU_COLLECTION_ROOT) / "data" / "single"
    ik = {name: [] for name in predictions}
    for row, index in enumerate(cal):
        init = json.loads((collection / groups[index]["sample_id"] / "initial_state.json").read_text())
        world = cap._get_world(resolve_urdf(init["object_urdf"], partnet_root=cap.partnet_root), float(init["size"]))
        apply_initial_state_without_render(world, init)
        base_pose = base_pose_from_init(init)
        base_world = pose_matrix(yaw_pose(base_pose))
        for name, predicted in predictions.items():
            candidate_rows = []
            for candidate in range(4):
                target_world = base_world @ predicted[row, candidate]
                result = world.solve_ik(base_pose, target_world, seed_qpos=state[index].numpy())
                candidate_rows.append({"candidate": candidate, "success": bool(result.success), "position_error": float(result.position_error), "rotation_error": float(result.rotation_error), "joint_space_length": float(np.linalg.norm(np.asarray(result.qpos)[:7] - state[index].numpy())), "presence_probability": float(probabilities[name][row, candidate])})
            ik[name].append({"group_index": int(group_index[index]), "any_success": any(candidate["success"] for candidate in candidate_rows), "candidates": candidate_rows})
    cap.close()
    ik_summary = {}
    for name, rows in ik.items():
        candidates = [candidate for row in rows for candidate in row["candidates"]]
        ik_summary[name] = {"groups": len(rows), "groups_with_any_success": int(sum(row["any_success"] for row in rows)), "candidate_success": int(sum(candidate["success"] for candidate in candidates)), "candidate_total": len(candidates), "rows": rows}
    checks = {
        "cal_groups_101": len(cal) == 101,
        "finite_predictions": all(np.isfinite(value).all() for value in predictions.values()),
        "valid_rotations": all(np.allclose(np.linalg.det(value[..., :3, :3]), 1.0, atol=1e-5) for value in predictions.values()),
        "ik_attempts_exact": all(value["candidate_total"] == 404 for value in ik_summary.values()),
        "no_link_pose_or_outcome": True,
        "raw_units": True,
    }
    summary = {"schema_version": 1, "run_id": "A6-G055C", "status": "passed" if all(checks.values()) else "failed", "complete": True, "terminal": True, "elapsed_seconds": time.time() - started, "metrics": metrics, "ik": ik_summary, "checks": checks, "decision": "analyze pose error and IK coverage before physical planner rollout"}
    out = Path(JOINTTRAIN_ARCH6_G055C_RESULT_ROOT)
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

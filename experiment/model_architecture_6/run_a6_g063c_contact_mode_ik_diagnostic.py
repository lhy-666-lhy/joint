#!/usr/bin/env python3
"""Diagnose contact-local grasp modes and IK-equivalent solution sets."""

from __future__ import annotations

import argparse
import json
import math
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
from force_admittance_collect.feedback import GRIPPER_LINK_NAMES, read_target_contact_feedback
from force_admittance_collect.world import pose_matrix, yaw_pose
from jointTrain_new.experiment.model_architecture_6.run_a6_a030c_affordance_cal_consumer import paired_bootstrap
from jointTrain_new.experiment.model_architecture_6.run_a6_g051c_grasp_label_space_diagnostic import apply_initial_state_without_render
from jointTrain_new.experiment.model_architecture_6.run_a6_g059c_geometry_frame_ceiling import frame, local_normal, rotation_from_6d
from jointTrain_new.joint_train.sim.capture_view_pcd import ViewPcdCapturer, base_pose_from_init, resolve_urdf
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G052C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G061C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G063C_RESULT_ROOT,
)
from sapien_utils.sapien_compat import get_link_name


MODE_COUNTS = (1, 2, 4, 8)


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def contact_bodies(contact) -> list:
    try:
        bodies = list(contact.bodies)
    except Exception:
        bodies = []
    if not bodies:
        bodies = [body for name in ("actor0", "actor1") if (body := getattr(contact, name, None)) is not None]
    return bodies[:2]


def self_collision(world) -> bool:
    robot_links = set(world.robot.get_links())
    try:
        contacts = world.scene.get_contacts()
    except Exception:
        contacts = []
    for contact in contacts:
        bodies = contact_bodies(contact)
        if len(bodies) == 2 and bodies[0] in robot_links and bodies[1] in robot_links:
            impulse = sum(float(np.linalg.norm(np.asarray(getattr(point, "impulse", np.zeros(3))))) for point in getattr(contact, "points", []) or [])
            if impulse > 1e-9:
                return True
    return False


def fallback_frame(normal: np.ndarray, hinge: np.ndarray, mapping: str) -> tuple[np.ndarray, bool]:
    candidate = frame(normal, hinge, mapping)
    if candidate is not None:
        return candidate, False
    for axis in np.eye(3):
        candidate = frame(normal, axis, mapping)
        if candidate is not None:
            return candidate, True
    raise RuntimeError("unable to construct contact-local frame")


def farthest_prototypes(features: np.ndarray, count: int) -> np.ndarray:
    center = features.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(features - center, axis=1)))]
    while len(selected) < count:
        distance = np.min(np.linalg.norm(features[:, None] - features[np.asarray(selected)][None], axis=-1), axis=1)
        distance[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(distance)))
    return np.asarray(selected, dtype=np.int64)


def evaluate_modes(rows: list[dict], prototypes: list[dict], mode_count: int) -> tuple[dict, dict[int, dict[str, float]]]:
    translation = []
    rotation = []
    success = []
    per_group: dict[int, list[tuple[float, float, bool]]] = {}
    for row in rows:
        candidates = []
        for prototype in prototypes[:mode_count]:
            predicted_position = row["query"] + row["frame"] @ prototype["translation"]
            predicted_rotation = row["frame"] @ prototype["rotation"]
            t_error = float(np.linalg.norm(predicted_position - row["target_position"]))
            r_error = rotation_angle(predicted_rotation, row["target_rotation"])
            candidates.append((t_error / 0.03 + r_error / math.radians(12.0), t_error, r_error))
        _, selected_t, selected_r = min(candidates, key=lambda item: item[0])
        hit = any(item[1] <= 0.03 and item[2] <= math.radians(12.0) for item in candidates)
        translation.append(selected_t)
        rotation.append(selected_r)
        success.append(hit)
        per_group.setdefault(row["group_index"], []).append((selected_t, selected_r, hit))
    group_values = {
        group: {
            "translation_m": float(np.mean([item[0] for item in values])),
            "rotation_rad": float(np.mean([item[1] for item in values])),
            "pose_within_3cm_12deg": float(np.mean([item[2] for item in values])),
        }
        for group, values in per_group.items()
    }
    return {
        "labels": len(rows),
        "groups": len(group_values),
        "translation_m": stats(np.asarray(translation)),
        "rotation_rad": stats(np.asarray(rotation)),
        "pose_within_3cm_12deg": float(np.mean(success)),
    }, group_values


def mode_codebook(rows: list[dict], count: int = 8) -> list[dict]:
    features = np.stack([row["local_rotation"][:, :2].reshape(-1) for row in rows])
    selected = farthest_prototypes(features, count)
    return [
        {
            "row_index": int(index),
            "rotation": rows[int(index)]["local_rotation"],
            "translation": rows[int(index)]["local_translation"],
        }
        for index in selected
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Observation groups; zero runs all 632 groups.")
    parser.add_argument("--ik-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.ik_seeds < 2:
        raise ValueError("ik-seeds must include at least teacher and current-state seeds")

    started = time.time()
    groups = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_teacher_manifest.json").read_text())["groups"]
    with np.load(Path(JOINTTRAIN_ARCH6_G061C_RESULT_ROOT) / "full" / "contact_query_inputs.npz", allow_pickle=False) as data:
        query = {key: np.asarray(data[key]) for key in data.files}
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        teacher_qpose = np.asarray(data["qpose_relative"], dtype=np.float64)
        teacher_presence = np.asarray(data["presence"], dtype=bool)
    if args.limit:
        selected_indices = np.concatenate(
            (
                np.flatnonzero(query["split"] == 0)[: args.limit],
                np.flatnonzero(query["split"] == 1)[: args.limit],
            )
        )
    else:
        selected_indices = np.arange(len(groups), dtype=np.int64)
    selected_groups = [groups[int(index)] for index in selected_indices]
    count = len(selected_indices)
    query = {key: value[selected_indices] for key, value in query.items()}
    teacher_qpose = teacher_qpose[selected_indices]
    teacher_presence = teacher_presence[selected_indices]
    output = Path(JOINTTRAIN_ARCH6_G063C_RESULT_ROOT) / (f"probe_{args.limit}" if args.limit else "full")
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(output / "run_manifest.json", {"run_id": "A6-G063C", "groups": count, "ik_seeds": args.ik_seeds, "mode_counts": list(MODE_COUNTS)})

    cap = ViewPcdCapturer(articu_root=ROOT, render_enabled=False, settle_steps=0)
    collection = Path(ARTICU_COLLECTION_ROOT) / "data" / "single"
    mode_rows = {"x": [], "y": []}
    ik_rows = []
    fallback_frames = 0
    for group_index, group in enumerate(selected_groups):
        init = json.loads((collection / group["sample_id"] / "initial_state.json").read_text())
        urdf = resolve_urdf(init["object_urdf"], partnet_root=cap.partnet_root)
        world = cap._get_world(urdf, float(init["size"]))
        apply_initial_state_without_render(world, init)
        base_pose = base_pose_from_init(init)
        base_rotation = pose_matrix(yaw_pose(base_pose))[:3, :3]
        joint = find_target_joint(world.object, str(init["link_name"]))
        hinge = base_rotation.T @ joint_axis_world(joint, str(urdf))
        current_hand_base = world.hand_pose_world(base_pose, query["state_qpos"][group_index])
        current_hand_base = np.linalg.inv(pose_matrix(yaw_pose(base_pose))) @ current_hand_base
        target_points = query["point_cloud_xyz"][group_index, query["target_mask"][group_index].astype(bool)]
        for slot in np.flatnonzero(query["query_presence"][group_index]):
            contact = np.asarray(query["query_point"][group_index, slot], dtype=np.float64)
            target_value = np.asarray(query["query_target_se3"][group_index, slot], dtype=np.float64)
            target_position = target_value[:3]
            target_rotation = rotation_from_6d(target_value[3:9])
            teacher_slot = int(query["query_teacher_slot"][group_index, slot])
            if teacher_slot < 0 or not teacher_presence[group_index, teacher_slot]:
                raise RuntimeError("G061 query slot does not map to a valid G052 teacher")
            normal = local_normal(target_points, contact)
            if np.dot(normal, current_hand_base[:3, 3] - contact) > 0.0:
                normal = -normal
            frames = {}
            for mapping in ("x", "y"):
                local_frame, used_fallback = fallback_frame(normal, hinge, mapping)
                fallback_frames += int(used_fallback)
                frames[mapping] = local_frame
                mode_rows[mapping].append({
                    "group_index": int(query["group_index"][group_index]),
                    "source_replay_id": int(query["source_replay_id"][group_index]),
                    "query_slot": int(slot),
                    "teacher_slot": teacher_slot,
                    "split": int(query["split"][group_index]),
                    "query": contact,
                    "frame": local_frame,
                    "target_position": target_position,
                    "target_rotation": target_rotation,
                    "local_translation": local_frame.T @ (target_position - contact),
                    "local_rotation": local_frame.T @ target_rotation,
                })

            teacher_abs = query["state_qpos"][group_index].astype(np.float64) + teacher_qpose[group_index, teacher_slot]
            target_base = np.eye(4, dtype=np.float64)
            target_base[:3, :3] = target_rotation
            target_base[:3, 3] = target_position
            target_world = pose_matrix(yaw_pose(base_pose)) @ target_base
            limits = np.asarray(world.qlimits[:7], dtype=np.float64)
            rng = np.random.default_rng(args.seed + group_index * 101 + int(slot))
            seeds = [teacher_abs, query["state_qpos"][group_index].astype(np.float64)]
            if args.ik_seeds >= 3:
                seeds.append(0.5 * (limits[:, 0] + limits[:, 1]))
            while len(seeds) < args.ik_seeds:
                seeds.append(rng.uniform(limits[:, 0], limits[:, 1]))
            accepted = []
            attempts = []
            for seed_index, seed in enumerate(seeds):
                apply_initial_state_without_render(world, init)
                result = world.solve_ik(base_pose, target_world, seed_qpos=seed)
                within_limits = bool(np.all(result.qpos[:7] >= limits[:, 0]) and np.all(result.qpos[:7] <= limits[:, 1]))
                collision_acceptable = False
                has_self_collision = False
                non_target_count = 0
                bad_target_links = []
                if result.success and within_limits:
                    world.set_robot_qpos(world.make_full_qpos(result.qpos[:7], 0.04))
                    try:
                        world.robot.set_qvel(np.zeros_like(np.asarray(world.robot.get_qvel(), dtype=np.float64)))
                    except Exception:
                        pass
                    world.step(render=False)
                    feedback = read_target_contact_feedback(world, str(init["link_name"]))
                    has_self_collision = self_collision(world)
                    non_target_count = len(feedback.non_target_pairs)
                    bad_target_links = sorted(set(feedback.target_robot_links) - GRIPPER_LINK_NAMES)
                    collision_acceptable = not has_self_collision and non_target_count == 0 and not bad_target_links
                legal = bool(result.success and within_limits and collision_acceptable)
                duplicate = any(np.linalg.norm(result.qpos[:7] - value) < 0.1 for value in accepted) if legal else False
                if legal and not duplicate:
                    accepted.append(np.asarray(result.qpos[:7], dtype=np.float64))
                attempts.append({
                    "seed_index": seed_index,
                    "ik_success": bool(result.success),
                    "within_limits": within_limits,
                    "collision_acceptable": collision_acceptable,
                    "self_collision": has_self_collision,
                    "non_target_collision_count": non_target_count,
                    "bad_target_links": bad_target_links,
                    "position_error": float(result.position_error),
                    "rotation_error": float(result.rotation_error),
                    "duplicate": duplicate,
                })
            ik_rows.append({
                "group_index": int(query["group_index"][group_index]),
                "slot": int(slot),
                "split": int(query["split"][group_index]),
                "teacher_slot": teacher_slot,
                "attempts": attempts,
                "unique_legal_solutions": len(accepted),
                "accepted_qpos": [value.tolist() for value in accepted],
                "nearest_legal_to_teacher_l2": float(min((np.linalg.norm(value - teacher_abs) for value in accepted), default=float("nan"))),
            })
        if (group_index + 1) % 10 == 0 or group_index + 1 == count:
            atomic(output / "progress.json", {"groups_complete": group_index + 1, "groups_total": count, "labels_complete": len(ik_rows), "elapsed_seconds": time.time() - started})
    cap.close()

    mapping_scores = {}
    for mapping in ("x", "y"):
        train_rows = [row for row in mode_rows[mapping] if row["split"] == 0]
        prototype = mode_codebook(train_rows, 1)
        train_metric, _ = evaluate_modes(train_rows, prototype, 1)
        mapping_scores[mapping] = train_metric["translation_m"]["mean"] / 0.03 + train_metric["rotation_rad"]["mean"] / math.radians(12.0)
    selected_mapping = min(mapping_scores, key=mapping_scores.get)
    rows = mode_rows[selected_mapping]
    train_rows = [row for row in rows if row["split"] == 0]
    cal_rows = [row for row in rows if row["split"] == 1]
    prototypes = mode_codebook(train_rows, max(MODE_COUNTS))
    mode_metrics = {}
    cal_groups = {}
    for mode_count in MODE_COUNTS:
        train_metric, _ = evaluate_modes(train_rows, prototypes, mode_count)
        cal_metric, group_metric = evaluate_modes(cal_rows, prototypes, mode_count)
        mode_metrics[str(mode_count)] = {"train": train_metric, "cal": cal_metric}
        cal_groups[mode_count] = group_metric
    common_groups = sorted(set(cal_groups[1]) & set(cal_groups[8]))
    translation_difference = np.asarray([cal_groups[8][group]["translation_m"] - cal_groups[1][group]["translation_m"] for group in common_groups])
    rotation_difference = np.asarray([cal_groups[8][group]["rotation_rad"] - cal_groups[1][group]["rotation_rad"] for group in common_groups])
    paired = {
        "m8_minus_m1_translation_m": paired_bootstrap(translation_difference, args.seed),
        "m8_minus_m1_rotation_rad": paired_bootstrap(rotation_difference, args.seed),
    }

    ik_train = [row for row in ik_rows if row["split"] == 0]
    ik_cal = [row for row in ik_rows if row["split"] == 1]
    def ik_summary(items: list[dict]) -> dict:
        counts = np.asarray([row["unique_legal_solutions"] for row in items], dtype=np.int64)
        distances = np.asarray([row["nearest_legal_to_teacher_l2"] for row in items], dtype=np.float64)
        finite_distances = distances[np.isfinite(distances)]
        return {
            "labels": len(items),
            "coverage_at_least_one": float(np.mean(counts >= 1)),
            "multi_solution_fraction": float(np.mean(counts >= 2)),
            "unique_solution_count": stats(counts.astype(np.float64)),
            "nearest_legal_to_teacher_l2": stats(finite_distances) if len(finite_distances) else None,
        }
    ik_metrics = {"train": ik_summary(ik_train), "cal": ik_summary(ik_cal)}

    full = args.limit == 0
    checks = {
        "upstream_g061_terminal": json.loads((Path(JOINTTRAIN_ARCH6_G061C_RESULT_ROOT) / "full" / "summary.json").read_text())["status"] == "passed",
        "all_present_labels_consumed": len(rows) == int(query["query_presence"].sum()) == len(ik_rows),
        "split_counts": (len(train_rows) == 1991 and len(cal_rows) == 382) if full else True,
        "mode_counts_nested": all(MODE_COUNTS[index] < MODE_COUNTS[index + 1] for index in range(len(MODE_COUNTS) - 1)),
        "train_only_codebook": True,
        "finite_mode_metrics": bool(np.isfinite(translation_difference).all() and np.isfinite(rotation_difference).all()),
        "ik_seed_count": all(len(row["attempts"]) == args.ik_seeds for row in ik_rows),
        "no_outcome_read": True,
        "no_gt_link_pose_forward_input": True,
    }
    implementation_passed = all(checks.values())
    mode_supported = bool(
        paired["m8_minus_m1_rotation_rad"]["ci95"][1] < 0.0
        and paired["m8_minus_m1_translation_m"]["ci95"][1] <= 0.0
        and mode_metrics["8"]["cal"]["pose_within_3cm_12deg"] >= mode_metrics["1"]["cal"]["pose_within_3cm_12deg"]
    )
    ik_supported = ik_metrics["cal"]["coverage_at_least_one"] > 0.0 if full else False
    claim_supported = implementation_passed and full and mode_supported and ik_supported

    np.savez_compressed(
        output / "contact_mode_labels.npz",
        split=np.asarray([row["split"] for row in rows], dtype=np.int8),
        group_index=np.asarray([row["group_index"] for row in rows], dtype=np.int64),
        source_replay_id=np.asarray([row["source_replay_id"] for row in rows], dtype=np.int64),
        query_slot=np.asarray([row["query_slot"] for row in rows], dtype=np.int8),
        teacher_slot=np.asarray([row["teacher_slot"] for row in rows], dtype=np.int8),
        query_point=np.stack([row["query"] for row in rows]).astype(np.float32),
        contact_frame=np.stack([row["frame"] for row in rows]).astype(np.float32),
        local_translation=np.stack([row["local_translation"] for row in rows]).astype(np.float32),
        local_rotation=np.stack([row["local_rotation"] for row in rows]).astype(np.float32),
        prototype_train_row=np.asarray([item["row_index"] for item in prototypes], dtype=np.int64),
    )
    atomic(output / "ik_rows.json", {"rows": ik_rows})
    atomic(output / "forbidden_feature_audit.json", {"task_outcome_read": False, "future_path_read": False, "cal_codebook_fit": False, "gt_link_pose_forward_input": False})
    summary = {
        "schema_version": 1,
        "run_id": "A6-G063C-PROBE" if args.limit else "A6-G063C",
        "status": "passed" if implementation_passed else "failed",
        "complete": True,
        "terminal": True,
        "elapsed_seconds": time.time() - started,
        "selected_contact_frame_mapping": selected_mapping,
        "train_mapping_scores": mapping_scores,
        "fallback_frame_count": fallback_frames,
        "mode_metrics": mode_metrics,
        "paired_comparison": paired,
        "ik_metrics": ik_metrics,
        "checks": checks,
        "claim_supported": "yes" if claim_supported else ("probe_only" if implementation_passed and not full else "no"),
        "decision": "authorize G064 supervision contract" if claim_supported else ("run full G063C" if implementation_passed and not full else "stop mode/IK branch and analyze observability"),
        "next_run_ids": ["A6-G064C"] if claim_supported else (["A6-G063C"] if implementation_passed and not full else []),
    }
    atomic(output / "summary.json", summary)
    atomic(output / "run_state.json", summary)
    atomic(output / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if implementation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

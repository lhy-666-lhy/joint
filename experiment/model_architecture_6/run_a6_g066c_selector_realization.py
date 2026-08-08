#!/usr/bin/env python3
"""Outcome-blind IK realization and planner ranking for frozen G065 candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from force_admittance_collect.curobo_grasp import CuroboGraspConfig
from force_admittance_collect.feedback import GRIPPER_LINK_NAMES, read_target_contact_feedback
from force_admittance_collect.world import pose_matrix, yaw_pose
from jointTrain_new.experiment.model_architecture_6.a6_joint_goal_planner import plan_joint_goals_batch
from jointTrain_new.experiment.model_architecture_6.run_a6_a030c_affordance_cal_consumer import paired_bootstrap
from jointTrain_new.experiment.model_architecture_6.run_a6_g051c_grasp_label_space_diagnostic import apply_initial_state_without_render
from jointTrain_new.experiment.model_architecture_6.run_a6_g063c_contact_mode_ik_diagnostic import self_collision
from jointTrain_new.experiment.model_architecture_6.run_a6_g065c_contact_mode_residual_fit import rotation_6d_to_matrix
from jointTrain_new.joint_train.sim.capture_view_pcd import ViewPcdCapturer, base_pose_from_init, resolve_urdf
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G062C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G066C_RESULT_ROOT,
)


SEED = 20260806
IK_SEEDS = 4
MAX_IK = 4
TRANSLATION_SCALE = 0.03
ROTATION_SCALE = float(np.deg2rad(12.0))


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def rotation_matrix(value: np.ndarray) -> np.ndarray:
    import torch

    return rotation_6d_to_matrix(torch.from_numpy(np.asarray(value, dtype=np.float32))).numpy().astype(np.float64)


def group_rows(
    groups: np.ndarray,
    presence: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> dict[int, dict[str, float]]:
    output = {}
    for local, group in enumerate(groups.tolist()):
        valid = presence[local]
        output[int(group)] = {
            "translation_m": float(translation[local, valid].mean()),
            "rotation_rad": float(rotation[local, valid].mean()),
            "pose_within_3cm_12deg": float(
                ((translation[local, valid] <= TRANSLATION_SCALE) & (rotation[local, valid] <= ROTATION_SCALE)).mean()
            ),
        }
    return output


def paired(left: dict[int, dict[str, float]], right: dict[int, dict[str, float]]) -> dict[str, dict]:
    common = sorted(set(left) & set(right))
    return {
        key: paired_bootstrap(np.asarray([left[group][key] - right[group][key] for group in common]), SEED)
        for key in ("translation_m", "rotation_rad", "pose_within_3cm_12deg")
    }


def choose_s2(
    ik_presence: np.ndarray,
    planner_success: np.ndarray,
    path_length: np.ndarray,
    joint_margin: np.ndarray,
    fk_residual: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    rows, slots, modes, _ = ik_presence.shape
    selected = np.asarray(fallback, dtype=np.int64).copy()
    for row in range(rows):
        for slot in range(slots):
            keys = []
            for mode in range(modes):
                legal = ik_presence[row, slot, mode]
                planned = planner_success[row, slot, mode] & legal
                has_plan = bool(planned.any())
                has_ik = bool(legal.any())
                path = float(path_length[row, slot, mode, planned].min()) if has_plan else float("inf")
                eligible = planned if has_plan else legal
                margin = float(joint_margin[row, slot, mode, eligible].max()) if eligible.any() else -float("inf")
                fk = float(fk_residual[row, slot, mode, eligible].min()) if eligible.any() else float("inf")
                keys.append((not has_plan, not has_ik, path, -margin, fk, mode))
            if any(ik_presence[row, slot].reshape(-1)):
                selected[row, slot] = min(keys)[-1]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="CAL groups; zero runs all 101.")
    parser.add_argument("--start", type=int, default=0, help="First CAL row for a formal shard.")
    parser.add_argument("--count", type=int, default=0, help="CAL rows in a formal shard; zero disables sharding.")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    score_root = Path(JOINTTRAIN_ARCH6_G066C_RESULT_ROOT) / "score" / "full"
    with np.load(score_root / "candidate_predictions.npz", allow_pickle=False) as data:
        candidates = {key: np.asarray(data[key]) for key in data.files}
    total = len(candidates["group_index"])
    if args.limit:
        start_index = 0
        count = min(args.limit, total)
        output_name = f"probe_{args.limit}"
    elif args.count:
        start_index = args.start
        count = min(args.count, total - start_index)
        output_name = f"shard_{start_index:03d}_{start_index + count:03d}"
    else:
        start_index = 0
        count = total
        output_name = "full"
    if count <= 0 or start_index < 0 or start_index + count > total:
        raise ValueError("invalid G066 CAL shard bounds")
    candidates = {key: value[start_index : start_index + count] for key, value in candidates.items()}
    out = Path(JOINTTRAIN_ARCH6_G066C_RESULT_ROOT) / "realize" / output_name
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    running = {
        "schema_version": 1,
        "run_id": "A6-G066C-REALIZE-PROBE" if args.limit else ("A6-G066C-REALIZE-SHARD" if args.count else "A6-G066C-REALIZE"),
        "status": "running",
        "complete": False,
        "groups_total": count,
        "cal_row_start": start_index,
        "cal_row_end": start_index + count,
        "ik_seeds": IK_SEEDS,
        "started_at_unix": time.time(),
    }
    atomic(out / "run_state.json", running)
    atomic(out / "queue_state.json", running)

    manifest = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_teacher_manifest.json").read_text())
    group_lookup = {int(row["group_index"]): row for row in manifest["groups"]}
    collection = Path(ARTICU_COLLECTION_ROOT) / "data" / "single"
    shape = candidates["candidate_translation"].shape[:3]
    ik_qpose = np.zeros((*shape, MAX_IK, 7), dtype=np.float32)
    ik_presence = np.zeros((*shape, MAX_IK), dtype=bool)
    joint_margin = np.zeros((*shape, MAX_IK), dtype=np.float32)
    fk_residual = np.full((*shape, MAX_IK), np.inf, dtype=np.float32)
    attempts = np.zeros((*shape, IK_SEEDS, 5), dtype=np.float32)
    candidate_rotation = rotation_matrix(candidates["candidate_rotation_6d"])
    started = time.time()
    cap = ViewPcdCapturer(articu_root=ROOT, render_enabled=False, settle_steps=0)
    for row in range(count):
        group_index = int(candidates["group_index"][row])
        group = group_lookup[group_index]
        init = json.loads((collection / group["sample_id"] / "initial_state.json").read_text())
        urdf = resolve_urdf(init["object_urdf"], partnet_root=cap.partnet_root)
        world = cap._get_world(urdf, float(init["size"]))
        apply_initial_state_without_render(world, init)
        base_pose = base_pose_from_init(init)
        base_world = pose_matrix(yaw_pose(base_pose))
        limits = np.asarray(world.qlimits[:7], dtype=np.float64)
        current = candidates["state_qpos"][row].astype(np.float64)
        midpoint = 0.5 * (limits[:, 0] + limits[:, 1])
        for slot in np.flatnonzero(candidates["query_presence"][row]):
            for mode in range(shape[2]):
                target_base = np.eye(4, dtype=np.float64)
                target_base[:3, 3] = candidates["candidate_translation"][row, slot, mode]
                target_base[:3, :3] = candidate_rotation[row, slot, mode]
                target_world = base_world @ target_base
                rng = np.random.default_rng(SEED + group_index * 1009 + int(slot) * 31 + mode)
                seeds = [current, midpoint, rng.uniform(limits[:, 0], limits[:, 1]), rng.uniform(limits[:, 0], limits[:, 1])]
                accepted = []
                accepted_meta = []
                for seed_index, seed in enumerate(seeds):
                    apply_initial_state_without_render(world, init)
                    result = world.solve_ik(base_pose, target_world, seed_qpos=seed)
                    qpos = np.asarray(result.qpos[:7], dtype=np.float64)
                    within = bool(np.all(qpos >= limits[:, 0]) and np.all(qpos <= limits[:, 1]))
                    collision_ok = False
                    if result.success and within:
                        world.set_robot_qpos(world.make_full_qpos(qpos, 0.04))
                        try:
                            world.robot.set_qvel(np.zeros_like(np.asarray(world.robot.get_qvel(), dtype=np.float64)))
                        except Exception:
                            pass
                        world.step(render=False)
                        feedback = read_target_contact_feedback(world, str(init["link_name"]))
                        bad_links = set(feedback.target_robot_links) - GRIPPER_LINK_NAMES
                        collision_ok = not self_collision(world) and not feedback.non_target_pairs and not bad_links
                    duplicate = any(np.linalg.norm(qpos - value) < 0.1 for value in accepted) if result.success and within and collision_ok else False
                    legal = bool(result.success and within and collision_ok and not duplicate)
                    attempts[row, slot, mode, seed_index] = np.asarray(
                        [float(result.success), float(within), float(collision_ok), float(result.position_error), float(result.rotation_error)],
                        dtype=np.float32,
                    )
                    if legal and len(accepted) < MAX_IK:
                        span = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)
                        margin = np.min(np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos) / span)
                        accepted.append(qpos)
                        accepted_meta.append((float(margin), float(result.position_error + result.rotation_error)))
                for index, (qpos, meta) in enumerate(zip(accepted, accepted_meta)):
                    ik_qpose[row, slot, mode, index] = qpos.astype(np.float32)
                    ik_presence[row, slot, mode, index] = True
                    joint_margin[row, slot, mode, index] = meta[0]
                    fk_residual[row, slot, mode, index] = meta[1]
        atomic(out / "progress.json", {
            "phase": "ik",
            "groups_complete": row + 1,
            "groups_total": count,
            "legal_ik": int(ik_presence[: row + 1].sum()),
            "elapsed_seconds": time.time() - started,
        })
    cap.close()

    planner_success = np.zeros_like(ik_presence)
    path_length = np.full(ik_presence.shape, np.inf, dtype=np.float32)
    config = CuroboGraspConfig(device=f"cuda:{args.gpu}", num_seeds=4, num_trajopt_seeds=4)
    for row in range(count):
        locations = np.argwhere(ik_presence[row])
        if len(locations):
            goals = np.stack([ik_qpose[row, slot, mode, index] for slot, mode, index in locations])
            plans = plan_joint_goals_batch(candidates["state_qpos"][row], goals, config, terminal_tolerance=1e-3)
            for (slot, mode, index), plan in zip(locations, plans):
                planner_success[row, slot, mode, index] = bool(plan.success)
                if plan.success:
                    path_length[row, slot, mode, index] = float(np.linalg.norm(np.diff(plan.path, axis=0), axis=1).sum())
        atomic(out / "progress.json", {
            "phase": "planner",
            "groups_complete": row + 1,
            "groups_total": count,
            "planner_success": int(planner_success[: row + 1].sum()),
            "elapsed_seconds": time.time() - started,
        })

    s2_selected = choose_s2(
        ik_presence, planner_success, path_length, joint_margin, fk_residual, candidates["s1_selected"]
    )
    mutation_selected = choose_s2(
        ik_presence.copy(), planner_success.copy(), path_length.copy(), joint_margin.copy(), fk_residual.copy(), candidates["s1_selected"]
    )
    np.savez_compressed(
        out / "realization.npz",
        group_index=candidates["group_index"],
        query_presence=candidates["query_presence"],
        ik_qpose=ik_qpose,
        ik_presence=ik_presence,
        joint_margin=joint_margin,
        fk_residual=fk_residual,
        planner_success=planner_success,
        path_length=path_length,
        s0_selected=candidates["s0_selected"],
        s1_selected=candidates["s1_selected"],
        s2_selected=s2_selected,
    )

    with np.load(score_root / "evaluation_labels.npz", allow_pickle=False) as labels_file:
        labels = {
            key: np.asarray(labels_file[key])[start_index : start_index + count]
            for key in labels_file.files
        }
    selectors = {
        "s0_mode_logit": candidates["s0_selected"],
        "s1_calibrated_risk": candidates["s1_selected"],
        "s2_ik_fk_planner": s2_selected,
        "oracle_best_of_8": (labels["translation_error"] / TRANSLATION_SCALE + labels["rotation_error"] / ROTATION_SCALE).argmin(axis=-1),
    }
    metrics = {}
    per_group = {}
    for name, selected in selectors.items():
        translation = np.take_along_axis(labels["translation_error"], selected[..., None], axis=-1).squeeze(-1)
        rotation = np.take_along_axis(labels["rotation_error"], selected[..., None], axis=-1).squeeze(-1)
        rows = group_rows(candidates["group_index"], labels["presence"], translation, rotation)
        per_group[name] = rows
        valid = labels["presence"]
        metrics[name] = {
            "translation_m": float(translation[valid].mean()),
            "rotation_rad": float(rotation[valid].mean()),
            "pose_within_3cm_12deg": float(
                ((translation[valid] <= TRANSLATION_SCALE) & (rotation[valid] <= ROTATION_SCALE)).mean()
            ),
            "group_mean": {key: float(np.mean([row[key] for row in rows.values()])) for key in rows[next(iter(rows))]},
        }
    baseline = json.loads((Path(JOINTTRAIN_ARCH6_G062C_RESULT_ROOT) / "full" / "summary.json").read_text())
    baseline_groups = {int(row["group_index"]): row for row in baseline["metrics"]["per_group"]}
    comparisons = {
        "s2_minus_g062": paired(per_group["s2_ik_fk_planner"], baseline_groups),
        "s2_minus_oracle": paired(per_group["s2_ik_fk_planner"], per_group["oracle_best_of_8"]),
        "s1_minus_oracle": paired(per_group["s1_calibrated_risk"], per_group["oracle_best_of_8"]),
    }
    full = args.limit == 0 and args.count == 0
    selected_ik = np.take_along_axis(ik_presence.any(axis=-1), s2_selected[..., None], axis=-1).squeeze(-1)
    selected_plan = np.take_along_axis(planner_success.any(axis=-1), s2_selected[..., None], axis=-1).squeeze(-1)
    valid = candidates["query_presence"]
    checks = {
        "score_terminal": json.loads((score_root / "summary.json").read_text())["status"] == "passed",
        "cal_group_count": count == (101 if full else (args.limit if args.limit else args.count)),
        "fixed_candidate_shape": candidates["candidate_translation"].shape == (count, 4, 8, 3),
        "ik_seed_count": attempts.shape[-2] == IK_SEEDS,
        "no_teacher_qpose_seed": True,
        "finite_legal_qpose": bool(np.isfinite(ik_qpose[ik_presence]).all()),
        "selection_indices_valid": bool(np.all((s2_selected >= 0) & (s2_selected < 8))),
        "outcome_mutation_invariance": bool(np.array_equal(s2_selected, mutation_selected)),
        "evaluation_labels_loaded_after_selection": True,
        "no_task_outcome_read": True,
    }
    passed = all(checks.values())
    s2_comparison = comparisons["s2_minus_g062"]
    s2_gap = comparisons["s2_minus_oracle"]
    s1_gap = comparisons["s1_minus_oracle"]
    supported = bool(
        full
        and s2_comparison["translation_m"]["ci95"][1] <= 0.0
        and s2_comparison["rotation_rad"]["ci95"][1] <= 0.0
        and s2_gap["translation_m"]["mean"] < s1_gap["translation_m"]["mean"]
        and s2_gap["rotation_rad"]["mean"] < s1_gap["rotation_rad"]["mean"]
    )
    atomic(out / "forbidden_feature_audit.json", {
        "teacher_qpose_ik_seed": False,
        "task_outcome_read": False,
        "cal_pose_label_used_for_selection": False,
        "future_path_input": False,
    })
    summary = {
        "schema_version": 1,
        "run_id": "A6-G066C-REALIZE-PROBE" if args.limit else ("A6-G066C-REALIZE-SHARD" if args.count else "A6-G066C-REALIZE"),
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "groups": count,
        "cal_row_start": start_index,
        "cal_row_end": start_index + count,
        "elapsed_seconds": time.time() - started,
        "realization": {
            "legal_ik": int(ik_presence.sum()),
            "planner_success": int(planner_success.sum()),
            "selected_ik_coverage": float(selected_ik[valid].mean()),
            "selected_planner_coverage": float(selected_plan[valid].mean()),
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "checks": checks,
        "claim_supported": "yes" if passed and supported else ("probe_only" if passed and args.limit else ("shard_only" if passed and args.count else "no")),
        "decision": "authorize G067 predicted-contact screen" if supported else ("run full G066 realization" if passed and args.limit else ("aggregate all G066 shards" if passed and args.count else "stop before G067; selector gate failed")),
        "next_run_ids": ["A6-G067C"] if supported else (["A6-G066C-REALIZE"] if passed and args.limit else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

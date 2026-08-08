#!/usr/bin/env python3
"""Generate full-horizon TRAIN recovery states with two teacher alignments."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute
from a6_operation_start_contract import load_operation_start
from force_admittance_collect.controller import (
    find_target_joint,
    set_joint_drive_properties,
    target_joint_index,
)
from jointTrain_new.joint_train.sim.capture_view_pcd import (
    ViewPcdCapturer,
    capture_current_world_point_cloud_with_target_mask,
    resolve_urdf,
)
from model.datasets import action_from_npz
from model.online_eval import step_to_qpos
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D160C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D180C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    PARTNET_DATASET_ROOT,
    PROJECT_ROOT,
)
from run_a6_d160c_train_recovery_supervision import (
    ACTION_DIM,
    EXECUTE_PREFIX,
    HORIZON,
    chunk,
    operation_bounds,
    sha256_file,
)
from run_a6_o000b_shared_input_contract import task_metadata
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_d180c_full_horizon_recovery_v1"
SNAPSHOTS_PER_TARGET = 8
TARGET_COUNT = 16


def selected_targets() -> list[dict]:
    selection = json.loads(
        (Path(JOINTTRAIN_ARCH6_D160C_RESULT_ROOT) / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    targets = list(selection["targets"])
    if len(targets) != TARGET_COUNT:
        raise ValueError("D160C selection does not contain 16 targets")
    return targets


def snapshot_calls(max_calls: int) -> list[int]:
    if max_calls < SNAPSHOTS_PER_TARGET:
        raise ValueError("operation horizon is too short for eight snapshots")
    values = np.rint(
        np.geomspace(1.0, float(max_calls), SNAPSHOTS_PER_TARGET)
    ).astype(np.int64) - 1
    values[0] = 0
    values[-1] = max_calls - 1
    if len(np.unique(values)) != SNAPSHOTS_PER_TARGET:
        values = np.rint(
            np.linspace(0, max_calls - 1, SNAPSHOTS_PER_TARGET)
        ).astype(np.int64)
    if len(np.unique(values)) != SNAPSHOTS_PER_TARGET:
        raise ValueError("could not construct unique snapshot calls")
    return values.tolist()


def nearest_progress_anchor(
    object_qpos: np.ndarray,
    joint_index: int,
    operation_start: int,
    operation_stop: int,
    current_joint_qpos: float,
) -> int:
    canonical = np.asarray(
        object_qpos[operation_start : operation_stop - 1, joint_index],
        dtype=np.float64,
    )
    if canonical.size == 0 or not np.isfinite(canonical).all():
        raise ValueError("invalid canonical object progress")
    return operation_start + int(np.argmin(np.abs(canonical - current_joint_qpos)))


def state_vector(history: deque[np.ndarray], command: np.ndarray) -> np.ndarray:
    hist = np.stack(list(history)[-4:])
    previous = np.stack(list(history)[-5:-1])
    qvel = 240.0 * (hist - previous)
    return np.concatenate([hist.reshape(-1), qvel.reshape(-1), command]).astype(
        np.float32
    )


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_D180C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    targets = selected_targets()
    normalizer = json.loads(
        (Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text(
            encoding="utf-8"
        )
    )
    action_std = torch.tensor(normalizer["std"], dtype=torch.float32).reshape(
        1, 1, ACTION_DIM
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("D180C requires CUDA")
    model = OperationMLPAbsolute().to(device)
    model.load_state_dict(
        torch.load(
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
            map_location=device,
            weights_only=False,
        )["model"],
        strict=True,
    )
    model.eval()
    action_std = action_std.to(device)

    common: dict[str, list[np.ndarray]] = {
        "point_cloud": [],
        "target_mask": [],
        "zero_affordance": [],
        "state_history": [],
        "context": [],
        "last_command": [],
    }
    labels = {
        alignment: {
            "absolute_action_target": [],
            "command_delta_target": [],
            "action_valid": [],
        }
        for alignment in ("time", "progress")
    }
    rows: list[dict] = []
    started = time.perf_counter()
    capturer = ViewPcdCapturer(
        articu_root=PROJECT_ROOT,
        partnet_root=PARTNET_DATASET_ROOT,
        render_enabled=True,
        settle_steps=0,
    )
    try:
        for target_index, item in enumerate(targets):
            trajectory = Path(ARTICU_COLLECTION_ROOT) / item["trajectory_relative_path"]
            if sha256_file(trajectory) != item["source_sha256"]:
                raise ValueError(f"source hash drift: {trajectory}")
            init = json.loads(
                (trajectory.parents[1] / "initial_state.json").read_text(
                    encoding="utf-8"
                )
            )
            with np.load(trajectory, allow_pickle=False) as source:
                actual_qpos = np.asarray(source["actual_joint_qpos"], dtype=np.float32)
                object_qpos = np.asarray(source["object_qpos"], dtype=np.float32)
                action = action_from_npz(
                    source, source="joint_command_qpos_repaired", include_finger=False
                )
                operation_start, operation_stop = operation_bounds(source["action_phase"])
            start = load_operation_start(trajectory)
            if start["operation_index"] != operation_start:
                raise ValueError("operation index mismatch")
            max_calls = int(np.ceil((operation_stop - operation_start - 1) / EXECUTE_PREFIX))
            selected_calls = snapshot_calls(max_calls)
            selected_call_set = set(selected_calls)
            link = str(item["target"]).split("/", 1)[1]
            world = capturer._get_world(
                resolve_urdf(init["object_urdf"], partnet_root=PARTNET_DATASET_ROOT),
                float(init["size"]),
            )
            set_joint_drive_properties(
                world.robot, finger_stiffness=4000.0, finger_damping=800.0
            )
            world.configure_contact_friction(
                link, static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            )
            joint = find_target_joint(world.object, link)
            if joint is None:
                raise ValueError(f"target joint not found: {item['target']}")
            joint_index = target_joint_index(world.object, joint)
            if joint_index >= object_qpos.shape[1]:
                raise ValueError("source object qpos does not contain target joint")
            capturer.apply_initial_state(world, init)
            world.object.set_qpos(start["object_qpos"])
            world.object.set_qvel(np.zeros_like(np.asarray(world.object.get_qvel())))
            world.set_robot_qpos(start["robot_qpos"])
            world.robot.set_qvel(np.zeros_like(np.asarray(world.robot.get_qvel())))
            command = np.asarray(start["command_qpos"], dtype=np.float32).copy()
            history: deque[np.ndarray] = deque(
                [np.asarray(start["robot_qpos"], dtype=np.float32).copy() for _ in range(5)],
                maxlen=5,
            )
            metadata = task_metadata(item["target"])
            camera = None
            for call_index in range(max_calls):
                cloud, mask, camera, _, _ = capture_current_world_point_cloud_with_target_mask(
                    world, link, camera=camera
                )
                state = state_vector(history, command)
                context = np.concatenate(
                    [np.zeros(34, dtype=np.float32), metadata]
                ).astype(np.float32)
                current_joint_qpos = float(world.object.get_qpos()[joint_index])
                time_anchor = min(
                    operation_start + call_index * EXECUTE_PREFIX,
                    operation_stop - 2,
                )
                progress_anchor = nearest_progress_anchor(
                    object_qpos,
                    joint_index,
                    operation_start,
                    operation_stop,
                    current_joint_qpos,
                )
                if call_index in selected_call_set:
                    common["point_cloud"].append(np.asarray(cloud, dtype=np.float32))
                    common["target_mask"].append(np.asarray(mask, dtype=bool))
                    common["zero_affordance"].append(
                        np.zeros(1024, dtype=np.float32)
                    )
                    common["state_history"].append(state)
                    common["context"].append(context)
                    common["last_command"].append(command.copy())
                    valid_by_alignment = {}
                    for alignment, anchor in (
                        ("time", time_anchor),
                        ("progress", progress_anchor),
                    ):
                        absolute, valid = chunk(action, anchor, operation_stop)
                        labels[alignment]["absolute_action_target"].append(absolute)
                        labels[alignment]["command_delta_target"].append(
                            absolute - command[None, :]
                        )
                        labels[alignment]["action_valid"].append(valid)
                        valid_by_alignment[alignment] = int(valid.sum())
                    rows.append(
                        {
                            "row_index": len(rows),
                            "target": item["target"],
                            "target_index": target_index,
                            "split": "A5_TRAIN",
                            "trajectory_relative_path": item[
                                "trajectory_relative_path"
                            ],
                            "source_sha256": item["source_sha256"],
                            "live_call": call_index,
                            "max_calls": max_calls,
                            "snapshot_calls": selected_calls,
                            "time_anchor_raw_index": time_anchor,
                            "progress_anchor_raw_index": progress_anchor,
                            "alignment_offset_rows": progress_anchor - time_anchor,
                            "current_target_joint_qpos": current_joint_qpos,
                            "time_state_deviation_l2_arm": float(
                                np.linalg.norm(
                                    np.asarray(world.robot.get_qpos(), dtype=np.float32)[:7]
                                    - actual_qpos[time_anchor, :7]
                                )
                            ),
                            "progress_state_deviation_l2_arm": float(
                                np.linalg.norm(
                                    np.asarray(world.robot.get_qpos(), dtype=np.float32)[:7]
                                    - actual_qpos[progress_anchor, :7]
                                )
                            ),
                            "valid_steps": valid_by_alignment,
                            "operation_start_command_source": start["command_source"],
                            "model_input_oracle_fields": [],
                            "label_only_oracle_fields": [
                                "current_target_joint_qpos",
                                "source_object_qpos",
                            ],
                        }
                    )

                inputs = (
                    torch.from_numpy(cloud[None]).to(device),
                    torch.from_numpy(mask[None]).to(device),
                    torch.zeros((1, 1024), device=device),
                    torch.from_numpy(state[None]).to(device),
                    torch.from_numpy(context[None]).to(device),
                )
                with torch.no_grad():
                    prediction = model(*inputs)
                    actions = (
                        torch.from_numpy(command[None, None, :]).to(device)
                        + prediction * action_std
                    ).cpu().numpy()[0]
                for action_row in actions[:EXECUTE_PREFIX]:
                    if not step_to_qpos(
                        world,
                        action_row,
                        1,
                        float(action_row[-1]),
                        operation_controller=None,
                        drive_mode="drive",
                    ):
                        raise RuntimeError("D180C rollout step failed")
                    command = np.asarray(action_row, dtype=np.float32)
                    history.append(
                        np.asarray(world.robot.get_qpos(), dtype=np.float32)[:ACTION_DIM]
                    )
    finally:
        capturer.close()

    common_arrays = {name: np.stack(values) for name, values in common.items()}
    recovery_arrays: dict[str, dict[str, np.ndarray]] = {}
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        prefix = {name: np.asarray(source[name]) for name in source.files}
    model_keys = tuple(prefix)
    augmented_arrays: dict[str, dict[str, np.ndarray]] = {}
    for alignment in ("time", "progress"):
        recovery = {
            name: np.stack(values) for name, values in labels[alignment].items()
        }
        recovery.update(
            {
                name: common_arrays[name]
                for name in (
                    "point_cloud",
                    "target_mask",
                    "zero_affordance",
                    "state_history",
                    "context",
                )
            }
        )
        augmented = {
            name: np.concatenate([prefix[name], recovery[name]], axis=0)
            for name in model_keys
        }
        recovery_arrays[alignment] = recovery
        augmented_arrays[alignment] = augmented
        destination = out / f"train_recovery_{alignment}1152.npz"
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **augmented)
        os.replace(temporary, destination)

    row_count = TARGET_COUNT * SNAPSHOTS_PER_TARGET
    alignment_offsets = np.asarray(
        [row["alignment_offset_rows"] for row in rows], dtype=np.int64
    )
    checks = {
        "targets_16": len({row["target"] for row in rows}) == TARGET_COUNT,
        "snapshots_8_each": len(rows) == row_count
        and all(
            sum(row["target"] == target for row in rows) == SNAPSHOTS_PER_TARGET
            for target in {row["target"] for row in rows}
        ),
        "full_horizon_spread": all(
            row["snapshot_calls"][0] == 0
            and row["snapshot_calls"][-1] == row["max_calls"] - 1
            for row in rows
        ),
        "correct_start_command": all(
            row["operation_start_command_source"]
            == "logged_operation_start_joint_command_qpos"
            for row in rows
        ),
        "train_only": all(row["split"] == "A5_TRAIN" for row in rows),
        "model_input_oracle_fields_empty": all(
            not row["model_input_oracle_fields"] for row in rows
        ),
        "common_shapes": common_arrays["state_history"].shape == (row_count, 81)
        and common_arrays["context"].shape == (row_count, 43)
        and common_arrays["point_cloud"].shape == (row_count, 1024, 3),
        "finite": all(np.isfinite(value).all() for value in common_arrays.values())
        and all(
            np.isfinite(value).all()
            for recovery in recovery_arrays.values()
            for value in recovery.values()
        ),
        "zero_contact": not bool(np.count_nonzero(common_arrays["context"][:, :34])),
        "zero_affordance": not bool(
            np.count_nonzero(common_arrays["zero_affordance"])
        ),
        "alignment_changes_labels": bool(np.count_nonzero(alignment_offsets)),
        "prefix_exact": all(
            np.array_equal(augmented_arrays[alignment][name][:1024], prefix[name])
            for alignment in ("time", "progress")
            for name in model_keys
        ),
    }
    passed = all(checks.values())
    atomic_json(
        out / "selection.json",
        {
            "targets": targets,
            "snapshots_per_target": SNAPSHOTS_PER_TARGET,
            "snapshot_schedule": "eight rounded geometric call indices including first and last",
        },
    )
    atomic_json(out / "recovery_rows.json", {"rows": rows})
    atomic_json(
        out / "forbidden_feature_audit.json",
        {
            "model_input_forbidden_fields": [],
            "object_qpos_in_model_input": False,
            "object_progress_in_model_input": False,
            "object_qpos_used_for_simulator_initialization": True,
            "object_qpos_used_for_progress_teacher_alignment": True,
            "progress_alignment_is_train_label_generation_only": True,
            "cal_mechdev_final_read": False,
            "result_json_read": False,
        },
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "scientific_scope": "full-horizon TRAIN-only recovery state generation",
        "checks": checks,
        "metrics": {
            "rows": len(rows),
            "total_policy_calls": int(sum(row["max_calls"] for row in rows[::8])),
            "alignment_changed_rows": int(np.count_nonzero(alignment_offsets)),
            "alignment_offset_abs_median": float(np.median(np.abs(alignment_offsets))),
            "alignment_offset_abs_max": int(np.max(np.abs(alignment_offsets))),
            "time_state_deviation_mean": float(
                np.mean([row["time_state_deviation_l2_arm"] for row in rows])
            ),
            "progress_state_deviation_mean": float(
                np.mean([row["progress_state_deviation_l2_arm"] for row in rows])
            ),
            "wall_seconds": time.perf_counter() - started,
        },
        "decision": "authorize matched time/progress recovery training"
        if passed
        else "repair D180C before training",
        "next_run_ids": ["A6-O181C", "A6-O182C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-D180C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

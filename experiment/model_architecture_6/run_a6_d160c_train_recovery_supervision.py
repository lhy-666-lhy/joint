#!/usr/bin/env python3
"""Generate TRAIN-only live-state recovery supervision with frozen O127C."""

from __future__ import annotations

import hashlib
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
    JOINTTRAIN_ARCH6_D040C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D160C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    PARTNET_DATASET_ROOT,
    PROJECT_ROOT,
)
from run_a6_o000b_shared_input_contract import task_metadata
from run_a6_o010r_mlp_fixed64 import atomic_json
from a6_operation_start_contract import load_operation_start


RUN_ID = "a6_d160c_train_recovery_supervision_v1"
TARGET_COUNT = 16
CALLS_PER_TARGET = 4
EXECUTE_PREFIX = 8
HORIZON = 32
ACTION_DIM = 9
SEED = 20260805


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def operation_bounds(phase: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(np.asarray(phase).astype(str) == "operation")
    if indices.size < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError("invalid operation phase")
    return int(indices[0]), int(indices[-1]) + 1


def chunk(action: np.ndarray, anchor: int, operation_stop: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        action[anchor + 1 : min(operation_stop, anchor + 1 + HORIZON), :ACTION_DIM],
        dtype=np.float32,
    )
    if values.shape[0] == 0:
        raise ValueError("empty recovery teacher chunk")
    valid = np.zeros((HORIZON,), dtype=bool)
    valid[: values.shape[0]] = True
    if values.shape[0] < HORIZON:
        values = np.concatenate(
            [values, np.repeat(values[-1:], HORIZON - values.shape[0], axis=0)], axis=0
        )
    return values, valid


def select_targets() -> list[dict]:
    manifest = json.loads(
        (Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT) / "full" / "input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    selected: list[dict] = []
    seen: set[str] = set()
    for row in manifest["rows"]:
        target = str(row["target"])
        if target in seen:
            continue
        seen.add(target)
        selected.append(
            {
                "target": target,
                "trajectory_relative_path": str(row["trajectory_relative_path"]),
                "source_sha256": str(row["source_sha256"]),
            }
        )
        if len(selected) == TARGET_COUNT:
            break
    if len(selected) != TARGET_COUNT:
        raise ValueError("not enough clean TRAIN targets")
    return selected


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_D160C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    selected = select_targets()
    normalizer = json.loads(
        (Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text(
            encoding="utf-8"
        )
    )
    action_std = torch.tensor(normalizer["std"], dtype=torch.float32).reshape(1, 1, ACTION_DIM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    arrays: dict[str, list[np.ndarray]] = {
        "point_cloud": [],
        "target_mask": [],
        "zero_affordance": [],
        "state_history": [],
        "context": [],
        "absolute_action_target": [],
        "command_delta_target": [],
        "action_valid": [],
        "last_command": [],
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
        for target_index, item in enumerate(selected):
            trajectory = Path(ARTICU_COLLECTION_ROOT) / item["trajectory_relative_path"]
            if sha256_file(trajectory) != item["source_sha256"]:
                raise ValueError(f"source hash drift: {trajectory}")
            init = json.loads((trajectory.parents[1] / "initial_state.json").read_text(encoding="utf-8"))
            with np.load(trajectory, allow_pickle=False) as source:
                actual_qpos = np.asarray(source["actual_joint_qpos"], dtype=np.float32)
                action_command = np.asarray(source["joint_command_qpos"], dtype=np.float32)
                action = action_from_npz(
                    source, source="joint_command_qpos_repaired", include_finger=False
                )
                operation_start, operation_stop = operation_bounds(source["action_phase"])
            start = load_operation_start(trajectory)
            if operation_start != start["operation_index"]:
                raise ValueError("operation start index changed between readers")
            start_robot = np.asarray(start["robot_qpos"], dtype=np.float32).reshape(-1)
            start_command = np.asarray(start["command_qpos"], dtype=np.float32).reshape(-1)
            start_object = np.asarray(start["object_qpos"], dtype=np.float32).reshape(-1)
            if start_object.size == 0 or not np.isfinite(start_object).all():
                start_object = np.asarray(init["initial_object_qpos"], dtype=np.float32).reshape(-1)
            if start_robot.shape[0] < ACTION_DIM or start_command.shape[0] < ACTION_DIM:
                raise ValueError("invalid operation start state")
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
            target_joint = find_target_joint(world.object, link)
            if target_joint is None:
                raise ValueError(f"target joint not found: {item['target']}")
            _ = target_joint_index(world.object, target_joint)
            capturer.apply_initial_state(world, init)
            world.object.set_qpos(start_object)
            world.object.set_qvel(np.zeros_like(np.asarray(world.object.get_qvel())))
            world.set_robot_qpos(start_robot)
            world.robot.set_qvel(np.zeros_like(np.asarray(world.robot.get_qvel())))
            command = start_command[:ACTION_DIM].astype(np.float32).copy()
            history = deque(
                [start_robot.astype(np.float32).copy() for _ in range(5)], maxlen=5
            )
            metadata = task_metadata(item["target"])
            for call_index in range(CALLS_PER_TARGET):
                cloud, mask, camera, _, _ = capture_current_world_point_cloud_with_target_mask(
                    world, link, camera=None if call_index == 0 else camera
                )
                hist = np.stack(list(history)[-4:])
                previous = np.stack(list(history)[-5:-1])
                qvel = 240.0 * (hist - previous)
                state = np.concatenate(
                    [hist.reshape(-1), qvel.reshape(-1), command], axis=0
                ).astype(np.float32)
                context = np.concatenate(
                    [np.zeros(34, dtype=np.float32), metadata], axis=0
                ).astype(np.float32)
                teacher_anchor = min(
                    operation_start + call_index * EXECUTE_PREFIX,
                    operation_stop - 2,
                )
                absolute, valid = chunk(action, teacher_anchor, operation_stop)
                delta = absolute - command[None, :]
                deviation = float(
                    np.linalg.norm(
                        np.asarray(world.robot.get_qpos(), dtype=np.float32)[:7]
                        - actual_qpos[teacher_anchor, :7]
                    )
                )
                row_index = len(rows)
                arrays["point_cloud"].append(np.asarray(cloud, dtype=np.float32))
                arrays["target_mask"].append(np.asarray(mask, dtype=bool))
                arrays["zero_affordance"].append(np.zeros(1024, dtype=np.float32))
                arrays["state_history"].append(state)
                arrays["context"].append(context)
                arrays["absolute_action_target"].append(absolute)
                arrays["command_delta_target"].append(delta)
                arrays["action_valid"].append(valid)
                arrays["last_command"].append(command.copy())
                rows.append(
                    {
                        "row_index": row_index,
                        "target": item["target"],
                        "trajectory_relative_path": item["trajectory_relative_path"],
                        "source_sha256": item["source_sha256"],
                        "split": "A5_TRAIN",
                        "live_call": call_index,
                        "teacher_anchor_raw_index": int(teacher_anchor),
                        "teacher_source": "joint_command_qpos_repaired",
                        "state_deviation_l2_arm": deviation,
                        "observation_source": "live_sapiens_observation",
                        "model_input_oracle_fields": [],
                        "operation_start_command_source": start["command_source"],
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
                        + prediction * action_std.to(device)
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
                        raise RuntimeError(
                            f"recovery teacher rollout step failed: {item['target']} call {call_index}"
                        )
                    command = np.asarray(action_row, dtype=np.float32)
                    history.append(np.asarray(world.robot.get_qpos(), dtype=np.float32)[:9])
    finally:
        capturer.close()

    stacked = {name: np.stack(values) for name, values in arrays.items()}
    input_path = out / "recovery_input.npz"
    temporary = input_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **stacked)
    os.replace(temporary, input_path)
    model_input_keys = (
        "point_cloud",
        "target_mask",
        "zero_affordance",
        "state_history",
        "context",
        "absolute_action_target",
        "command_delta_target",
        "action_valid",
    )
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        original = {name: np.asarray(source[name]) for name in model_input_keys}
    augmented = {
        name: np.concatenate([original[name], stacked[name]], axis=0)
        for name in model_input_keys
    }
    augmented_path = out / "train_recovery1088.npz"
    augmented_temporary = augmented_path.with_suffix(".tmp.npz")
    np.savez_compressed(augmented_temporary, **augmented)
    os.replace(augmented_temporary, augmented_path)
    source_hashes = sorted({row["source_sha256"] for row in rows})
    checks = {
        "target_count_16": len({row["target"] for row in rows}) == TARGET_COUNT,
        "rows_64": len(rows) == TARGET_COUNT * CALLS_PER_TARGET,
        "augmented_rows_1088": all(value.shape[0] == 1088 for value in augmented.values()),
        "original_prefix_exact": all(
            np.array_equal(augmented[name][:1024], original[name])
            for name in model_input_keys
        ),
        "sidecars_not_in_model_input": set(augmented) == set(model_input_keys),
        "four_calls_each": all(
            sum(row["target"] == target and row["live_call"] == call for row in rows) == 1
            for target in {row["target"] for row in rows}
            for call in range(CALLS_PER_TARGET)
        ),
        "train_only": all(row["split"] == "A5_TRAIN" for row in rows),
        "state_context_shape": stacked["state_history"].shape == (64, 81)
        and stacked["context"].shape == (64, 43),
        "point_mask_shape": stacked["point_cloud"].shape == (64, 1024, 3)
        and stacked["target_mask"].shape == (64, 1024),
        "zero_contact": not bool(np.count_nonzero(stacked["context"][:, :34])),
        "zero_affordance": not bool(np.count_nonzero(stacked["zero_affordance"])),
        "finite": all(np.isfinite(value).all() for value in stacked.values()),
        "action_valid": bool(stacked["action_valid"].any(axis=1).all()),
        "delta_roundtrip": float(
            np.max(
                np.abs(
                    stacked["last_command"][:, None, :]
                    + stacked["command_delta_target"]
                    - stacked["absolute_action_target"]
                )
            )
        )
        <= 1e-6,
        "source_hashes_recorded": len(source_hashes) == TARGET_COUNT,
        "model_input_oracle_fields_empty": all(
            not row["model_input_oracle_fields"] for row in rows
        ),
        "no_cal_or_heldout_read": True,
    }
    passed = all(checks.values())
    atomic_json(
        out / "selection.json",
        {
            "target_count": TARGET_COUNT,
            "calls_per_target": CALLS_PER_TARGET,
            "execute_prefix": EXECUTE_PREFIX,
            "targets": selected,
            "source_hashes": source_hashes,
        },
    )
    atomic_json(out / "recovery_rows.json", {"rows": rows})
    atomic_json(
        out / "input_manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "split": "A5_TRAIN",
            "observation_source": "live_sapiens_observation",
            "array_shapes": {name: list(value.shape) for name, value in stacked.items()},
            "input_sha256": sha256_file(input_path),
            "augmented_input_sha256": sha256_file(augmented_path),
            "source_hashes": source_hashes,
            "teacher_source": "TRAIN canonical future command only",
        },
    )
    atomic_json(
        out / "forbidden_feature_audit.json",
        {
            "loaded_fields_for_training_input": [
                "actual_joint_qpos",
                "joint_command_qpos",
                "action_phase",
                "joint_command_qpos_repaired",
                "point_cloud",
                "target_mask",
            ],
            "model_input_forbidden_fields": [],
            "object_qpos_used_only_for_simulator_initialization": True,
            "object_qpos_in_model_input": False,
            "object_progress_in_model_input": False,
            "contact_feedback_in_model_input": False,
            "teacher_action_in_model_input": False,
            "future_label_used_only_for_train_supervision": True,
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
        "scientific_scope": "TRAIN-only live-state recovery supervision generation",
        "checks": checks,
        "metrics": {
            "wall_seconds": time.perf_counter() - started,
            "state_deviation_l2_mean": float(np.mean([row["state_deviation_l2_arm"] for row in rows])),
            "state_deviation_l2_median": float(np.median([row["state_deviation_l2_arm"] for row in rows])),
            "state_deviation_l2_max": float(np.max([row["state_deviation_l2_arm"] for row in rows])),
        },
        "decision": "recovery rows valid; authorize additive recovery training" if passed else "repair recovery data contract",
        "next_run_ids": ["A6-O161C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-D160C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

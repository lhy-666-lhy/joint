"""Shared physical executor for Architecture 6 grasp-interface pilots."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from a6_operation_models import OperationMLPRecoveryResidual
from force_admittance_collect.feedback import (
    classify_grasp_contact,
    read_target_contact_feedback,
)
from jointTrain_new.experiment.model_architecture_2.eval_architecture_v2_teleport import (
    planned_qpath_schedule,
)
from jointTrain_new.joint_train.sim.capture_view_pcd import (
    capture_current_world_point_cloud_with_target_mask,
)
from model.online_eval import step_to_qpos
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O185C_RESULT_ROOT,
)
from run_a6_o000b_shared_input_contract import task_metadata
from run_a6_o126c_zero_contact_live_probe import independent_world


FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0
HOLD_OPEN_STEPS = 30
CLOSE_STEPS = 80
SETTLE_STEPS = 120
MAX_POLICY_CALLS = 650
EXECUTE_PREFIX = 8


def load_operation_policy(device: torch.device):
    checkpoint = Path(JOINTTRAIN_ARCH6_O185C_RESULT_ROOT) / "last.pt"
    model = OperationMLPRecoveryResidual().to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    model.eval()
    normalizer = json.loads(
        (Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text(
            encoding="utf-8"
        )
    )
    std = torch.tensor(normalizer["std"], dtype=torch.float32, device=device).reshape(
        1, 1, 9
    )
    return model, std, checkpoint


def _lock_target(world, target_index: int, target_qpos: float) -> None:
    qpos = np.asarray(world.object.get_qpos(), dtype=np.float64).copy()
    qvel = np.asarray(world.object.get_qvel(), dtype=np.float64).copy()
    qpos[target_index] = target_qpos
    qvel[target_index] = 0.0
    world.object.set_qpos(qpos)
    world.object.set_qvel(qvel)


def _execute_locked(
    world,
    arm_qpos: np.ndarray,
    finger: float,
    ticks: int,
    target_index: int,
    target_qpos: float,
) -> None:
    command = np.concatenate(
        (np.asarray(arm_qpos, dtype=np.float32).reshape(7), [finger, finger])
    ).astype(np.float32)
    for _ in range(int(ticks)):
        if not step_to_qpos(
            world,
            command,
            1,
            float(finger),
            operation_controller=None,
            drive_mode="drive",
        ):
            raise RuntimeError("grasp qpath executor stopped")
        _lock_target(world, target_index, target_qpos)


def _execute_grasp(world, qpath: np.ndarray, link: str, target_index: int) -> dict:
    path = np.asarray(qpath, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 7:
        raise ValueError(f"physical pilot requires a [T>=2,7] path, got {path.shape}")
    if not np.isfinite(path).all():
        raise ValueError("physical pilot path is nonfinite")
    target_qpos = float(world.object.get_qpos()[target_index])
    schedule, schedule_metrics = planned_qpath_schedule(
        path,
        max_ticks=8,
        min_joint_step=0.003,
        reference_joint_step=0.02,
    )
    for path_index, ticks in schedule:
        _execute_locked(
            world, path[path_index], FINGER_OPEN, ticks, target_index, target_qpos
        )
    _execute_locked(
        world, path[-1], FINGER_OPEN, HOLD_OPEN_STEPS, target_index, target_qpos
    )
    _execute_locked(
        world, path[-1], FINGER_CLOSED, CLOSE_STEPS, target_index, target_qpos
    )
    _execute_locked(
        world, path[-1], FINGER_CLOSED, SETTLE_STEPS, target_index, target_qpos
    )
    actual_arm = np.asarray(world.robot.get_qpos(), dtype=np.float32)[:7]
    drift = abs(float(world.object.get_qpos()[target_index]) - target_qpos)
    feedback = read_target_contact_feedback(world, link)
    return {
        **schedule_metrics,
        "hold_open_steps": HOLD_OPEN_STEPS,
        "close_steps": CLOSE_STEPS,
        "settle_steps": SETTLE_STEPS,
        "terminal_tracking_max_error": float(np.max(np.abs(actual_arm - path[-1]))),
        "target_joint_abs_drift": drift,
        "target_lock_qpos": target_qpos,
        "feedback": feedback,
    }


def _run_operation(
    world,
    link: str,
    target: str,
    target_index: int,
    span: float,
    model,
    std: torch.Tensor,
    device: torch.device,
) -> dict:
    command = np.asarray(world.robot.get_qpos(), dtype=np.float32)[:9].copy()
    command[7:9] = FINGER_CLOSED
    history = deque(
        [np.asarray(world.robot.get_qpos(), dtype=np.float32)[:9].copy() for _ in range(5)],
        maxlen=5,
    )
    meta = task_metadata(target)
    start_joint = float(world.object.get_qpos()[target_index])
    camera = None
    trace = []
    inference_seconds = 0.0
    reason = "max_calls"
    for call in range(MAX_POLICY_CALLS):
        cloud, mask, camera, _, _ = capture_current_world_point_cloud_with_target_mask(
            world, link, camera=camera
        )
        hist = np.stack(list(history)[-4:])
        prev = np.stack(list(history)[-5:-1])
        qvel = 240.0 * (hist - prev)
        state = np.concatenate((hist.reshape(-1), qvel.reshape(-1), command)).astype(
            np.float32
        )
        context = np.concatenate((np.zeros(34, dtype=np.float32), meta))
        inputs = (
            torch.from_numpy(cloud[None]).to(device),
            torch.from_numpy(mask[None]).to(device),
            torch.zeros((1, 1024), dtype=torch.float32, device=device),
            torch.from_numpy(state[None]).to(device),
            torch.from_numpy(context[None]).to(device),
        )
        started = time.perf_counter()
        with torch.no_grad():
            actions = (
                torch.from_numpy(command[None, None]).to(device) + model(*inputs) * std
            ).cpu().numpy()[0]
        inference_seconds += time.perf_counter() - started
        for action in actions[:EXECUTE_PREFIX]:
            if not step_to_qpos(
                world,
                action,
                1,
                float(action[-1]),
                operation_controller=None,
                drive_mode="drive",
            ):
                raise RuntimeError("operation executor stopped")
            command = np.asarray(action, dtype=np.float32)
            history.append(np.asarray(world.robot.get_qpos(), dtype=np.float32)[:9])
        progress = (float(world.object.get_qpos()[target_index]) - start_joint) / span
        feedback = read_target_contact_feedback(world, link)
        trace.append(
            {
                "call": call + 1,
                "progress": progress,
                "target_contact": bool(feedback.target_contact),
                "gripper_contact": bool(feedback.gripper_target_contact),
            }
        )
        if progress >= 0.4:
            reason = "opening_stop"
            break
    return {
        "calls": len(trace),
        "physics_steps": len(trace) * EXECUTE_PREFIX,
        "termination": reason,
        "task_success": reason == "opening_stop",
        "final_progress": float(trace[-1]["progress"]),
        "contact_fraction": float(np.mean([row["target_contact"] for row in trace])),
        "gripper_contact_fraction": float(
            np.mean([row["gripper_contact"] for row in trace])
        ),
        "inference_seconds": inference_seconds,
        "trace": trace,
    }


def run_physical_episode(
    *,
    route: str,
    group_index: int,
    group_id: str,
    sample_id: str,
    target: str,
    init: dict,
    qpath: np.ndarray,
    model,
    std: torch.Tensor,
    device: torch.device,
) -> dict:
    link = target.split("/", 1)[1]
    started = time.perf_counter()
    with independent_world(init, link) as (capturer, world, target_index, span):
        capturer.apply_initial_state(world, init)
        grasp = _execute_grasp(world, qpath, link, target_index)
        feedback = grasp.pop("feedback")
        grasp_quality = classify_grasp_contact(
            feedback, target_joint_abs_drift=grasp["target_joint_abs_drift"]
        )
        operation = _run_operation(
            world, link, target, target_index, span, model, std, device
        )
    return {
        "route": route,
        "group_index": int(group_index),
        "group_id": group_id,
        "sample_id": sample_id,
        "target": target,
        "fresh_world": True,
        "qpath_shape": list(np.asarray(qpath).shape),
        "grasp": {**grasp, **grasp_quality},
        "operation": operation,
        "wall_seconds": time.perf_counter() - started,
    }

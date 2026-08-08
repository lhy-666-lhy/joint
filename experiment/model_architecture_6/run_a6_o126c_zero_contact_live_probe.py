#!/usr/bin/env python3
"""Architecture 2/3-style live closed-loop probe for zero-contact A6 policies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute, OperationParallelAbsolute
from a6_operation_start_contract import load_operation_start
from force_admittance_collect.controller import (
    find_target_joint,
    set_joint_drive_properties,
    target_joint_index,
)
from force_admittance_collect.feedback import read_target_contact_feedback
from jointTrain_new.joint_train.sim.capture_view_pcd import (
    ViewPcdCapturer,
    capture_current_world_point_cloud_with_target_mask,
    resolve_urdf,
)
from model.online_eval import step_to_qpos
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O122C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O123C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O126C_RESULT_ROOT,
    PARTNET_DATASET_ROOT,
    PROJECT_ROOT,
)
from run_a6_o000b_shared_input_contract import task_metadata
from run_a6_o010r_mlp_fixed64 import atomic_json


ARMS = {
    "mlp": (
        OperationMLPAbsolute,
        Path(JOINTTRAIN_ARCH6_O122C_RESULT_ROOT) / "last.pt",
    ),
    "parallel": (
        OperationParallelAbsolute,
        Path(JOINTTRAIN_ARCH6_O123C_RESULT_ROOT) / "last.pt",
    ),
    "repeat_last": (None, None),
}
STATE_APPENDERS = {}
MODEL_INPUT_FEATURES = {}
RUN_ID = "a6_o126c_zero_contact_live_probe_v1"
SCIENTIFIC_SCOPE = "bounded live closed-loop interface probe"
WORLD_RESET_MODE = "independent_world_per_arm"


def start_fields(path):
    record = load_operation_start(path)
    return record["robot_qpos"], record["object_qpos"], record["command_qpos"]


@contextmanager
def independent_world(init: dict, link: str):
    """Construct and configure a fresh SAPIEN world for exactly one policy arm."""
    world = None
    capturer = ViewPcdCapturer(
        articu_root=PROJECT_ROOT,
        partnet_root=PARTNET_DATASET_ROOT,
        render_enabled=True,
        settle_steps=0,
    )
    try:
        world = capturer._get_world(
            resolve_urdf(init["object_urdf"], partnet_root=PARTNET_DATASET_ROOT),
            float(init["size"]),
        )
        set_joint_drive_properties(
            world.robot,
            finger_stiffness=4000.0,
            finger_damping=800.0,
        )
        world.configure_contact_friction(
            link,
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        )
        joint = find_target_joint(world.object, link)
        idx = target_joint_index(world.object, joint)
        limits = np.asarray(joint.get_limits(), dtype=np.float64).reshape(-1, 2)[0]
        span = float(limits[1] - limits[0])
        yield capturer, world, idx, span
    finally:
        capturer.close()
        if world is not None and hasattr(world, "close"):
            world.close()


def evaluate_arm(
    *,
    arm: str,
    model,
    world,
    idx: int,
    span: float,
    init: dict,
    link: str,
    target: str,
    selected: dict,
    start: dict,
    start_robot: np.ndarray,
    start_obj: np.ndarray,
    start_cmd: np.ndarray,
    meta: np.ndarray,
    std: torch.Tensor,
    device: torch.device,
    max_calls: int,
    execute_prefix: int,
    world_creation_index: int,
) -> dict:
    command = start_cmd.copy()
    history = deque(
        [start_robot.astype(np.float32).copy() for _ in range(5)], maxlen=5
    )
    camera = None
    start_joint = float(world.object.get_qpos()[idx])
    trace = []
    inference = 0.0
    observation_seconds = 0.0
    execution_seconds = 0.0
    arm_started = time.perf_counter()
    reason = "max_calls"

    for call in range(max_calls):
        observation_started = time.perf_counter()
        cloud, mask, camera, _, _ = capture_current_world_point_cloud_with_target_mask(
            world, link, camera=camera
        )
        observation_seconds += time.perf_counter() - observation_started
        hist = np.stack(list(history)[-4:])
        prev = np.stack(list(history)[-5:-1])
        qvel = 240.0 * (hist - prev)
        state = np.concatenate([hist.reshape(-1), qvel.reshape(-1), command[:9]]).astype(
            np.float32
        )
        context = np.concatenate([np.zeros(34, dtype=np.float32), meta]).astype(
            np.float32
        )
        if arm in STATE_APPENDERS:
            state = STATE_APPENDERS[arm](world, init, cloud, mask, state)

        if arm == "repeat_last":
            actions = np.repeat(command[None, :9], 32, axis=0).astype(np.float32)
        else:
            inputs = [
                torch.from_numpy(cloud[None]).to(device),
                torch.from_numpy(mask[None]).to(device),
                torch.zeros((1, 1024), device=device),
                torch.from_numpy(state[None]).to(device),
                torch.from_numpy(context[None]).to(device),
            ]
            inference_started = time.perf_counter()
            with torch.no_grad():
                pred = model(*inputs)
                actions = (
                    torch.from_numpy(command[None, None, :9]).to(device) + pred * std
                ).cpu().numpy()[0]
            inference += time.perf_counter() - inference_started

        execution_started = time.perf_counter()
        for action in actions[:execute_prefix]:
            if not step_to_qpos(
                world,
                action,
                1,
                float(action[-1]),
                operation_controller=None,
                drive_mode="drive",
            ):
                raise RuntimeError("world step failed")
            command = action.astype(np.float32)
            history.append(np.asarray(world.robot.get_qpos(), dtype=np.float32)[:9])
        execution_seconds += time.perf_counter() - execution_started

        current = float(world.object.get_qpos()[idx])
        progress = (current - start_joint) / span
        feedback = read_target_contact_feedback(world, link)
        trace.append(
            {
                "call": call + 1,
                "progress": progress,
                "target_qpos": current,
                "target_contact": feedback.target_contact,
                "gripper_contact": feedback.gripper_target_contact,
                "contact_points": feedback.point_count,
            }
        )
        if progress >= 0.4:
            reason = "opening_stop"
            break

    return {
        "arm": arm,
        "target": target,
        "trajectory_relative_path": selected["trajectory_relative_path"],
        "max_calls": max_calls,
        "execute_prefix": execute_prefix,
        "max_physics_steps": max_calls * execute_prefix,
        "calls": len(trace),
        "physics_steps": len(trace) * execute_prefix,
        "termination": reason,
        "final_progress": trace[-1]["progress"],
        "contact_fraction": float(np.mean([x["target_contact"] for x in trace])),
        "gripper_contact_fraction": float(
            np.mean([x["gripper_contact"] for x in trace])
        ),
        "inference_seconds": inference,
        "observation_seconds": observation_seconds,
        "execution_seconds": execution_seconds,
        "wall_seconds": time.perf_counter() - arm_started,
        "trace": trace,
        "model_input_features": MODEL_INPUT_FEATURES.get(arm, []),
        "model_input_oracle_fields": [],
        "evaluator_only_object_qpos": True,
        "operation_start_index": start["operation_index"],
        "operation_start_command_source": start["command_source"],
        "operation_start_command_finger": start_cmd[7:9].tolist(),
        "start_command_vs_logged_max_abs": None
        if start["logged_command_qpos"] is None
        else float(np.max(np.abs(start_cmd - start["logged_command_qpos"]))),
        "start_command_vs_raw_max_abs": None
        if start["raw_command_qpos"] is None
        else float(np.max(np.abs(start_cmd - start["raw_command_qpos"]))),
        "start_command_vs_repaired_max_abs": float(
            np.max(np.abs(start_cmd - start["repaired_command_qpos"]))
        ),
        "world_reset_mode": WORLD_RESET_MODE,
        "world_creation_index": world_creation_index,
        "world_configured_for_arm": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=3)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--execute-prefix", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.execute_prefix <= 32:
        raise ValueError("--execute-prefix must be between 1 and 32")

    manifest = json.load(
        open(Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT) / "full/input_manifest.json")
    )
    first = []
    for row in manifest["rows"]:
        if row["anchor_rank"] == 0 and row["target"] not in [x["target"] for x in first]:
            first.append(row)
    selected = first[args.target_index]
    trajectory = Path(ARTICU_COLLECTION_ROOT) / selected["trajectory_relative_path"]
    init = json.load(open(trajectory.parents[1] / "initial_state.json"))
    start = load_operation_start(trajectory)
    start_robot = start["robot_qpos"]
    start_obj = start["object_qpos"]
    start_cmd = start["command_qpos"]
    target = str(selected["target"])
    link = target.split("/", 1)[1]
    normalizer = json.load(
        open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    std = torch.tensor(normalizer["std"], dtype=torch.float32).reshape(1, 1, 9).to(device)
    meta = task_metadata(target)

    models = {}
    for arm, (factory, path) in ARMS.items():
        if factory is None:
            continue
        model = factory().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["model"], strict=True
        )
        model.eval()
        models[arm] = model

    rows = []
    for world_creation_index, arm in enumerate(ARMS):
        with independent_world(init, link) as (capturer, world, idx, span):
            capturer.apply_initial_state(world, init)
            world.object.set_qpos(start_obj)
            world.object.set_qvel(np.zeros_like(world.object.get_qvel()))
            world.set_robot_qpos(start_robot)
            world.robot.set_qvel(np.zeros_like(world.robot.get_qvel()))
            row = evaluate_arm(
                arm=arm,
                model=models.get(arm),
                world=world,
                idx=idx,
                span=span,
                init=init,
                link=link,
                target=target,
                selected=selected,
                start=start,
                start_robot=start_robot,
                start_obj=start_obj,
                start_cmd=start_cmd,
                meta=meta,
                std=std,
                device=device,
                max_calls=args.max_calls,
                execute_prefix=args.execute_prefix,
                world_creation_index=world_creation_index,
            )
            checkpoint = ARMS[arm][1]
            row["model_checkpoint"] = None if checkpoint is None else str(checkpoint)
        row["world_closed_after_arm"] = True
        rows.append(row)

    checks = {
        "all_configured_arms": len(rows) == len(ARMS),
        "finite": all(
            np.isfinite(
                [
                    row["final_progress"],
                    row["contact_fraction"],
                    row["inference_seconds"],
                    row["observation_seconds"],
                    row["execution_seconds"],
                    row["wall_seconds"],
                ]
            ).all()
            for row in rows
        ),
        "zero_model_oracle_fields": all(not row["model_input_oracle_fields"] for row in rows),
        "calls_bounded": all(row["calls"] <= args.max_calls for row in rows),
        "prefix_exact": all(row["execute_prefix"] == args.execute_prefix for row in rows),
        "physics_budget_recorded": all(
            row["max_physics_steps"] == args.max_calls * args.execute_prefix for row in rows
        ),
        "start_command_source_valid": start["command_source"]
        in {
            "logged_operation_start_joint_command_qpos",
            "joint_command_qpos[operation_start_index]",
            "repaired_joint_command_qpos[operation_start_index]",
        },
        "start_command_finger_zero": bool(np.max(np.abs(start_cmd[7:9])) <= 1e-7),
        "start_command_matches_repaired": bool(
            np.max(np.abs(start_cmd - start["repaired_command_qpos"])) <= 1e-7
        ),
        "independent_world_per_arm": all(
            row["world_reset_mode"] == WORLD_RESET_MODE
            and row["world_configured_for_arm"]
            and row["world_closed_after_arm"]
            for row in rows
        ),
        "world_creation_indices_exact": [row["world_creation_index"] for row in rows]
        == list(range(len(ARMS))),
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_O126C_RESULT_ROOT) / (
        f"probe_calls_{args.max_calls}_target_{args.target_index}"
    )
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 2,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": SCIENTIFIC_SCOPE,
        "execute_prefix": args.execute_prefix,
        "world_reset_mode": WORLD_RESET_MODE,
        "rows": rows,
        "checks": checks,
        "decision": "live adapter valid; expand exact-paired bounded screen"
        if passed
        else "repair live adapter",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

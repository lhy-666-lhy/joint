from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from model.datasets import repaired_joint_command_qpos


ACTION_DIM = 9


def _finite_prefix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] < ACTION_DIM or not np.isfinite(array[:ACTION_DIM]).all():
        raise ValueError(f"invalid {name}")
    return array[:ACTION_DIM].copy()


def load_operation_start(path: str | Path) -> dict[str, Any]:
    """Load the A6 operation reset state without trusting the deprecated command field."""
    with np.load(path, allow_pickle=False) as data:
        phases = np.asarray(data["action_phase"]).astype(str).reshape(-1)
        operation_indices = np.flatnonzero(phases == "operation")
        if operation_indices.size == 0:
            raise ValueError("trajectory has no operation phase")
        operation_index = int(operation_indices[0])
        if "operation_start_index" in data.files:
            stored = np.asarray(data["operation_start_index"]).reshape(-1)
            if stored.size != 1 or int(stored[0]) != operation_index:
                raise ValueError("operation_start_index disagrees with action_phase")

        robot_source = (
            data["operation_start_robot_qpos"]
            if "operation_start_robot_qpos" in data.files
            else data["actual_joint_qpos"][operation_index]
        )
        object_source = (
            data["operation_start_object_qpos"]
            if "operation_start_object_qpos" in data.files
            else data["object_qpos"][operation_index]
        )
        robot = _finite_prefix(robot_source, name="operation start robot qpos")
        object_qpos = np.asarray(object_source, dtype=np.float64).reshape(-1).copy()
        if object_qpos.size == 0 or not np.isfinite(object_qpos).all():
            raise ValueError("invalid operation start object qpos")

        repaired = _finite_prefix(
            repaired_joint_command_qpos(data)[operation_index],
            name="repaired operation start command",
        )
        raw = None
        if "joint_command_qpos" in data.files:
            candidate = np.asarray(data["joint_command_qpos"], dtype=np.float32)
            if candidate.ndim == 2 and operation_index < candidate.shape[0]:
                candidate = candidate[operation_index].reshape(-1)
                if candidate.shape[0] >= ACTION_DIM and np.isfinite(candidate[:ACTION_DIM]).all():
                    raw = candidate[:ACTION_DIM].copy()
        logged = None
        if "logged_operation_start_joint_command_qpos" in data.files:
            candidate = np.asarray(
                data["logged_operation_start_joint_command_qpos"], dtype=np.float32
            ).reshape(-1)
            if candidate.shape[0] >= ACTION_DIM and np.isfinite(candidate[:ACTION_DIM]).all():
                logged = candidate[:ACTION_DIM].copy()

        if logged is not None:
            command = logged
            command_source = "logged_operation_start_joint_command_qpos"
        elif raw is not None:
            command = raw
            command_source = "joint_command_qpos[operation_start_index]"
        else:
            command = repaired
            command_source = "repaired_joint_command_qpos[operation_start_index]"

        bad = None
        if "operation_start_joint_command_qpos" in data.files:
            candidate = np.asarray(
                data["operation_start_joint_command_qpos"], dtype=np.float32
            ).reshape(-1)
            if candidate.shape[0] >= ACTION_DIM and np.isfinite(candidate[:ACTION_DIM]).all():
                bad = candidate[:ACTION_DIM].copy()

        return {
            "operation_index": operation_index,
            "robot_qpos": robot.astype(np.float64),
            "object_qpos": object_qpos,
            "command_qpos": command,
            "command_source": command_source,
            "logged_command_qpos": logged,
            "raw_command_qpos": raw,
            "repaired_command_qpos": repaired,
            "deprecated_command_qpos": bad,
            "consumed_fields": [
                "action_phase",
                "operation_start_index",
                "operation_start_robot_qpos",
                "operation_start_object_qpos",
                command_source.split("[")[0],
            ],
        }


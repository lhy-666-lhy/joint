"""Single-segment joint-goal planning without dataset-specific inputs."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np

from force_admittance_collect.curobo_grasp import (
    CuroboGraspConfig,
    _import_curobo_motion_gen,
    _joint_state_batch,
    _make_motion_gen,
    _result_success_array,
)


@dataclass(frozen=True)
class JointGoalPlan:
    success: bool
    path: np.ndarray
    reason: str
    start_max_error: float
    terminal_max_error: float


def _position_array(value: object) -> np.ndarray:
    position = value.position if hasattr(value, "position") else value
    if hasattr(position, "detach"):
        position = position.detach().to("cpu").numpy()
    return np.asarray(position, dtype=np.float64)


def extract_trajopt_paths(result: object, batch_size: int) -> np.ndarray:
    """Extract [batch, time, 7] paths from a CuRobo TrajOptResult."""
    plan = getattr(result, "interpolated_solution", None)
    if plan is None:
        plan = getattr(result, "solution", None)
    if plan is None:
        return np.zeros((0, 0, 7), dtype=np.float64)
    paths = _position_array(plan)
    if paths.ndim == 2 and batch_size == 1:
        paths = paths[None]
    if paths.ndim != 3 or paths.shape[0] != batch_size or paths.shape[2] < 7:
        return np.zeros((0, 0, 7), dtype=np.float64)
    return paths[:, :, :7].copy()


def validate_joint_goal_inputs(
    start_qpos: np.ndarray, goal_qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    goals = np.asarray(goal_qpos, dtype=np.float32)
    if goals.ndim != 2 or goals.shape[1] != 7 or goals.shape[0] < 1:
        raise ValueError(f"goal qpos must have shape [N, 7], got {goals.shape}")
    starts = np.asarray(start_qpos, dtype=np.float32)
    if starts.shape == (7,):
        starts = np.broadcast_to(starts, goals.shape).copy()
    if starts.shape != goals.shape:
        raise ValueError(f"start/goal shape mismatch: {starts.shape}/{goals.shape}")
    if not np.isfinite(starts).all() or not np.isfinite(goals).all():
        raise ValueError("joint-goal inputs must be finite")
    return starts, goals


def plan_joint_goals_batch(
    start_qpos: np.ndarray,
    goal_qpos: np.ndarray,
    config: CuroboGraspConfig,
    *,
    terminal_tolerance: float = 1e-3,
) -> list[JointGoalPlan]:
    starts, goals = validate_joint_goal_inputs(start_qpos, goal_qpos)
    tolerance = float(terminal_tolerance)
    if tolerance <= 0.0:
        raise ValueError("terminal tolerance must be positive")
    torch, _, _, _, _, _, JointState, _, _, _ = _import_curobo_motion_gen()
    from curobo.rollout.rollout_base import Goal

    motion_gen = _make_motion_gen(config)
    try:
        result = motion_gen.js_trajopt_solver.solve_batch(
            Goal(
                current_state=_joint_state_batch(starts, JointState, config.device),
                goal_state=_joint_state_batch(goals, JointState, config.device),
            ),
            num_seeds=max(1, int(config.num_seeds)),
            return_all_solutions=False,
        )
        success = _result_success_array(result)
        paths = extract_trajopt_paths(result, goals.shape[0])
        outputs: list[JointGoalPlan] = []
        for index in range(goals.shape[0]):
            solver_success = index < success.size and bool(success[index])
            path = (
                np.asarray(paths[index], dtype=np.float32)
                if index < paths.shape[0]
                else np.zeros((0, 7), dtype=np.float32)
            )
            finite = bool(
                path.ndim == 2
                and path.shape[0] >= 2
                and path.shape[1] == 7
                and np.isfinite(path).all()
            )
            start_error = (
                float(np.max(np.abs(path[0] - starts[index])))
                if finite
                else float("inf")
            )
            terminal_error = (
                float(np.max(np.abs(path[-1] - goals[index])))
                if finite
                else float("inf")
            )
            contract_success = bool(
                solver_success
                and finite
                and start_error <= tolerance
                and terminal_error <= tolerance
            )
            outputs.append(
                JointGoalPlan(
                    success=contract_success,
                    path=path if finite else np.zeros((0, 7), dtype=np.float32),
                    reason=(
                        ""
                        if contract_success
                        else "solver_or_terminal_contract_failed"
                    ),
                    start_max_error=start_error,
                    terminal_max_error=terminal_error,
                )
            )
        return outputs
    finally:
        del motion_gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

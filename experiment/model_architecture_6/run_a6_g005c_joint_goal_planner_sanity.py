#!/usr/bin/env python3
"""Run a qpose-only current-state joint planner sanity on eight CAL targets."""

from __future__ import annotations

import inspect
import ast
import hashlib
import json
import os
import sys
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_joint_goal_planner import plan_joint_goals_batch  # noqa: E402
from a6_grasp_path_utils import resample_joint_path  # noqa: E402
from force_admittance_collect.curobo_grasp import CuroboGraspConfig  # noqa: E402
from path_config import (  # noqa: E402
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G005C_SANITY_RESULT_ROOT,
)
RUN_ID = "a6_g005c_joint_goal_planner_sanity_v7"
TARGET_COUNT = 8
TERMINAL_TOLERANCE = 1e-3
FORBIDDEN_SOURCE_TOKENS = (
    "grasp_plan_qpath",
    "pregrasp",
    "traj_labels",
    "traj_teacher_manifest",
    "trajectory_relative_path",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_read_string_literals(source: str) -> set[str]:
    """Return string literals used by explicit file-read calls in source."""
    tree = ast.parse(textwrap.dedent(source))
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in {"open", "load", "read_text", "read_bytes"}:
            continue
        read_expression = [node.func, *node.args]
        for expression in read_expression:
            for child in ast.walk(expression):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    values.add(child.value.lower())
    return values


def first_distinct_cal_groups(groups: list[dict], count: int) -> list[dict]:
    selected: list[dict] = []
    targets: set[str] = set()
    for group in groups:
        target = str(group["target"])
        if group["split"] != "A5_CAL" or target in targets:
            continue
        if int(group["selected_candidate_count"]) < 1:
            raise ValueError(f"CAL group has no qpose teacher: {group['sample_id']}")
        selected.append(group)
        targets.add(target)
        if len(selected) == count:
            break
    return selected


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_G005C_SANITY_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = now()
    running = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "running",
        "complete": False,
        "terminal": False,
        "resource_mode": "gpu",
        "pid": os.getpid(),
        "started_at": started_at,
    }
    atomic_json(out_dir / "run_state.json", running)
    atomic_json(
        out_dir / "queue_state.json",
        {**running, "jobs": [{"id": "A6-G005C-SANITY", "status": "running"}]},
    )
    atomic_json(
        out_dir / "command.json",
        {"argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    )
    try:
        source_root = Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)
        g000 = json.loads((source_root / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (source_root / "qpose_teacher_manifest.json").read_text(encoding="utf-8")
        )
        with np.load(source_root / "qpose_labels.npz", allow_pickle=False) as labels:
            initial_arm = np.asarray(labels["initial_arm_qpos"], dtype=np.float32)
            qpose = np.asarray(labels["qpose_absolute"], dtype=np.float32)
            presence = np.asarray(labels["presence"], dtype=bool)
        selected = first_distinct_cal_groups(manifest["groups"], TARGET_COUNT)
        indices = np.asarray([int(row["group_index"]) for row in selected], dtype=np.int64)
        starts = initial_arm[indices]
        candidate_count = qpose.shape[1]
        valid_mask = presence[indices]

        config = CuroboGraspConfig(device="cuda:0", num_seeds=8, num_trajopt_seeds=8)
        planner_started = time.perf_counter()
        plans_grid: list[list[object | None]] = [
            [None for _ in range(candidate_count)] for _ in selected
        ]
        for group_index in range(len(selected)):
            for candidate_index in range(candidate_count):
                if not valid_mask[group_index, candidate_index]:
                    continue
                plans_grid[group_index][candidate_index] = plan_joint_goals_batch(
                    starts[group_index],
                    qpose[indices[group_index], candidate_index][None],
                    config,
                    terminal_tolerance=TERMINAL_TOLERANCE,
                )[0]
        planner_seconds = time.perf_counter() - planner_started
        plans = [
            plan
            for group in plans_grid
            for plan in group
            if plan is not None
        ]
        max_waypoints = max((plan.path.shape[0] for plan in plans), default=0)
        group_count = len(selected)
        raw_paths = np.zeros((group_count, candidate_count, max_waypoints, 7), dtype=np.float32)
        raw_valid = np.zeros((group_count, candidate_count, max_waypoints), dtype=bool)
        l64_paths = np.zeros((group_count, candidate_count, 64, 7), dtype=np.float32)
        success = np.zeros((group_count, candidate_count), dtype=bool)
        terminal_error = np.full((group_count, candidate_count), np.inf, dtype=np.float32)
        path_length = np.full((group_count, candidate_count), np.inf, dtype=np.float32)
        selected_candidate = np.full((group_count,), -1, dtype=np.int64)
        rows = []
        for group_index, group in enumerate(selected):
            candidates = []
            for candidate_index in range(candidate_count):
                if not valid_mask[group_index, candidate_index]:
                    candidates.append({"candidate_index": candidate_index, "planner_success": False, "reason": "presence_mask_false"})
                    continue
                plan = plans_grid[group_index][candidate_index]
                success[group_index, candidate_index] = plan.success
                terminal_error[group_index, candidate_index] = plan.terminal_max_error
                if plan.success:
                    path_length[group_index, candidate_index] = float(np.linalg.norm(np.diff(plan.path, axis=0), axis=1).sum())
                if plan.path.shape[0]:
                    raw_paths[group_index, candidate_index, : plan.path.shape[0]] = plan.path
                    raw_valid[group_index, candidate_index, : plan.path.shape[0]] = True
                if plan.success:
                    l64_paths[group_index, candidate_index] = resample_joint_path(plan.path)
                    l64_paths[group_index, candidate_index, 0] = starts[group_index]
                    l64_paths[group_index, candidate_index, -1] = qpose[indices[group_index], candidate_index]
                candidates.append({
                    "candidate_index": candidate_index,
                    "teacher_id": str(group["candidates"][candidate_index]["teacher_id"]),
                    "planner_success": bool(plan.success),
                    "reason": plan.reason,
                    "raw_waypoints": int(plan.path.shape[0]),
                    "start_max_error": float(plan.start_max_error),
                    "terminal_max_error": float(plan.terminal_max_error),
                    "joint_space_length": float(path_length[group_index, candidate_index]),
                })
            eligible = [row for row in candidates if row["planner_success"]]
            if eligible:
                selected_candidate[group_index] = min(
                    eligible, key=lambda row: (row["joint_space_length"], row["candidate_index"])
                )["candidate_index"]
            rows.append({
                "target_index": group_index,
                "group_index": int(group["group_index"]),
                "group_id": str(group["group_id"]),
                "sample_id": str(group["sample_id"]),
                "target": str(group["target"]),
                "selected_candidate": int(selected_candidate[group_index]),
                "candidates": candidates,
            })
        atomic_npz(
            out_dir / "planned_paths.npz",
            {
                "group_index": indices,
                "start_qpos": starts,
                "goal_qpos": qpose[indices],
                "success": success,
                "terminal_error": terminal_error,
                "path_length": path_length,
                "selected_candidate": selected_candidate,
                "raw_path": raw_paths,
                "raw_valid": raw_valid,
                "path_l64": l64_paths,
            },
        )
        atomic_json(out_dir / "attempts.json", {"rows": rows})

        planner_read_literals = file_read_string_literals(
            inspect.getsource(plan_joint_goals_batch)
        )
        runner_read_literals = file_read_string_literals(inspect.getsource(main))
        read_literals = planner_read_literals | runner_read_literals
        forbidden_hits = sorted(
            token
            for token in FORBIDDEN_SOURCE_TOKENS
            if any(token in value for value in read_literals)
        )
        successful = [
            candidate
            for row in rows
            for candidate in row["candidates"]
            if candidate.get("planner_success")
        ]
        checks = {
            "g000c_terminal_pass": g000.get("status") == "passed"
            and all(bool(value) for value in g000.get("checks", {}).values()),
            "eight_distinct_cal_targets": len(rows) == TARGET_COUNT
            and len({row["target"] for row in rows}) == TARGET_COUNT,
            "qpose_sidecar_only": not forbidden_hits,
            "all_attempts_terminal": len(plans) == int(valid_mask.sum()),
            "successful_paths_finite_t_by_7": all(
                plan.path.ndim == 2
                and plan.path.shape[0] >= 2
                and plan.path.shape[1] == 7
                and np.isfinite(plan.path).all()
                for plan in plans
                if plan.success
            ),
            "successful_terminal_contract": all(
                row["start_max_error"] <= TERMINAL_TOLERANCE
                and row["terminal_max_error"] <= TERMINAL_TOLERANCE
                for row in successful
            ),
            "successful_l64_endpoint_exact": all(
                np.array_equal(l64_paths[group_index, candidate_index, 0], starts[group_index])
                and np.array_equal(
                    l64_paths[group_index, candidate_index, -1],
                    qpose[indices[group_index], candidate_index],
                )
                for group_index, row in enumerate(rows)
                for candidate_index, candidate in enumerate(row["candidates"])
                if candidate.get("planner_success")
            ),
            "finite_metrics": bool(np.isfinite(planner_seconds)),
        }
        forbidden_audit = {
            "planner_input_fields": ["current_arm_qpos", "open_terminal_qpose"],
            "loaded_sidecars": ["qpose_labels.npz", "qpose_teacher_manifest.json"],
            "file_read_string_literals": sorted(read_literals),
            "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
            "forbidden_source_hits": forbidden_hits,
            "stored_path_or_pregrasp_read": False,
            "outcome_read": False,
        }
        atomic_json(out_dir / "forbidden_feature_audit.json", forbidden_audit)
        passed = all(checks.values())
        metrics = {
            "targets": len(rows),
            "candidates": int(valid_mask.sum()),
            "planner_success": int(success.sum()),
            "planner_coverage": float(success[valid_mask].mean()),
            "groups_with_successful_candidate": int(np.count_nonzero(success.any(axis=1))),
            "route_level_selected_groups": int(np.count_nonzero(selected_candidate >= 0)),
            "planner_seconds": planner_seconds,
            "successful_waypoints": [row["raw_waypoints"] for row in successful],
            "max_start_error": max(
                (row["start_max_error"] for row in successful), default=None
            ),
            "max_terminal_error": max(
                (row["terminal_max_error"] for row in successful), default=None
            ),
        }
        summary = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "status": "passed" if passed else "failed",
            "complete": True,
            "terminal": True,
            "failure_class": None if passed else "implementation_failure",
            "scientific_scope": "qpose-only current-state single-segment planner candidate-set sanity",
            "metrics": metrics,
            "checks": checks,
            "source_hashes": {
                "g000_summary": sha256_file(source_root / "summary.json"),
                "qpose_labels": sha256_file(source_root / "qpose_labels.npz"),
                "qpose_manifest": sha256_file(source_root / "qpose_teacher_manifest.json"),
            },
            "decision": (
                "authorize candidate-set GT-TRAJ/GT-QPOSE physical pilot"
                if passed and int(np.count_nonzero(selected_candidate >= 0)) == len(rows)
                else "diagnose qpose-only planner before physical rollout"
            ),
            "started_at": started_at,
            "completed_at": now(),
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(
            out_dir / "queue_state.json",
            {**summary, "jobs": [{"id": "A6-G005C-SANITY", "status": summary["status"]}]},
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if passed else 2
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "status": "failed",
            "complete": True,
            "terminal": True,
            "failure_class": "implementation_crash",
            "error": f"{type(exc).__name__}: {exc}",
            "decision": "repair G005C sanity without changing qpose interface",
            "started_at": started_at,
            "completed_at": now(),
        }
        atomic_json(out_dir / "summary.json", failure)
        atomic_json(out_dir / "run_state.json", failure)
        atomic_json(
            out_dir / "queue_state.json",
            {**failure, "jobs": [{"id": "A6-G005C-SANITY", "status": "failed"}]},
        )
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

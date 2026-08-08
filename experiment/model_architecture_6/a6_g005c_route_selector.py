"""Outcome-blind route selection and aggregation for the G005C qpose ceiling."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import numpy as np


def select_route_rows(rows: list[dict], group_indices: list[int]) -> dict[str, dict]:
    """Select using planner outputs only; physical outcomes are never ranking inputs."""
    selected = {}
    for group_index in group_indices:
        candidates = [row for row in rows if int(row["group_index"]) == group_index]
        if not candidates:
            raise ValueError(f"group {group_index} has no successful planner candidate")
        row = min(
            candidates,
            key=lambda item: (float(item["joint_space_length"]), int(item["candidate_index"])),
        )
        selected[str(group_index)] = {
            "group_id": str(row["group_id"]),
            "candidate_index": int(row["candidate_index"]),
            "planner_success": True,
            "joint_space_length": float(row["joint_space_length"]),
            "selection_key": [float(row["joint_space_length"]), int(row["candidate_index"])],
            "strict_grasp_pass": bool(row["grasp"]["strict_grasp_pass"]),
            "task_success": bool(row["operation"]["task_success"]),
            "progress": float(row["operation"]["final_progress"]),
            "contact_fraction": float(row["operation"]["contact_fraction"]),
        }
    return selected


def selected_indices(selected: dict[str, dict]) -> dict[str, int]:
    return {key: int(value["candidate_index"]) for key, value in selected.items()}


def aggregate_summary(
    rows: list[dict],
    groups: list[dict],
    read_literals: set[str],
    *,
    run_id: str,
    forbidden_tokens: tuple[str, ...],
) -> dict:
    group_indices = [int(group["group_index"]) for group in groups]
    selected = select_route_rows(rows, group_indices)

    outcome_mutated = copy.deepcopy(rows)
    for row in outcome_mutated:
        row["grasp"]["strict_grasp_pass"] = not bool(row["grasp"]["strict_grasp_pass"])
        row["operation"]["task_success"] = not bool(row["operation"]["task_success"])
        row["operation"]["final_progress"] = -1234.0
        row["operation"]["contact_fraction"] = -1234.0
    selector_outcome_invariant = selected_indices(selected) == selected_indices(
        select_route_rows(outcome_mutated, group_indices)
    )

    forbidden_hits = sorted(
        token for token in forbidden_tokens if any(token in value for value in read_literals)
    )
    selected_rows = list(selected.values())
    checks = {
        "fresh_world_each": all(row["fresh_world"] for row in rows),
        "l64_each": all(row["qpath_shape"] == [64, 7] for row in rows),
        "fixed_stage_ticks": all(
            row["grasp"]["hold_open_steps"] == 30
            and row["grasp"]["close_steps"] == 80
            and row["grasp"]["settle_steps"] == 120
            for row in rows
        ),
        "fixed_operation_budget": all(row["operation"]["calls"] <= 650 for row in rows),
        "qpose_consumer_forbidden_reads_absent": not forbidden_hits,
        "all_groups_selected": len(selected) == len(group_indices),
        "selector_outcome_invariant": selector_outcome_invariant,
    }
    status = "passed" if all(checks.values()) else "failed"
    return {
        "schema_version": 2,
        "run_id": run_id,
        "status": status,
        "complete": True,
        "terminal": True,
        "scientific_scope": "GT-QPOSE K4 candidate physical ceiling",
        "groups": len(groups),
        "candidates": len(rows),
        "candidate_strict_grasp_success": sum(bool(row["grasp"]["strict_grasp_pass"]) for row in rows),
        "candidate_task_success": sum(bool(row["operation"]["task_success"]) for row in rows),
        "route_level_strict_grasp_success": sum(row["strict_grasp_pass"] for row in selected_rows),
        "route_level_task_success": sum(row["task_success"] for row in selected_rows),
        "route_level_mean_progress": float(np.mean([row["progress"] for row in selected_rows])),
        "route_level_mean_contact_fraction": float(np.mean([row["contact_fraction"] for row in selected_rows])),
        "route_level_selection": "minimum (joint_space_length, candidate_index); no outcome read",
        "selector_input_fields": ["group_index", "joint_space_length", "candidate_index"],
        "selected": selected,
        "checks": checks,
        "file_read_string_literals": sorted(read_literals),
        "forbidden_read_hits": forbidden_hits,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

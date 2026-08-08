#!/usr/bin/env python3
"""Physical candidate-level screen for replay-pass GT trajectory teachers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_grasp_operation_pilot import load_operation_policy, run_physical_episode
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G005C_GT_TRAJ_CANDIDATE_PILOT_RESULT_ROOT,
)
from run_a6_g005c_joint_goal_planner_sanity import first_distinct_cal_groups


RUN_ID = "a6_g005c_gt_traj_candidate_physical_pilot_v2"


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-candidates", type=int, default=0)
    args = parser.parse_args()
    out = Path(JOINTTRAIN_ARCH6_G005C_GT_TRAJ_CANDIDATE_PILOT_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)
    manifest = json.loads((source / "traj_teacher_manifest.json").read_text())
    groups = first_distinct_cal_groups(manifest["groups"], args.targets)
    with np.load(source / "traj_labels.npz", allow_pickle=False) as labels:
        paths = np.asarray(labels["path_absolute"], dtype=np.float32)
        presence = np.asarray(labels["presence"], dtype=bool)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, std, checkpoint = load_operation_policy(device)
    rows_path = out / "rows.json"
    rows = (
        json.loads(rows_path.read_text())["rows"]
        if args.resume and rows_path.is_file()
        else []
    )
    completed_keys = {(int(row["group_index"]), int(row["candidate_index"])) for row in rows}
    total_candidates = sum(int(presence[int(group["group_index"])].sum()) for group in groups)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    running = {"schema_version": 1, "run_id": RUN_ID, "status": "running", "complete": False, "terminal": False, "pid": os.getpid(), "targets": len(groups), "total_candidates": total_candidates, "completed_candidates": len(rows), "started_at": started_at}
    atomic_json(out / "command.json", {"argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    atomic_json(out / "run_state.json", running)
    atomic_json(out / "queue_state.json", {**running, "jobs": [{"id": "GT-TRAJ-CANDIDATES", "status": "running"}]})
    new_candidates = 0
    stop_after_candidate = False
    for group in groups:
        index = int(group["group_index"])
        init_path = Path(ARTICU_COLLECTION_ROOT) / "data" / "single" / str(group["sample_id"]) / "initial_state.json"
        init = json.loads(init_path.read_text())
        for candidate_index in range(paths.shape[1]):
            if not presence[index, candidate_index]:
                continue
            if (index, candidate_index) in completed_keys:
                continue
            path = paths[index, candidate_index]
            path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            row = run_physical_episode(
                route="gt_traj_candidate",
                group_index=index,
                group_id=str(group["group_id"]),
                sample_id=str(group["sample_id"]),
                target=str(group["target"]),
                init=init,
                qpath=path,
                model=model,
                std=std,
                device=device,
            )
            row.update({"candidate_index": candidate_index, "joint_space_length": path_length})
            rows.append(row)
            new_candidates += 1
            atomic_json(out / "rows.json", {"rows": rows})
            atomic_json(out / "run_state.json", {**running, "completed_candidates": len(rows), "last_group_index": index, "last_candidate_index": candidate_index, "heartbeat_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if args.max_new_candidates > 0 and new_candidates >= args.max_new_candidates:
                stop_after_candidate = True
                break
        if stop_after_candidate:
            break
    if len(rows) < total_candidates:
        partial = {
            **running,
            "status": "partial",
            "completed_candidates": len(rows),
            "remaining_candidates": total_candidates - len(rows),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        atomic_json(out / "run_state.json", partial)
        atomic_json(out / "queue_state.json", {**partial, "jobs": [{"id": "GT-TRAJ-CANDIDATES", "status": "partial"}]})
        print(json.dumps(partial))
        return 0
    selected = {}
    for group in groups:
        candidates = [row for row in rows if row["group_index"] == int(group["group_index"])]
        selected_row = min(candidates, key=lambda row: (row["joint_space_length"], row["candidate_index"]))
        selected[str(group["group_index"])] = {
            "candidate_index": int(selected_row["candidate_index"]),
            "strict_grasp_pass": bool(selected_row["grasp"]["strict_grasp_pass"]),
            "task_success": bool(selected_row["operation"]["task_success"]),
            "progress": float(selected_row["operation"]["final_progress"]),
        }
    summary = {
        "schema_version": 1, "run_id": RUN_ID, "status": "passed", "complete": True, "terminal": True,
        "scientific_scope": "GT-TRAJ replay-pass K4 candidate physical ceiling",
        "groups": len(groups), "candidates": len(rows),
        "candidate_strict_grasp_success": sum(bool(row["grasp"]["strict_grasp_pass"]) for row in rows),
        "candidate_task_success": sum(bool(row["operation"]["task_success"]) for row in rows),
        "route_level_selection": "shortest joint-space path, slot tie-break; no outcome read",
        "selected": selected,
        "checks": {
            "fresh_world_each": all(row["fresh_world"] for row in rows),
            "l64_each": all(row["qpath_shape"] == [64, 7] for row in rows),
            "fixed_stage_ticks": all(row["grasp"]["hold_open_steps"] == 30 and row["grasp"]["close_steps"] == 80 and row["grasp"]["settle_steps"] == 120 for row in rows),
            "fixed_operation_budget": all(row["operation"]["calls"] <= 650 for row in rows),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "GT-TRAJ-CANDIDATES", "status": "passed"}]})
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

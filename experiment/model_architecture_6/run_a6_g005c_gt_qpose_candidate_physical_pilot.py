#!/usr/bin/env python3
"""Physical candidate-level screen for GT qpose plus online planner."""

from __future__ import annotations

import argparse
import inspect
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

from a6_g005c_route_selector import aggregate_summary
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G005C_GT_QPOSE_CANDIDATE_PILOT_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G005C_SANITY_RESULT_ROOT,
)
from run_a6_g005c_joint_goal_planner_sanity import first_distinct_cal_groups, file_read_string_literals


RUN_ID = "a6_g005c_gt_qpose_candidate_physical_pilot_v2"
FORBIDDEN = ("traj_labels", "traj_teacher_manifest", "grasp_plan_qpath", "pregrasp")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-candidates", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    out = Path(JOINTTRAIN_ARCH6_G005C_GT_QPOSE_CANDIDATE_PILOT_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)
    planner = Path(JOINTTRAIN_ARCH6_G005C_SANITY_RESULT_ROOT)
    manifest = json.loads((source / "qpose_teacher_manifest.json").read_text())
    groups = first_distinct_cal_groups(manifest["groups"], args.targets)
    with np.load(planner / "planned_paths.npz", allow_pickle=False) as labels:
        indices = np.asarray(labels["group_index"], dtype=np.int64)
        success = np.asarray(labels["success"], dtype=bool)
        paths = np.asarray(labels["path_l64"], dtype=np.float32)
        path_length = np.asarray(labels["path_length"], dtype=np.float32)
    rows_path = out / "rows.json"
    rows = (
        json.loads(rows_path.read_text())["rows"]
        if args.resume and rows_path.is_file()
        else []
    )
    if args.aggregate_only:
        if not rows_path.is_file():
            raise FileNotFoundError(f"missing completed rows: {rows_path}")
        read_literals = file_read_string_literals(inspect.getsource(main))
        summary = aggregate_summary(
            rows, groups, read_literals, run_id=RUN_ID, forbidden_tokens=FORBIDDEN
        )
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "run_state.json", summary)
        atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "GT-QPOSE-CANDIDATES", "status": summary["status"]}]})
        print(json.dumps(summary))
        return 0 if summary["status"] == "passed" else 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from a6_grasp_operation_pilot import load_operation_policy, run_physical_episode

    model, std, _ = load_operation_policy(device)
    completed_keys = {(int(row["group_index"]), int(row["candidate_index"])) for row in rows}
    total_candidates = int(success[: len(groups)].sum())
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    running = {"schema_version": 1, "run_id": RUN_ID, "status": "running", "complete": False, "terminal": False, "pid": os.getpid(), "targets": len(groups), "total_candidates": total_candidates, "completed_candidates": len(rows), "started_at": started_at}
    atomic_json(out / "command.json", {"argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    atomic_json(out / "run_state.json", running)
    atomic_json(out / "queue_state.json", {**running, "jobs": [{"id": "GT-QPOSE-CANDIDATES", "status": "running"}]})
    new_candidates = 0
    stop_after_candidate = False
    for group in groups:
        group_index = int(group["group_index"])
        local_index = int(np.flatnonzero(indices == group_index)[0])
        init_path = Path(ARTICU_COLLECTION_ROOT) / "data" / "single" / str(group["sample_id"]) / "initial_state.json"
        init = json.loads(init_path.read_text())
        for candidate_index in range(paths.shape[1]):
            if not success[local_index, candidate_index]:
                continue
            if (group_index, candidate_index) in completed_keys:
                continue
            row = run_physical_episode(
                route="gt_qpose_candidate",
                group_index=group_index,
                group_id=str(group["group_id"]),
                sample_id=str(group["sample_id"]),
                target=str(group["target"]),
                init=init,
                qpath=paths[local_index, candidate_index],
                model=model,
                std=std,
                device=device,
            )
            row.update({"candidate_index": candidate_index, "joint_space_length": float(path_length[local_index, candidate_index])})
            rows.append(row)
            new_candidates += 1
            atomic_json(out / "rows.json", {"rows": rows})
            atomic_json(out / "run_state.json", {**running, "completed_candidates": len(rows), "last_group_index": group_index, "last_candidate_index": candidate_index, "heartbeat_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
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
        atomic_json(out / "queue_state.json", {**partial, "jobs": [{"id": "GT-QPOSE-CANDIDATES", "status": "partial"}]})
        print(json.dumps(partial))
        return 0
    read_literals = file_read_string_literals(inspect.getsource(main))
    summary = aggregate_summary(
        rows, groups, read_literals, run_id=RUN_ID, forbidden_tokens=FORBIDDEN
    )
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "GT-QPOSE-CANDIDATES", "status": summary["status"]}]})
    print(json.dumps(summary))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

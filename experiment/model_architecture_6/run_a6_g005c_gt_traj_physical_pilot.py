#!/usr/bin/env python3
"""Execute GT trajectory labels for the G005C physical interface pilot."""

from __future__ import annotations

import argparse
import hashlib
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
    JOINTTRAIN_ARCH6_G005C_GT_TRAJ_RAW_DIAGNOSTIC_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G005C_GT_TRAJ_PILOT_RESULT_ROOT,
)
from run_a6_g005c_joint_goal_planner_sanity import first_distinct_cal_groups


RUN_ID = "a6_g005c_gt_traj_physical_pilot_v2"


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--path-mode", choices=("l64", "raw"), default="l64")
    args = parser.parse_args()
    out = Path(
        JOINTTRAIN_ARCH6_G005C_GT_TRAJ_PILOT_RESULT_ROOT
        if args.path_mode == "l64"
        else JOINTTRAIN_ARCH6_G005C_GT_TRAJ_RAW_DIAGNOSTIC_RESULT_ROOT
    )
    out.mkdir(parents=True, exist_ok=True)
    source = Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)
    manifest_path = source / "traj_teacher_manifest.json"
    label_path = source / "traj_labels.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = first_distinct_cal_groups(manifest["groups"], args.targets)
    with np.load(label_path, allow_pickle=False) as labels:
        paths = np.asarray(labels["path_absolute"], dtype=np.float32)
        presence = np.asarray(labels["presence"], dtype=bool)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, std, checkpoint = load_operation_policy(device)
    rows = []
    for group in groups:
        index = int(group["group_index"])
        if not presence[index, 0]:
            raise ValueError(f"missing GT trajectory for group {index}")
        init_path = (
            Path(ARTICU_COLLECTION_ROOT)
            / "data"
            / "single"
            / str(group["sample_id"])
            / "initial_state.json"
        )
        init = json.loads(init_path.read_text(encoding="utf-8"))
        qpath = paths[index, 0]
        if args.path_mode == "raw":
            raw_source_path = (
                Path(ARTICU_COLLECTION_ROOT)
                / str(group["candidates"][0]["trajectory_relative_path"])
            )
            with np.load(raw_source_path, allow_pickle=False) as raw_source:
                qpath = np.asarray(raw_source["grasp_plan_qpath"], dtype=np.float32)
        rows.append(
            run_physical_episode(
                route="gt_traj" if args.path_mode == "l64" else "gt_traj_raw",
                group_index=index,
                group_id=str(group["group_id"]),
                sample_id=str(group["sample_id"]),
                target=str(group["target"]),
                init=init,
                qpath=qpath,
                model=model,
                std=std,
                device=device,
            )
        )
        atomic_json(out / "rows.json", {"rows": rows})
    summary = {
        "schema_version": 1,
        "run_id": (
            RUN_ID
            if args.path_mode == "l64"
            else "a6_g005c_gt_traj_raw_diagnostic_v1"
        ),
        "status": "passed",
        "complete": True,
        "terminal": True,
        "scientific_scope": (
            "GT trajectory physical interface pilot"
            if args.path_mode == "l64"
            else "raw-qpath diagnostic for L64 interface failure"
        ),
        "route": "gt_traj" if args.path_mode == "l64" else "gt_traj_raw",
        "targets": len(rows),
        "strict_grasp_success": sum(row["grasp"]["strict_grasp_pass"] for row in rows),
        "task_success": sum(row["operation"]["task_success"] for row in rows),
        "mean_progress": float(np.mean([row["operation"]["final_progress"] for row in rows])),
        "checks": {
            "fresh_world_each": all(row["fresh_world"] for row in rows),
            "path_shape_valid": all(
                len(row["qpath_shape"]) == 2
                and row["qpath_shape"][0] >= 2
                and row["qpath_shape"][1] == 7
                for row in rows
            ),
            "l64_each": (
                all(row["qpath_shape"] == [64, 7] for row in rows)
                if args.path_mode == "l64"
                else None
            ),
            "fixed_stage_ticks": all(
                row["grasp"]["hold_open_steps"] == 30
                and row["grasp"]["close_steps"] == 80
                and row["grasp"]["settle_steps"] == 120
                for row in rows
            ),
            "fixed_operation_budget": all(row["operation"]["calls"] <= 650 for row in rows),
        },
        "source_hashes": {
            "traj_labels": sha256(label_path),
            "traj_manifest": sha256(manifest_path),
            "operation_checkpoint": sha256(checkpoint),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

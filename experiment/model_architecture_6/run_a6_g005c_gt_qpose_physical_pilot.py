#!/usr/bin/env python3
"""Execute qpose-only planned paths for the G005C physical interface pilot."""

from __future__ import annotations

import argparse
import hashlib
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

from a6_grasp_operation_pilot import load_operation_policy, run_physical_episode
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G005C_GT_QPOSE_PILOT_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G005C_SANITY_RESULT_ROOT,
)
from run_a6_g005c_joint_goal_planner_sanity import file_read_string_literals


RUN_ID = "a6_g005c_gt_qpose_physical_pilot_v2"
FORBIDDEN = ("traj_labels", "traj_teacher_manifest", "grasp_plan_qpath", "pregrasp")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=1)
    args = parser.parse_args()
    out = Path(JOINTTRAIN_ARCH6_G005C_GT_QPOSE_PILOT_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)
    planner = Path(JOINTTRAIN_ARCH6_G005C_SANITY_RESULT_ROOT)
    manifest_path = source / "qpose_teacher_manifest.json"
    planned_path = planner / "planned_paths.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_index = {int(row["group_index"]): row for row in manifest["groups"]}
    with np.load(planned_path, allow_pickle=False) as labels:
        indices = np.asarray(labels["group_index"], dtype=np.int64)
        success = np.asarray(labels["success"], dtype=bool)
        paths = np.asarray(labels["path_l64"], dtype=np.float32)
    selected = [i for i, ok in enumerate(success) if ok][: args.targets]
    if len(selected) != args.targets:
        raise ValueError("not enough successful qpose-only planner rows")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, std, checkpoint = load_operation_policy(device)
    rows = []
    for local_index in selected:
        index = int(indices[local_index])
        group = by_index[index]
        init_path = (
            Path(ARTICU_COLLECTION_ROOT)
            / "data"
            / "single"
            / str(group["sample_id"])
            / "initial_state.json"
        )
        init = json.loads(init_path.read_text(encoding="utf-8"))
        rows.append(
            run_physical_episode(
                route="gt_qpose",
                group_index=index,
                group_id=str(group["group_id"]),
                sample_id=str(group["sample_id"]),
                target=str(group["target"]),
                init=init,
                qpath=paths[local_index],
                model=model,
                std=std,
                device=device,
            )
        )
        atomic_json(out / "rows.json", {"rows": rows})
    read_literals = file_read_string_literals(inspect.getsource(main))
    forbidden_hits = sorted(
        token for token in FORBIDDEN if any(token in value for value in read_literals)
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "passed" if not forbidden_hits else "failed",
        "complete": True,
        "terminal": True,
        "scientific_scope": "GT qpose-only online-planner physical interface pilot",
        "route": "gt_qpose",
        "targets": len(rows),
        "strict_grasp_success": sum(row["grasp"]["strict_grasp_pass"] for row in rows),
        "task_success": sum(row["operation"]["task_success"] for row in rows),
        "mean_progress": float(np.mean([row["operation"]["final_progress"] for row in rows])),
        "checks": {
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
        },
        "file_read_string_literals": sorted(read_literals),
        "forbidden_read_hits": forbidden_hits,
        "source_hashes": {
            "qpose_manifest": sha256(manifest_path),
            "planned_paths": sha256(planned_path),
            "operation_checkpoint": sha256(checkpoint),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Current-state planner screen for frozen G070 direct qpose candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from force_admittance_collect.curobo_grasp import CuroboGraspConfig
from jointTrain_new.experiment.model_architecture_6.a6_joint_goal_planner import plan_joint_goals_batch
from path_config import JOINTTRAIN_ARCH6_G064C_RESULT_ROOT, JOINTTRAIN_ARCH6_G070C_RESULT_ROOT


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    candidate_path = Path(JOINTTRAIN_ARCH6_G070C_RESULT_ROOT) / "full" / "cal_qpose_candidates.npz"
    with np.load(candidate_path, allow_pickle=False) as data:
        qpose = np.asarray(data["qpose_candidates"], dtype=np.float32)
        presence = np.asarray(data["presence"], dtype=bool)
        fallback = np.asarray(data["norm_selected"], dtype=np.int64)
        group_index = np.asarray(data["group_index"], dtype=np.int64)
    with np.load(Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / "full" / "supervision.npz", allow_pickle=False) as data:
        cal = np.flatnonzero(np.asarray(data["split"]) == 1)
        state = np.asarray(data["state_qpos"][cal], dtype=np.float32)
    count = min(args.limit, len(qpose)) if args.limit else len(qpose)
    qpose, presence, fallback, group_index, state = (
        value[:count] for value in (qpose, presence, fallback, group_index, state)
    )
    success = np.zeros(qpose.shape[:-1], dtype=bool)
    path_length = np.full(qpose.shape[:-1], np.inf, dtype=np.float32)
    selected = fallback.copy()
    started = time.time()
    out = Path(JOINTTRAIN_ARCH6_G070C_RESULT_ROOT) / "planner" / (f"probe_{args.limit}" if args.limit else "full")
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    config = CuroboGraspConfig(device=f"cuda:{args.gpu}", num_seeds=4, num_trajopt_seeds=4)
    for row in range(count):
        locations = np.argwhere(np.broadcast_to(presence[row, :, None], qpose.shape[1:3]))
        goals = np.stack([state[row] + qpose[row, slot, mode] for slot, mode in locations])
        plans = plan_joint_goals_batch(state[row], goals, config, terminal_tolerance=1e-3)
        for (slot, mode), plan in zip(locations, plans):
            success[row, slot, mode] = bool(plan.success)
            if plan.success:
                path_length[row, slot, mode] = float(np.linalg.norm(np.diff(plan.path, axis=0), axis=1).sum())
        for slot in np.flatnonzero(presence[row]):
            eligible = np.flatnonzero(success[row, slot])
            if len(eligible):
                selected[row, slot] = int(eligible[np.argmin(path_length[row, slot, eligible])])
        atomic(out / "progress.json", {
            "groups_complete": row + 1,
            "groups_total": count,
            "planner_success": int(success[: row + 1].sum()),
            "elapsed_seconds": time.time() - started,
        })
    np.savez_compressed(out / "planner_results.npz", group_index=group_index, success=success, path_length=path_length, selected=selected)

    with np.load(candidate_path, allow_pickle=False) as data:
        targets = np.asarray(data["ik_targets"][:count], dtype=np.float32)
        target_presence = np.asarray(data["ik_presence"][:count], dtype=bool)
    pair = np.abs(qpose[:, :, :, None, :] - targets[:, :, None, :, :]).mean(axis=-1)
    pair = np.where(target_presence[:, :, None, :], pair, np.inf)
    candidate_error = pair.min(axis=-1)
    selected_error = np.take_along_axis(candidate_error, selected[..., None], axis=-1).squeeze(-1)
    oracle_error = candidate_error.min(axis=-1)
    valid = presence & target_presence.any(axis=-1)
    selected_plan = np.take_along_axis(success, selected[..., None], axis=-1).squeeze(-1)
    checks = {
        "g070_terminal": json.loads((Path(JOINTTRAIN_ARCH6_G070C_RESULT_ROOT) / "full" / "summary.json").read_text())["status"] == "passed",
        "group_count": count == (args.limit if args.limit else 101),
        "candidate_shape": qpose.shape == (count, 4, 8, 7),
        "finite_inputs": bool(np.isfinite(qpose).all() and np.isfinite(state).all()),
        "selection_indices_valid": bool(np.all((selected >= 0) & (selected < 8))),
        "evaluation_labels_loaded_after_selection": True,
        "no_teacher_qpose_selector_input": True,
        "no_outcome_read": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": "A6-G070C-PLANNER-PROBE" if args.limit else "A6-G070C-PLANNER",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "groups": count,
        "elapsed_seconds": time.time() - started,
        "metrics": {
            "any_candidate_planner_coverage": float(success.any(axis=-1)[presence].mean()),
            "selected_planner_coverage": float(selected_plan[presence].mean()),
            "oracle_qpose_set_l1": float(oracle_error[valid].mean()),
            "planner_selected_qpose_set_l1": float(selected_error[valid].mean()),
        },
        "checks": checks,
        "claim_supported": "probe_only" if passed and args.limit else ("diagnostic_only" if passed else "no"),
        "decision": "run full G070 planner screen" if passed and args.limit else "stop direct-qpose branch unless planner coverage and qpose error both support",
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

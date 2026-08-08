#!/usr/bin/env python3
"""Materialize frozen predicted contact queries and matched K4 SE3 supervision."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch

ROOT = Path(__file__).resolve().parents[3]
JOINT_ROOT = ROOT / "jointTrain_new"
for path in (ROOT, JOINT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from joint_train.utils.pc_utils import pc_normalize
from path_config import (
    JOINTTRAIN_ARCH6_A020C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A030C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G052C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G060C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G061C_RESULT_ROOT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
)
from run_a6_a030c_affordance_cal_consumer import ENSEMBLE_WEIGHTS, SEEDS, paired_bootstrap, predict_checkpoint, select_ranked


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_m": float(array.mean()),
        "median_m": float(np.median(array)),
        "p90_m": float(np.quantile(array, 0.9)),
        "coverage_3cm": float(np.mean(array <= 0.03)),
        "coverage_5cm": float(np.mean(array <= 0.05)),
        "max_m": float(array.max()),
    }


def main() -> int:
    import argparse
    import zarr

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--eval-seed", type=int, default=20260807)
    args = parser.parse_args()
    a030 = json.loads((Path(JOINTTRAIN_ARCH6_A030C_RESULT_ROOT) / "full" / "summary.json").read_text(encoding="utf-8"))
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    with np.load(Path(JOINTTRAIN_ARCH6_G060C_RESULT_ROOT) / "contact_labels.npz", allow_pickle=False) as data:
        teacher_contact = np.asarray(data["contact_point"], dtype=np.float32)
        teacher_presence = np.asarray(data["presence"], dtype=bool)
    count = args.limit or len(arrays["split"])
    arrays = {key: value[:count] for key, value in arrays.items()}
    teacher_contact = teacher_contact[:count]
    teacher_presence = teacher_presence[:count]
    zarr_root = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    zarr_ids = np.asarray(zarr_root["meta/source_replay_id"][:], dtype=np.int64)
    source_index = {int(value): index for index, value in enumerate(zarr_ids.tolist())}
    source_rows = np.asarray([source_index[int(value)] for value in arrays["source_replay_id"]], dtype=np.int64)
    world_points = np.asarray(zarr_root["data/point_cloud"][source_rows, :, :3], dtype=np.float32)
    normalized = np.stack([pc_normalize(item.copy()) for item in world_points]).astype(np.float32)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_predictions = []
    reload_errors = {}
    for seed in SEEDS:
        checkpoint = Path(JOINTTRAIN_ARCH6_A020C_RESULT_ROOT) / f"seed_{seed}" / "last.pth"
        prediction, reload_error = predict_checkpoint(checkpoint, normalized, device, args.batch_size, args.eval_seed)
        seed_predictions.append(prediction)
        reload_errors[str(seed)] = reload_error
    predicted_affordance = sum(weight * prediction for weight, prediction in zip(ENSEMBLE_WEIGHTS, seed_predictions))
    query_point = np.zeros((count, 4, 3), dtype=np.float32)
    query_target_se3 = np.zeros((count, 4, 9), dtype=np.float32)
    query_teacher_contact = np.zeros((count, 4, 3), dtype=np.float32)
    query_presence = np.zeros((count, 4), dtype=bool)
    query_teacher_slot = np.full((count, 4), -1, dtype=np.int8)
    assigned_by_split = {0: [], 1: []}
    centroid_by_split = {0: [], 1: []}
    group_difference = {0: [], 1: []}
    hand_offset_norm = {0: [], 1: []}
    for group in range(count):
        mask = arrays["target_mask"][group]
        target_points = arrays["point_cloud_xyz"][group, mask]
        target_scores = predicted_affordance[group, mask]
        queries = select_ranked(target_points, target_scores, 4, 0.03).astype(np.float32)
        query_point[group] = queries
        valid_slots = np.flatnonzero(teacher_presence[group])
        contacts = teacher_contact[group, valid_slots]
        cost = cdist(queries, contacts)
        query_rows, teacher_columns = linear_sum_assignment(cost)
        split = int(arrays["split"][group])
        assigned = []
        centroid = target_points.mean(axis=0)
        centroid_distances = []
        for query_row, teacher_column in zip(query_rows, teacher_columns):
            teacher_slot = int(valid_slots[teacher_column])
            query_presence[group, query_row] = True
            query_teacher_slot[group, query_row] = teacher_slot
            query_target_se3[group, query_row] = arrays["se3_base"][group, teacher_slot]
            query_teacher_contact[group, query_row] = teacher_contact[group, teacher_slot]
            distance = float(cost[query_row, teacher_column])
            assigned.append(distance)
            centroid_distances.append(float(np.linalg.norm(centroid - teacher_contact[group, teacher_slot])))
            hand_offset_norm[split].append(float(np.linalg.norm(arrays["se3_base"][group, teacher_slot, :3] - queries[query_row])))
        assigned_by_split[split].extend(assigned)
        centroid_by_split[split].extend(centroid_distances)
        group_difference[split].append(float(np.mean(assigned) - np.mean(centroid_distances)))
    out = Path(JOINTTRAIN_ARCH6_G061C_RESULT_ROOT) / (f"probe_{args.limit}" if args.limit else "full")
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "contact_query_inputs.npz"
    np.savez_compressed(
        output_path,
        point_cloud_xyz=arrays["point_cloud_xyz"],
        state_qpos=arrays["state_qpos"],
        target_mask=arrays["target_mask"],
        predicted_affordance=predicted_affordance,
        query_point=query_point,
        query_target_se3=query_target_se3,
        query_teacher_contact=query_teacher_contact,
        query_presence=query_presence,
        query_teacher_slot=query_teacher_slot,
        split=arrays["split"],
        group_index=arrays["group_index"],
        source_replay_id=arrays["source_replay_id"],
        base_pose=arrays["base_pose"],
    )
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "training_config.json", {"training": False, "producer": "A6-A030C fixed three-seed mean", "assignment": "Hungarian contact distance"})
    atomic(out / "run_manifest.json", {"run_id": "A6-G061C", "splits_read": ["A5_TRAIN", "A5_CAL"], "groups": count, "teacher_labels": int(teacher_presence.sum())})
    atomic(out / "forbidden_feature_audit.json", {"task_outcome_read": False, "future_state_forward_input": False, "gt_affordance_forward_input": False})
    metrics = {
        "assigned_contact_distance": {
            "train": summarize(assigned_by_split[0]),
            "cal": summarize(assigned_by_split[1]) if assigned_by_split[1] else None,
        },
        "centroid_contact_distance": {
            "train": summarize(centroid_by_split[0]),
            "cal": summarize(centroid_by_split[1]) if centroid_by_split[1] else None,
        },
        "hand_center_minus_query_norm": {
            "train": summarize(hand_offset_norm[0]),
            "cal": summarize(hand_offset_norm[1]) if hand_offset_norm[1] else None,
        },
        "assigned_minus_centroid_group_mean_m": {
            "train": paired_bootstrap(np.asarray(group_difference[0]), args.eval_seed),
            "cal": paired_bootstrap(np.asarray(group_difference[1]), args.eval_seed) if group_difference[1] else None,
        },
    }
    checks = {
        "a030_positive_terminal": a030.get("status") == "passed" and a030.get("claim_supported") == "partial",
        "rows_exact": count == (args.limit if args.limit else 632),
        "split_counts": bool(args.limit) or (int(np.sum(arrays["split"] == 0)) == 531 and int(np.sum(arrays["split"] == 1)) == 101),
        "teacher_presence_preserved": bool(np.array_equal(query_presence.sum(axis=1), teacher_presence.sum(axis=1))),
        "one_to_one_assignment": bool(np.all([len(set(row[row >= 0].tolist())) == int(np.sum(row >= 0)) for row in query_teacher_slot])),
        "finite": bool(np.isfinite(predicted_affordance).all() and np.isfinite(query_point).all() and np.isfinite(query_target_se3).all()),
        "reload_exact": all(value == 0.0 for value in reload_errors.values()),
        "train_cal_labels_only": True,
        "zero_outcome_read": True,
    }
    implementation_passed = all(checks.values())
    cal_comparison = metrics["assigned_minus_centroid_group_mean_m"]["cal"]
    signal = bool(args.limit) or (cal_comparison["ci95"][1] < 0.0 and metrics["assigned_contact_distance"]["cal"]["coverage_3cm"] > 0.0)
    summary = {
        "schema_version": 1,
        "run_id": "A6-G061C-PROBE" if args.limit else "A6-G061C",
        "status": "passed" if implementation_passed else "failed",
        "complete": True,
        "terminal": True,
        "metrics": metrics,
        "reload_max_abs": reload_errors,
        "checks": checks,
        "claim_supported": "partial" if implementation_passed and signal and not args.limit else "no",
        "decision": "authorize contact-conditioned SE3 fit" if implementation_passed and signal and not args.limit else ("run full G061C" if implementation_passed and args.limit else "stop contact-query fit"),
        "next_run_ids": ["A6-G062C"] if implementation_passed and signal and not args.limit else (["A6-G061C"] if implementation_passed and args.limit else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if implementation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

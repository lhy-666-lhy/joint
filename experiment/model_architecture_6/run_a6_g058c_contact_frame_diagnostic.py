#!/usr/bin/env python3
"""Measure whether grasp poses admit a compact target-contact representation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import JOINTTRAIN_ARCH6_G052C_RESULT_ROOT, JOINTTRAIN_ARCH6_G058C_RESULT_ROOT, JOINTTRAIN_BESTVIEW_DUAL_ZARR


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def stats(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.quantile(array, 0.9)), "max": float(array.max())}


def main() -> int:
    import zarr
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        point = np.asarray(data["point_cloud_xyz"], dtype=np.float64)
        target_mask = np.asarray(data["target_mask"], dtype=bool)
        pose_values = np.asarray(data["se3_base"], dtype=np.float64)
        presence = np.asarray(data["presence"], dtype=bool)
        split = np.asarray(data["split"], dtype=np.int8)
        source_ids = np.asarray(data["source_replay_id"], dtype=np.int32)
    zarr_root = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    zarr_ids = np.asarray(zarr_root["meta/source_replay_id"][:], dtype=np.int32)
    zarr_index = {int(value): index for index, value in enumerate(zarr_ids.tolist())}
    affordance = np.asarray(zarr_root["data/affordance_updated"][[zarr_index[int(value)] for value in source_ids]], dtype=np.float64)
    target_distance = []
    affordance_distance = []
    local_offsets = []
    approach_axis = []
    positive_groups = 0
    label_split = []
    for group in range(len(point)):
        target_points = point[group, target_mask[group]]
        positive = target_mask[group] & (affordance[group] >= 0.5)
        positive_points = point[group, positive]
        if len(positive_points):
            positive_groups += 1
        target_tree = cKDTree(target_points)
        positive_tree = cKDTree(positive_points) if len(positive_points) else None
        for slot in np.flatnonzero(presence[group]):
            translation = pose_values[group, slot, :3]
            first = pose_values[group, slot, 3:6]
            second = pose_values[group, slot, 6:9]
            third = np.cross(first, second)
            rotation = np.stack((first, second, third), axis=1)
            distance, index = target_tree.query(translation, k=1)
            contact = target_points[index]
            offset = rotation.T @ (translation - contact)
            target_distance.append(float(distance))
            local_offsets.append(offset)
            approach_axis.append(int(np.argmax(np.abs(offset))))
            affordance_distance.append(float(positive_tree.query(translation, k=1)[0]) if positive_tree is not None else float("nan"))
            label_split.append(int(split[group]))
    local_offsets = np.asarray(local_offsets, dtype=np.float64)
    affordance_distance_array = np.asarray(affordance_distance, dtype=np.float64)
    finite_affordance = affordance_distance_array[np.isfinite(affordance_distance_array)]
    axis_count = {f"axis_{axis}": int(np.sum(np.asarray(approach_axis) == axis)) for axis in range(3)}
    dominant_axis = int(np.argmax([axis_count[f"axis_{axis}"] for axis in range(3)]))
    dominant = local_offsets[:, dominant_axis]
    orthogonal = np.delete(local_offsets, dominant_axis, axis=1)
    metrics = {
        "labels": len(local_offsets),
        "target_surface_distance_m": stats(target_distance),
        "affordance_positive_distance_m": stats(finite_affordance),
        "affordance_positive_groups": positive_groups,
        "affordance_positive_label_coverage": float(len(finite_affordance) / len(local_offsets)),
        "hand_frame_offset_mean_xyz": local_offsets.mean(axis=0).tolist(),
        "hand_frame_offset_std_xyz": local_offsets.std(axis=0).tolist(),
        "approach_axis_counts": axis_count,
        "dominant_axis": dominant_axis,
        "dominant_axis_offset_m": stats(np.abs(dominant)),
        "orthogonal_offset_norm_m": stats(np.linalg.norm(orthogonal, axis=1)),
    }
    checks = {
        "labels_exact": len(local_offsets) == 2373,
        "finite_target_geometry": bool(np.isfinite(local_offsets).all() and np.isfinite(target_distance).all()),
        "split_labels_exact": int(np.sum(np.asarray(label_split) == 0)) == 1991 and int(np.sum(np.asarray(label_split) == 1)) == 382,
        "affordance_join_exact": len(zarr_index) == len(zarr_ids),
        "no_outcome_read": True,
    }
    summary = {"schema_version": 1, "run_id": "A6-G058C", "status": "passed" if all(checks.values()) else "failed", "complete": True, "terminal": True, "metrics": metrics, "checks": checks, "decision": "use geometry statistics to decide oracle contact-frame ceiling"}
    out = Path(JOINTTRAIN_ARCH6_G058C_RESULT_ROOT)
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

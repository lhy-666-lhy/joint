#!/usr/bin/env python3
"""Test whether GT affordance top-K points cover K4 teacher contacts."""

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

from path_config import JOINTTRAIN_ARCH6_G052C_RESULT_ROOT, JOINTTRAIN_ARCH6_G060C_RESULT_ROOT, JOINTTRAIN_BESTVIEW_DUAL_ZARR


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def select_ranked(points: np.ndarray, scores: np.ndarray, count: int, min_distance: float = 0.0) -> np.ndarray:
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        if not selected or min(np.linalg.norm(points[index] - points[chosen]) for chosen in selected) >= min_distance:
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.argsort(-scores, kind="stable"):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return points[selected]


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean_m": float(array.mean()), "median_m": float(np.median(array)), "p90_m": float(np.quantile(array, 0.9)), "coverage_3cm": float(np.mean(array <= 0.03)), "coverage_5cm": float(np.mean(array <= 0.05)), "max_m": float(array.max())}


def main() -> int:
    import zarr
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        point = np.asarray(data["point_cloud_xyz"], dtype=np.float64)
        target_mask = np.asarray(data["target_mask"], dtype=bool)
        pose = np.asarray(data["se3_base"], dtype=np.float64)
        presence = np.asarray(data["presence"], dtype=bool)
        split = np.asarray(data["split"], dtype=np.int8)
        source_ids = np.asarray(data["source_replay_id"], dtype=np.int32)
    zarr_root = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    ids = np.asarray(zarr_root["meta/source_replay_id"][:], dtype=np.int32)
    index = {int(value): row for row, value in enumerate(ids.tolist())}
    affordance = np.asarray(zarr_root["data/affordance_updated"][[index[int(value)] for value in source_ids]], dtype=np.float64)
    rows = {name: {0: [], 1: []} for name in ("target_centroid", "affordance_top1", "affordance_top4", "affordance_top4_nms3cm", "affordance_positive_oracle")}
    contacts = np.zeros((632, 4, 3), dtype=np.float32)
    positive_group = {0: 0, 1: 0}
    for group in range(632):
        target_points = point[group, target_mask[group]]
        target_scores = affordance[group, target_mask[group]]
        tree = cKDTree(target_points)
        selected = {
            "target_centroid": target_points.mean(axis=0, keepdims=True),
            "affordance_top1": select_ranked(target_points, target_scores, 1),
            "affordance_top4": select_ranked(target_points, target_scores, 4),
            "affordance_top4_nms3cm": select_ranked(target_points, target_scores, 4, min_distance=0.03),
        }
        positive = target_points[target_scores >= 0.5]
        if len(positive):
            positive_group[int(split[group])] += 1
            selected["affordance_positive_oracle"] = positive
        for slot in np.flatnonzero(presence[group]):
            _, contact_index = tree.query(pose[group, slot, :3], k=1)
            contact = target_points[contact_index]
            contacts[group, slot] = contact
            for name, proposals in selected.items():
                rows[name][int(split[group])].append(float(cKDTree(proposals).query(contact, k=1)[0]))
    metrics = {}
    for name, by_split in rows.items():
        metrics[name] = {"train": summarize(by_split[0]), "cal": summarize(by_split[1])}
    checks = {
        "contact_labels": int(presence.sum()) == 2373,
        "finite": bool(np.isfinite(contacts).all()),
        "split_label_counts": len(rows["affordance_top1"][0]) == 1991 and len(rows["affordance_top1"][1]) == 382,
        "affordance_join_exact": len(index) == len(ids),
        "no_outcome_read": True,
    }
    out = Path(JOINTTRAIN_ARCH6_G060C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "contact_labels.npz", contact_point=contacts, presence=presence, split=split)
    summary = {"schema_version": 1, "run_id": "A6-G060C", "status": "passed" if all(checks.values()) else "failed", "complete": True, "terminal": True, "metrics": metrics, "positive_groups": {"train": positive_group[0], "cal": positive_group[1]}, "checks": checks, "decision": "use GT affordance coverage to decide contact proposal training"}
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

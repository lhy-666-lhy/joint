"""Deterministic affordance-map interventions for causal policy diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def replay_affordance(root, replay_id: int, label_source: str) -> np.ndarray:
    source = str(label_source).lower()
    if source == "initial":
        values = root["data"]["affordance_initial"][int(replay_id)]
    elif source == "updated":
        values = root["data"].get(
            "affordance_updated", root["data"]["point_cloud"][:, :, 3]
        )[int(replay_id)]
    else:
        raise ValueError(f"unknown label_source={label_source}")
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def transfer_map(
    source_xyz: np.ndarray,
    source_values: np.ndarray,
    query_xyz: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(np.asarray(source_xyz, dtype=np.float64))
    _, indices = tree.query(np.asarray(query_xyz, dtype=np.float64), k=1)
    return np.asarray(source_values, dtype=np.float32)[indices]


def intervene_affordance(
    root,
    *,
    xyz: np.ndarray,
    correct: np.ndarray,
    obj_key: str,
    replay_id: int,
    label_source: str,
    intervention: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    kind = str(intervention).lower()
    values = np.asarray(correct, dtype=np.float32)
    metadata: dict[str, Any] = {"intervention": kind, "donor_replay_id": None}
    if kind == "correct":
        output = values.copy()
    elif kind == "zero":
        output = np.zeros_like(values)
    elif kind == "swap":
        keys = [str(item) for item in root["meta"]["replay_obj_keys"][:]]
        donors = [index for index, key in enumerate(keys) if key == str(obj_key) and index != replay_id]
        if not donors:
            raise ValueError(f"no same-target donor replay for {obj_key}/{replay_id}")
        donor_id = int(donors[0])
        donor_xyz = np.asarray(root["data"]["point_cloud"][donor_id, :, :3], dtype=np.float32)
        donor_values = replay_affordance(root, donor_id, label_source)
        output = transfer_map(donor_xyz, donor_values, xyz)
        metadata["donor_replay_id"] = donor_id
    elif kind == "shift":
        xyz_array = np.asarray(xyz, dtype=np.float32)
        extent = np.ptp(xyz_array, axis=0)
        axis = int(np.argmax(extent))
        shift = np.zeros(3, dtype=np.float32)
        shift[axis] = 0.1 * float(extent[axis])
        output = transfer_map(xyz_array, values, xyz_array - shift)
        metadata["shift_axis"] = axis
        metadata["shift_xyz"] = shift.tolist()
    else:
        raise ValueError(f"unknown intervention={intervention}")
    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
    metadata.update(
        {
            "map_mean": float(output.mean()),
            "map_l1_from_correct": float(np.mean(np.abs(output - values))),
            "map_max_from_correct": float(np.max(np.abs(output - values))),
        }
    )
    return output, metadata

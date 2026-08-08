"""Pure joint-path utilities shared by grasp label and planner consumers."""

from __future__ import annotations

import numpy as np


def resample_joint_path(qpath: np.ndarray, length: int = 64) -> np.ndarray:
    path = np.asarray(qpath, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 7:
        raise ValueError(f"invalid grasp qpath shape: {path.shape}")
    if not np.isfinite(path).all():
        raise ValueError("nonfinite grasp qpath")
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0.0:
        raise ValueError("zero-length grasp qpath")
    keep = np.concatenate(([True], np.diff(cumulative) > 0.0))
    positions = cumulative[keep]
    values = path[keep]
    samples = np.linspace(0.0, cumulative[-1], int(length), dtype=np.float64)
    result = np.stack(
        [np.interp(samples, positions, values[:, joint]) for joint in range(7)], axis=1
    )
    result[0] = path[0]
    result[-1] = path[-1]
    return result.astype(np.float32)

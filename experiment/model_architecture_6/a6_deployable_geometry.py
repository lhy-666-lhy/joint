#!/usr/bin/env python3
"""Deployable robot-FK and visible-target geometry features for A6."""

from __future__ import annotations

import numpy as np


FK_TARGET_FEATURE_DIM = 4
FK_TARGET_STATE_DIM = 81 + FK_TARGET_FEATURE_DIM
FK_TARGET_FEATURE_NAMES = (
    "hand_minus_visible_target_centroid_x",
    "hand_minus_visible_target_centroid_y",
    "hand_minus_visible_target_centroid_z",
    "visible_target_fraction",
)


def raw_fk_target_feature(
    world,
    base_pose,
    robot_qpos: np.ndarray,
    point_cloud: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(target_mask, dtype=bool).reshape(-1)
    if points.shape[0] == 0 or mask.shape[0] != points.shape[0]:
        raise ValueError("point cloud and target mask are not aligned")
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains nonfinite values")

    target_points = points[mask] if bool(mask.any()) else points
    target_centroid = target_points.mean(axis=0)
    hand_position = np.asarray(
        world.hand_pose_world(base_pose, np.asarray(robot_qpos, dtype=np.float64))[:3, 3],
        dtype=np.float64,
    )
    feature = np.concatenate(
        (hand_position - target_centroid, [float(mask.mean())]), axis=0
    ).astype(np.float32)
    if feature.shape != (FK_TARGET_FEATURE_DIM,) or not np.isfinite(feature).all():
        raise ValueError("invalid FK/target-relative feature")
    return feature


def normalize_fk_target_feature(
    feature: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(FK_TARGET_FEATURE_DIM)
    mean = np.asarray(mean, dtype=np.float32).reshape(FK_TARGET_FEATURE_DIM)
    std = np.asarray(std, dtype=np.float32).reshape(FK_TARGET_FEATURE_DIM)
    if not np.isfinite(std).all() or bool((std <= 0.0).any()):
        raise ValueError("invalid FK/target-relative normalizer")
    normalized = (feature - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("nonfinite normalized FK/target-relative feature")
    return normalized.astype(np.float32)

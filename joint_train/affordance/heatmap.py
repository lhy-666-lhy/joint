"""Volume-scaled Gaussian affordance (copied from articu_sapien heatmap logic).

sigma = sigma_coeff * AABB_volume^(1/3)
scores = max(score / max(score), 0)  # negatives clamped to 0
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIGMA_COEFF = 0.04008
DEFAULT_FIXED_SIGMA = 0.035
MIN_VOLUME_SCALED_SIGMA = 1e-4


def aabb_extent(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((3,), dtype=np.float32)
    return np.maximum(points.max(axis=0) - points.min(axis=0), 0.0).astype(np.float32)


def aabb_volume(points: np.ndarray) -> float:
    return float(np.prod(aabb_extent(points)))


def volume_cbrt(points: np.ndarray) -> float:
    return float(aabb_volume(points) ** (1.0 / 3.0))


def compute_volume_scaled_sigma(
    points: np.ndarray,
    *,
    sigma_coeff: float = DEFAULT_SIGMA_COEFF,
    min_sigma: float = MIN_VOLUME_SCALED_SIGMA,
) -> float:
    cbrt = volume_cbrt(points)
    return float(max(float(min_sigma), float(sigma_coeff) * cbrt))


def resolve_heatmap_sigma(
    points: np.ndarray,
    *,
    sigma: float | None = None,
    sigma_coeff: float = DEFAULT_SIGMA_COEFF,
    scale_sigma_by_volume_cbrt: bool = True,
) -> dict:
    extent = aabb_extent(points)
    volume = float(np.prod(extent))
    cbrt = float(volume ** (1.0 / 3.0))
    if scale_sigma_by_volume_cbrt:
        used = compute_volume_scaled_sigma(points, sigma_coeff=sigma_coeff)
    else:
        used = float(DEFAULT_FIXED_SIGMA if sigma is None else sigma)
    return {
        "sigma": float(used),
        "sigma_coeff": float(sigma_coeff),
        "scale_sigma_by_volume_cbrt": bool(scale_sigma_by_volume_cbrt),
        "aabb_extent_xyz": extent,
        "aabb_volume": float(volume),
        "aabb_volume_cbrt": float(cbrt),
    }


def _nearest_gaussian(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(centers).query(points, k=1)
    except Exception:
        diff = points[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return np.exp(-0.5 * (dist / max(1e-6, float(sigma))) ** 2).astype(np.float32)


def heatmap_scores(
    points: np.ndarray,
    centers: np.ndarray,
    success_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Nearest-Gaussian affordance with failure suppression; clamp negatives to 0."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    success_mask = np.asarray(success_mask, dtype=bool).reshape(-1)
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if centers.size == 0:
        return np.zeros((points.shape[0],), dtype=np.float32)
    success_centers = centers[success_mask]
    failure_centers = centers[~success_mask]
    if success_centers.shape[0] == 0:
        return np.zeros((points.shape[0],), dtype=np.float32)
    scores = _nearest_gaussian(points, success_centers, sigma)
    if failure_centers.shape[0] > 0:
        scores = scores - 0.35 * _nearest_gaussian(points, failure_centers, sigma)
    max_value = float(np.max(scores))
    if max_value > 1e-8:
        scores = scores / max_value
    scores = np.maximum(scores, 0.0)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)

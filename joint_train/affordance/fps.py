"""Farthest-point sampling (copied from articu_sapien export script)."""

from __future__ import annotations

import numpy as np


def fps_indices_cpu(points: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = points.shape[0]
    k = int(num_points)
    if n == 0:
        return np.zeros((k,), dtype=np.int64)
    if n <= k:
        idx = np.arange(n, dtype=np.int64)
        if n < k:
            pad = np.random.default_rng(seed).choice(n, size=k - n, replace=True)
            idx = np.concatenate([idx, pad.astype(np.int64)])
        return idx
    rng = np.random.default_rng(seed)
    selected = np.zeros(k, dtype=np.int64)
    selected[0] = int(rng.integers(0, n))
    dists = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        last = points[selected[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = int(np.argmax(dists))
    return selected


def fps_indices_torch(
    points: np.ndarray, num_points: int, seed: int = 0, device: str = "cuda:0"
) -> np.ndarray:
    import torch

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]
    k = int(num_points)
    if n == 0:
        return np.zeros((k,), dtype=np.int64)
    if n <= k:
        idx = np.arange(n, dtype=np.int64)
        if n < k:
            pad = np.random.default_rng(seed).choice(n, size=k - n, replace=True)
            idx = np.concatenate([idx, pad.astype(np.int64)])
        return idx

    x = torch.as_tensor(pts, device=device, dtype=torch.float32)
    selected = torch.empty(k, dtype=torch.long, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    selected[0] = torch.randint(0, n, (1,), generator=g, device=device).item()
    dists = torch.full((n,), 1e10, device=device, dtype=torch.float32)
    for i in range(1, k):
        last = x[selected[i - 1]]
        d = torch.sum((x - last) ** 2, dim=1)
        dists = torch.minimum(dists, d)
        selected[i] = torch.argmax(dists)
    return selected.detach().cpu().numpy().astype(np.int64)


def fps_indices(
    points: np.ndarray, num_points: int, seed: int = 0, device: str = "cpu"
) -> np.ndarray:
    if str(device).startswith("cuda"):
        return fps_indices_torch(points, num_points, seed=seed, device=device)
    return fps_indices_cpu(points, num_points, seed=seed)

"""Pure-PyTorch fallbacks for pointnet2_ops / knn_cuda when CUDA extensions are missing.

Architecture stays identical; only FPS/KNN kernels are replaced.
Import/apply this BEFORE importing Point_M2AE_* or modules.
"""

from __future__ import annotations

import sys
import types

import torch


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz: (B, N, 3) -> indices (B, npoint)"""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N), idx: (B, npoint) -> (B, C, npoint)"""
    B, C, N = features.shape
    npoint = idx.shape[1]
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(features, 2, idx_expand)


class KNN:
    def __init__(self, k: int = 16, transpose_mode: bool = True):
        self.k = int(k)
        self.transpose_mode = bool(transpose_mode)

    def __call__(self, ref: torch.Tensor, query: torch.Tensor):
        """
        Match knn_cuda.KNN with transpose_mode=True:
          ref/xyz: (B, N, 3), query/center: (B, G, 3)
          returns (dist, idx) with idx (B, G, k)
        """
        dist = torch.cdist(query, ref)  # B G N
        d, idx = torch.topk(dist, self.k, largest=False, dim=-1)
        return d, idx


def apply() -> None:
    if "pointnet2_ops" not in sys.modules:
        pn2 = types.ModuleType("pointnet2_ops")
        utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
        utils.furthest_point_sample = furthest_point_sample
        utils.gather_operation = gather_operation
        pn2.pointnet2_utils = utils
        sys.modules["pointnet2_ops"] = pn2
        sys.modules["pointnet2_ops.pointnet2_utils"] = utils

    if "knn_cuda" not in sys.modules:
        knn_mod = types.ModuleType("knn_cuda")
        knn_mod.KNN = KNN
        sys.modules["knn_cuda"] = knn_mod

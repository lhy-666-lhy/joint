"""Point cloud preprocess aligned with Point-M2AE pcd_affordance_dataset."""

from __future__ import annotations

import numpy as np
import torch


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Centroid + unit-sphere radius normalize (same as Point-M2AE)."""
    pc = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = float(np.max(np.sqrt(np.sum(pc**2, axis=1))))
    if m < 1e-6:
        m = 1.0
    return (pc / m).astype(np.float32)


def pc_normalize_torch(pc: torch.Tensor) -> torch.Tensor:
    """Batched (B,N,3) or single (N,3) normalize, matching pc_normalize."""
    squeeze = False
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
        squeeze = True
    # B,N,3
    centroid = pc.mean(dim=1, keepdim=True)
    pc = pc - centroid
    m = torch.sqrt((pc**2).sum(dim=-1)).amax(dim=1, keepdim=True).clamp_min(1e-6)
    pc = pc / m.unsqueeze(-1)
    return pc.squeeze(0) if squeeze else pc


def augment_xyz_m2ae(xyz: np.ndarray) -> np.ndarray:
    """Dataset-side augment: Z-rot ±30° + jitter σ=0.005 (Point-M2AE)."""
    angle = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    xyz = xyz @ rot.T
    xyz = xyz + np.random.normal(0.0, 0.005, xyz.shape).astype(np.float32)
    return xyz.astype(np.float32)

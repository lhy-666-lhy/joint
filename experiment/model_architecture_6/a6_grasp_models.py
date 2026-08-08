"""Matched deployable grasp proposal models for G010/G020."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "jointTrain_new"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn as nn
import torch.nn.functional as F

from jointTrain_new.experiment.model_architecture_2.architecture_models import PointCloudContextEncoder
from jointTrain_new.joint_train.models.pointnet_encoder import PointNetEncoderXYZA, StateEncoder

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first = F.normalize(value[..., :3], dim=-1, eps=1e-6)
    second_raw = value[..., 3:6]
    second = F.normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


class ConcatAffordanceEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.point = PointNetEncoderXYZA(in_channels=4, out_channels=hidden_dim)
        self.state = StateEncoder(state_dim=7, out_channels=hidden_dim, hidden=hidden_dim)
        self.null = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, xyz: torch.Tensor, state: torch.Tensor, affordance: torch.Tensor) -> torch.Tensor:
        scene = self.point(torch.cat([xyz, affordance.unsqueeze(-1)], dim=-1))
        return torch.stack([scene, self.state(state), self.null.expand(xyz.shape[0], -1)], dim=1)


class TargetMaskEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.point = PointNetEncoderXYZA(in_channels=4, out_channels=hidden_dim)
        self.state = StateEncoder(state_dim=7, out_channels=hidden_dim, hidden=hidden_dim)
        self.null = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, xyz: torch.Tensor, state: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        scene = self.point(torch.cat([xyz, target_mask.unsqueeze(-1)], dim=-1))
        return torch.stack([scene, self.state(state), self.null.expand(xyz.shape[0], -1)], dim=1)


class DualPoolTargetMaskEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.point = PointNetEncoderXYZA(in_channels=3, out_channels=hidden_dim)
        self.state = StateEncoder(state_dim=7, out_channels=hidden_dim, hidden=hidden_dim)
        self.null_target = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, xyz: torch.Tensor, state: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.point.encode_points(xyz)
        scene = self.point.final_projection(tokens.amax(dim=1))
        valid = target_mask.to(torch.bool)
        target_pooled = tokens.masked_fill(~valid.unsqueeze(-1), torch.finfo(tokens.dtype).min).amax(dim=1)
        has_target = valid.any(dim=1, keepdim=True)
        target_pooled = torch.where(has_target, target_pooled, torch.zeros_like(target_pooled))
        target = self.point.final_projection(target_pooled)
        target = torch.where(has_target, target, self.null_target.expand(xyz.shape[0], -1))
        return torch.stack([scene, target, self.state(state)], dim=1)


class TargetLocalEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scene = PointNetEncoderXYZA(in_channels=3, out_channels=hidden_dim)
        self.target = PointNetEncoderXYZA(in_channels=3, out_channels=hidden_dim)
        self.centroid = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.state = StateEncoder(state_dim=7, out_channels=hidden_dim, hidden=hidden_dim)
        self.null_target = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, xyz: torch.Tensor, state: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        scene = self.scene(xyz)
        valid = target_mask.to(torch.bool)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(xyz.dtype)
        center = (xyz * valid.unsqueeze(-1).to(xyz.dtype)).sum(dim=1) / count
        tokens = self.target.encode_points(xyz - center.unsqueeze(1))
        pooled = tokens.masked_fill(~valid.unsqueeze(-1), torch.finfo(tokens.dtype).min).amax(dim=1)
        has_target = valid.any(dim=1, keepdim=True)
        pooled = torch.where(has_target, pooled, torch.zeros_like(pooled))
        target = self.target.final_projection(pooled) + self.centroid(center)
        target = torch.where(has_target, target, self.null_target.expand(xyz.shape[0], -1))
        return torch.stack([scene, target, self.state(state)], dim=1)


class GraspProposalBase(nn.Module):
    def __init__(self, output_kind: str, hidden_dim: int = 256, candidate_count: int = 4, use_affordance: bool = False, affordance_encoding: str = "weight", use_target_mask: bool = False, target_mask_encoding: str = "concat"):
        super().__init__()
        if output_kind not in {"traj", "qpose", "se3"}:
            raise ValueError(output_kind)
        self.output_kind = output_kind
        self.candidate_count = candidate_count
        self.use_affordance = bool(use_affordance)
        self.use_target_mask = bool(use_target_mask)
        if self.use_affordance and self.use_target_mask:
            raise ValueError("affordance and target mask must be isolated experiment arms")
        self.affordance_encoding = str(affordance_encoding)
        self.target_mask_encoding = str(target_mask_encoding)
        if self.use_target_mask:
            if self.target_mask_encoding == "concat":
                self.encoder = TargetMaskEncoder(hidden_dim)
            elif self.target_mask_encoding == "dual":
                self.encoder = DualPoolTargetMaskEncoder(hidden_dim)
            elif self.target_mask_encoding == "local":
                self.encoder = TargetLocalEncoder(hidden_dim)
            else:
                raise ValueError(f"unknown target mask encoding: {self.target_mask_encoding}")
        elif self.use_affordance and self.affordance_encoding == "concat":
            self.encoder = ConcatAffordanceEncoder(hidden_dim)
        else:
            self.encoder = PointCloudContextEncoder(state_dim=7, context_dim=0, hidden_dim=hidden_dim)
        self.queries = nn.Parameter(torch.randn(1, candidate_count, hidden_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        output_dim = 64 * 7 if output_kind == "traj" else (9 if output_kind == "se3" else 7)
        self.value_head = nn.Linear(hidden_dim, output_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)

    def forward(self, point_cloud_xyz: torch.Tensor, state_qpos: torch.Tensor, affordance: torch.Tensor | None = None, *, target_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.use_affordance and affordance is None:
            raise ValueError("affordance input required")
        if self.use_target_mask and target_mask is None:
            raise ValueError("target mask input required")
        if self.use_target_mask:
            memory = self.encoder(point_cloud_xyz, state_qpos, target_mask)
        elif self.use_affordance and self.affordance_encoding == "concat":
            memory = self.encoder(point_cloud_xyz, state_qpos, affordance)
        else:
            memory = self.encoder(point_cloud_xyz, state_qpos, point_cloud_affordance=affordance if self.use_affordance else None)
        scene = memory.flatten(1).unsqueeze(1).expand(-1, self.candidate_count, -1)
        queries = self.queries.expand(point_cloud_xyz.shape[0], -1, -1)
        feature = self.decoder(torch.cat([scene, queries], dim=-1))
        values = self.value_head(feature)
        if self.output_kind == "traj":
            values = values.reshape(-1, self.candidate_count, 64, 7)
        return {"values": values, "presence_logits": self.presence_head(feature).squeeze(-1)}


class ContactConditionedSE3(nn.Module):
    """Predict one base-frame hand pose for each frozen spatial contact query."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.scene = PointNetEncoderXYZA(in_channels=4, out_channels=hidden_dim)
        self.state = StateEncoder(state_dim=7, out_channels=hidden_dim, hidden=hidden_dim)
        self.query = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.offset_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 6)
        self.presence_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(
        self,
        point_cloud_xyz: torch.Tensor,
        state_qpos: torch.Tensor,
        affordance: torch.Tensor,
        query_point: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scene = self.scene(torch.cat([point_cloud_xyz, affordance.unsqueeze(-1)], dim=-1))
        state = self.state(state_qpos)
        nearest = torch.cdist(query_point, point_cloud_xyz).argmin(dim=-1)
        query_score = affordance.gather(1, nearest)
        query = self.query(torch.cat([query_point, query_score.unsqueeze(-1)], dim=-1))
        count = query_point.shape[1]
        feature = self.decoder(
            torch.cat(
                [scene.unsqueeze(1).expand(-1, count, -1), state.unsqueeze(1).expand(-1, count, -1), query],
                dim=-1,
            )
        )
        translation = query_point + self.offset_head(feature)
        values = torch.cat([translation, self.rotation_head(feature)], dim=-1)
        return {"values": values, "presence_logits": self.presence_head(feature).squeeze(-1)}


def contact_se3_loss(output: dict[str, torch.Tensor], target: torch.Tensor, presence: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    valid = presence.to(torch.bool)
    translation = (output["values"][..., :3] - target[..., :3]).abs().mean(dim=-1)
    predicted_rotation = rotation_6d_to_matrix(output["values"][..., 3:9])
    target_rotation = rotation_6d_to_matrix(target[..., 3:9])
    relative = predicted_rotation.transpose(-1, -2) @ target_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    rotation = torch.acos(cosine)
    pose = ((translation + rotation) * valid).sum() / valid.sum().clamp_min(1)
    presence_loss = F.binary_cross_entropy_with_logits(output["presence_logits"], valid.to(output["presence_logits"].dtype))
    loss = pose + 0.1 * presence_loss
    return loss, {
        "pose_loss": float(pose.detach()),
        "presence_loss": float(presence_loss.detach()),
        "valid_fraction": float(valid.float().mean()),
    }


def _pair_cost(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "traj":
        point = (pred[:, :, None] - target[:, None]).abs().mean(dim=(-1, -2))
        endpoint = (pred[:, :, None, -1] - target[:, None, :, -1]).abs().mean(dim=-1)
        diff_pred, diff_target = pred[:, :, 1:] - pred[:, :, :-1], target[:, None, :, 1:] - target[:, None, :, :-1]
        diff = (diff_pred[:, :, None] - diff_target).abs().mean(dim=(-1, -2))
        cost = point + endpoint + 0.1 * diff
    elif kind == "se3":
        translation = (pred[:, :, None, :3] - target[:, None, :, :3]).abs().mean(dim=-1)
        pred_rotation = rotation_6d_to_matrix(pred[..., 3:9])[:, :, None]
        target_rotation = rotation_6d_to_matrix(target[..., 3:9])[:, None]
        relative = pred_rotation.transpose(-1, -2) @ target_rotation
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        cost = translation + torch.acos(cosine)
    else:
        cost = (pred[:, :, None] - target[:, None]).abs().mean(dim=-1)
    return cost.masked_fill(~valid[:, None], 0.0)


def grasp_set_loss(output: dict[str, torch.Tensor], target: torch.Tensor, presence: torch.Tensor, kind: str) -> tuple[torch.Tensor, dict[str, float]]:
    pred = output["values"]
    logits = output["presence_logits"]
    valid = presence.to(torch.bool)
    pair = _pair_cost(pred, target, valid, kind)
    candidates = []
    for perm in PERMUTATIONS:
        index = torch.as_tensor(perm, device=pred.device)
        matched = pair[:, torch.arange(4, device=pred.device), index]
        target_presence = valid[:, index].to(logits.dtype)
        presence_cost = F.binary_cross_entropy_with_logits(logits, target_presence, reduction="none")
        candidates.append((matched * valid[:, index]).sum(dim=1) / valid.sum(dim=1).clamp_min(1) + 0.1 * presence_cost.mean(dim=1))
    costs = torch.stack(candidates, dim=1)
    best = costs.min(dim=1).values
    return best.mean(), {"set_loss": float(best.mean().detach()), "valid_slots": float(valid.float().mean().detach())}

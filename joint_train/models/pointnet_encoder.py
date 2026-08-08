"""PointNet encoder for (N,4) xyz+affordance and state MLP (DP3-style)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PointNetEncoderXYZA(nn.Module):
    """MLP 4->64->128->256, max-pool, project to 64 + LayerNorm."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 64,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
    ):
        super().__init__()
        block = [64, 128, 256]
        layers: list[nn.Module] = []
        last = in_channels
        for c in block:
            layers.append(nn.Linear(last, c))
            if use_layernorm:
                layers.append(nn.LayerNorm(c))
            layers.append(nn.ReLU())
            last = c
        self.mlp = nn.Sequential(*layers)
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        else:
            self.final_projection = nn.Linear(block[-1], out_channels)

    def encode_points(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        points = self.encode_points(x)
        pooled = torch.max(points, dim=1)[0]
        return self.final_projection(pooled)


class StateEncoder(nn.Module):
    """qpos || grasp_onehot -> 64-d feature."""

    def __init__(self, state_dim: int = 11, out_channels: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ObsEncoder(nn.Module):
    """feature1 (pc) || feature2 (state) -> 128-d final feature."""

    def __init__(
        self,
        pc_in_channels: int = 4,
        state_dim: int = 11,
        feature_dim: int = 64,
    ):
        super().__init__()
        self.pc_encoder = PointNetEncoderXYZA(
            in_channels=pc_in_channels,
            out_channels=feature_dim,
            use_layernorm=True,
            final_norm="layernorm",
        )
        self.state_encoder = StateEncoder(state_dim=state_dim, out_channels=feature_dim)
        self.output_dim = feature_dim * 2

    def forward(self, point_cloud: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: (B, N, 4) or (B, T, N, 4)
        state: (B, D) or (B, T, D)
        returns: (B, 128) or (B, T, 128)
        """
        squeeze_t = False
        if point_cloud.dim() == 3:
            # (B, N, C) — single step
            f1 = self.pc_encoder(point_cloud)
            f2 = self.state_encoder(state)
            return torch.cat([f1, f2], dim=-1)

        # (B, T, N, C)
        b, t, n, c = point_cloud.shape
        pc_flat = point_cloud.reshape(b * t, n, c)
        st_flat = state.reshape(b * t, -1)
        f1 = self.pc_encoder(pc_flat)
        f2 = self.state_encoder(st_flat)
        out = torch.cat([f1, f2], dim=-1)
        return out.reshape(b, t, -1)


class ResidualAffordanceAdapter(nn.Module):
    """Pool xyz tokens with state queries and an explicit affordance prior."""

    def __init__(self, state_dim: int = 11, feature_dim: int = 64):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.state_query = nn.Linear(state_dim, feature_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        xyz: torch.Tensor,
        affordance: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if xyz.dim() != 4 or affordance.dim() != 3 or state.dim() != 3:
            raise ValueError((xyz.shape, affordance.shape, state.shape))
        b, t, n, _ = xyz.shape
        xyz_flat = xyz.reshape(b * t, n, 3)
        aff_flat = affordance.reshape(b * t, n).clamp(0.0, 1.0)
        state_flat = state.reshape(b * t, -1)

        center = xyz_flat.mean(dim=1, keepdim=True)
        radius = torch.linalg.vector_norm(xyz_flat - center, dim=-1).amax(dim=1, keepdim=True)
        xyz_normalized = (xyz_flat - center) / radius.clamp_min(1e-6).unsqueeze(-1)
        tokens = self.point_encoder(xyz_normalized)
        query = self.state_query(state_flat)
        logits = torch.einsum("bd,bnd->bn", query, tokens) / math.sqrt(tokens.shape[-1])
        logits = logits + torch.log(aff_flat + 1e-4)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bn,bnd->bd", weights, tokens)
        residual = torch.sigmoid(self.gate_logit) * self.output_projection(pooled)
        return residual.reshape(b, t, -1)


class PointTokenCrossAttentionObsEncoder(nn.Module):
    """Use robot state as a query over lightweight per-point tokens."""

    def __init__(
        self,
        pc_in_channels: int = 4,
        state_dim: int = 11,
        feature_dim: int = 64,
        token_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.point_token_encoder = nn.Sequential(
            nn.Linear(pc_in_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
        )
        self.query_projection = nn.Linear(state_dim, token_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(token_dim)
        self.pc_projection = nn.Sequential(
            nn.Linear(token_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.state_encoder = StateEncoder(state_dim=state_dim, out_channels=feature_dim)
        self.output_dim = feature_dim * 2

    def _forward_flat(self, point_cloud: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tokens = self.point_token_encoder(point_cloud)
        query = self.query_projection(state).unsqueeze(1)
        attended, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        attended = self.attention_norm(query + attended)
        attended = self.feed_forward_norm(attended + self.feed_forward(attended))
        pc_feature = self.pc_projection(attended.squeeze(1))
        state_feature = self.state_encoder(state)
        return torch.cat([pc_feature, state_feature], dim=-1)

    def forward(self, point_cloud: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if point_cloud.dim() == 3:
            return self._forward_flat(point_cloud, state)

        b, t, n, c = point_cloud.shape
        output = self._forward_flat(
            point_cloud.reshape(b * t, n, c),
            state.reshape(b * t, -1),
        )
        return output.reshape(b, t, -1)

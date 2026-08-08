#!/usr/bin/env python3
"""Shared Architecture 6 operation encoders and fixed-batch models."""

from __future__ import annotations

import torch
from torch import nn


HIDDEN_DIM = 256
STATE_DIM = 81
GEOMETRY_STATE_DIM = 85
CONTEXT_DIM = 43
ACTION_HORIZON = 32
ACTION_DIM = 9


class PointCloudContextEncoder(nn.Module):
    """Shared XYZ/mask/affordance, state, and context encoder."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
        state_dim: int = STATE_DIM,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.point_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(CONTEXT_DIM),
            nn.Linear(CONTEXT_DIM, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 256, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
        target_mask: torch.Tensor,
        affordance: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        point_input = torch.cat(
            (
                point_cloud,
                target_mask.unsqueeze(-1).to(point_cloud.dtype),
                affordance.unsqueeze(-1).to(point_cloud.dtype),
            ),
            dim=-1,
        )
        scene = self.point_encoder(point_input).amax(dim=1)
        return self.fusion(
            torch.cat(
                (scene, self.state_encoder(state), self.context_encoder(context)), dim=-1
            )
        )


class OperationMLPAbsolute(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
        state_dim: int = STATE_DIM,
    ) -> None:
        super().__init__()
        self.encoder = PointCloudContextEncoder(
            hidden_dim=hidden_dim, dropout=dropout, state_dim=state_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, ACTION_HORIZON * ACTION_DIM),
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
        target_mask: torch.Tensor,
        affordance: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(point_cloud, target_mask, affordance, state, context)
        return self.decoder(encoded).reshape(-1, ACTION_HORIZON, ACTION_DIM)


class OperationMLPGeometryResidual(OperationMLPAbsolute):
    """Preserve the 81D baseline encoder and add a zero-init geometry residual."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
        geometry_dim: int = 4,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, state_dim=STATE_DIM)
        self.geometry_dim = int(geometry_dim)
        self.geometry_projection = nn.Linear(self.geometry_dim, hidden_dim)
        nn.init.zeros_(self.geometry_projection.weight)
        nn.init.zeros_(self.geometry_projection.bias)

    def forward(
        self,
        point_cloud: torch.Tensor,
        target_mask: torch.Tensor,
        affordance: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        expected = STATE_DIM + self.geometry_dim
        if state.shape[-1] != expected:
            raise ValueError(f"geometry state width must be {expected}, got {state.shape[-1]}")
        encoded = self.encoder(
            point_cloud, target_mask, affordance, state[:, :STATE_DIM], context
        )
        encoded = encoded + self.geometry_projection(state[:, STATE_DIM:])
        return self.decoder(encoded).reshape(-1, ACTION_HORIZON, ACTION_DIM)


class OperationMLPRecoveryResidual(nn.Module):
    """Frozen baseline MLP with a zero-init recovery correction head."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.baseline = OperationMLPAbsolute(hidden_dim=hidden_dim, dropout=dropout)
        self.recovery_head = nn.Linear(hidden_dim, ACTION_HORIZON * ACTION_DIM)
        nn.init.zeros_(self.recovery_head.weight)
        nn.init.zeros_(self.recovery_head.bias)

    def forward(
        self,
        point_cloud: torch.Tensor,
        target_mask: torch.Tensor,
        affordance: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.baseline.encoder(
            point_cloud, target_mask, affordance, state, context
        )
        baseline = self.baseline.decoder(encoded)
        correction = self.recovery_head(encoded)
        return (baseline + correction).reshape(-1, ACTION_HORIZON, ACTION_DIM)


class OperationParallelAbsolute(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
        state_dim: int = STATE_DIM,
    ) -> None:
        super().__init__()
        self.encoder = PointCloudContextEncoder(
            hidden_dim=hidden_dim, dropout=dropout, state_dim=state_dim
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.queries = nn.Parameter(torch.empty(1, ACTION_HORIZON, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, ACTION_DIM)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        point_cloud: torch.Tensor,
        target_mask: torch.Tensor,
        affordance: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encoder(
            point_cloud, target_mask, affordance, state, context
        ).unsqueeze(1)
        queries = self.queries.expand(point_cloud.shape[0], -1, -1)
        return self.action_head(self.decoder(queries, memory))


class OperationCausalAbsolute(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
        state_dim: int = STATE_DIM,
    ) -> None:
        super().__init__()
        self.encoder = PointCloudContextEncoder(
            hidden_dim=hidden_dim, dropout=dropout, state_dim=state_dim
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.action_embedding = nn.Linear(ACTION_DIM, hidden_dim)
        self.position = nn.Parameter(torch.empty(1, ACTION_HORIZON, hidden_dim))
        self.bos = nn.Parameter(torch.zeros(1, 1, ACTION_DIM))
        self.action_head = nn.Linear(hidden_dim, ACTION_DIM)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def _decode_tokens(
        self, previous_actions: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        length = previous_actions.shape[1]
        tokens = self.action_embedding(previous_actions) + self.position[:, :length]
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=tokens.device), diagonal=1
        )
        return self.action_head(self.decoder(tokens, memory, tgt_mask=causal_mask))

    def forward(
        self,
        point_cloud: torch.Tensor,
        target_mask: torch.Tensor,
        affordance: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
        teacher_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encoder(
            point_cloud, target_mask, affordance, state, context
        ).unsqueeze(1)
        bos = self.bos.expand(point_cloud.shape[0], -1, -1)
        if teacher_actions is not None:
            previous = torch.cat((bos, teacher_actions[:, :-1]), dim=1)
            return self._decode_tokens(previous, memory)
        generated: list[torch.Tensor] = []
        previous = bos
        for _ in range(ACTION_HORIZON):
            prediction = self._decode_tokens(previous, memory)[:, -1:]
            generated.append(prediction)
            previous = torch.cat((previous, prediction), dim=1)
        return torch.cat(generated, dim=1)

"""Joint policy: Point-M2AE affordance + PointNet/state encoder + DP3 diffusion head."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vendor.dp3.conditional_unet1d import ConditionalUnet1D
from vendor.dp3.mask_generator import LowdimMaskGenerator
from vendor.dp3.normalizer import LinearNormalizer
from vendor.point_m2ae.Point_M2AE_Afford import Point_M2AE_Afford

from joint_train.utils.pc_utils import pc_normalize_torch

from .pointnet_encoder import (
    ObsEncoder,
    PointTokenCrossAttentionObsEncoder,
    ResidualAffordanceAdapter,
)


class JointDiffusionPolicy(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int = 9,
        state_dim: int = 11,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        encoder_output_dim: int = 64,
        diffusion_step_embed_dim: int = 128,
        down_dims=(512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        obs_as_global_cond: bool = True,
        use_ddim: bool = True,
        normalize_pc_for_affordance: bool = True,
        reuse_static_point_feature: bool = False,
        condition_mode: str = "affordance",
        obs_encoder_variant: str = "pointnet",
        affordance_adapter: str = "none",
        affordance_aux_weight: float = 0.0,
        contact_condition: str = "none",
        dual_head: bool = False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.normalize_pc_for_affordance = bool(normalize_pc_for_affordance)
        self.reuse_static_point_feature = bool(reuse_static_point_feature)
        self.condition_mode = str(condition_mode).lower()
        if self.condition_mode not in {"affordance", "no_map"}:
            raise ValueError(f"unknown condition_mode={condition_mode}")
        self.obs_encoder_variant = str(obs_encoder_variant).lower()
        if self.obs_encoder_variant not in {"pointnet", "token_cross_attention"}:
            raise ValueError(f"unknown obs_encoder_variant={obs_encoder_variant}")
        if self.reuse_static_point_feature and self.obs_encoder_variant != "pointnet":
            raise ValueError("reuse_static_point_feature is only supported by pointnet")
        self.affordance_adapter_mode = str(affordance_adapter).lower()
        if self.affordance_adapter_mode not in {"none", "residual"}:
            raise ValueError(f"unknown affordance_adapter={affordance_adapter}")
        self.contact_condition = str(contact_condition).lower()
        if self.contact_condition not in {"none", "coordinate"}:
            raise ValueError(f"unknown contact_condition={contact_condition}")
        if self.affordance_adapter_mode == "residual":
            if self.obs_encoder_variant != "pointnet":
                raise ValueError("residual affordance adapter requires pointnet")
            if self.contact_condition != "none":
                raise ValueError("residual affordance adapter cannot be combined with contact conditioning")
        self.encoder_output_dim = int(encoder_output_dim)
        self.affordance_aux_weight = float(affordance_aux_weight)
        if self.affordance_aux_weight < 0.0:
            raise ValueError("affordance_aux_weight must be non-negative")
        if self.affordance_aux_weight > 0.0 and self.obs_encoder_variant != "pointnet":
            raise ValueError("affordance auxiliary supervision requires pointnet")
        if self.affordance_aux_weight > 0.0:
            if self.condition_mode != "no_map":
                raise ValueError("affordance auxiliary supervision requires no_map conditioning")
            if self.affordance_adapter_mode != "none" or self.contact_condition != "none":
                raise ValueError(
                    "affordance auxiliary supervision cannot be combined with adapter/contact conditioning"
                )
        self.dual_head = bool(dual_head)

        self.affordance_net = Point_M2AE_Afford(
            cls_dim=1, num_categories=16, dual_head=self.dual_head
        )
        if self.obs_encoder_variant == "pointnet":
            self.obs_encoder = ObsEncoder(
                pc_in_channels=4,
                state_dim=state_dim,
                feature_dim=encoder_output_dim,
            )
        else:
            self.obs_encoder = PointTokenCrossAttentionObsEncoder(
                pc_in_channels=4,
                state_dim=state_dim,
                feature_dim=encoder_output_dim,
            )
        obs_feature_dim = self.obs_encoder.output_dim  # 128
        self.affordance_adapter = None
        if self.affordance_adapter_mode == "residual":
            self.affordance_adapter = ResidualAffordanceAdapter(
                state_dim=state_dim,
                feature_dim=encoder_output_dim,
            )
        self.contact_encoder = None
        if self.contact_condition == "coordinate":
            self.contact_encoder = nn.Sequential(
                nn.Linear(6, encoder_output_dim),
                nn.LayerNorm(encoder_output_dim),
                nn.ReLU(),
                nn.Linear(encoder_output_dim, encoder_output_dim),
                nn.LayerNorm(encoder_output_dim),
            )

        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        else:
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
        )
        # Initialize this after the action path so enabling the training-only
        # head does not change PointNet/DP3 initialization for a fixed seed.
        self.affordance_aux_head = None
        if self.affordance_aux_weight > 0.0:
            self.affordance_aux_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        if use_ddim:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="sample",
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="sample",
            )
        self.num_inference_steps = int(num_inference_steps)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

    # ----- affordance -----
    def predict_affordance(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3) or (B, 3, N)
        returns: (B, N) scores in [0,1]

        When ``normalize_pc_for_affordance`` is True (default), applies the same
        pc_normalize as Point-M2AE / Stage-1 training before the backbone.
        """
        if xyz.dim() != 3:
            raise ValueError(f"expected 3D tensor, got {xyz.shape}")
        if xyz.shape[-1] == 3:
            xyz_bn3 = xyz
        elif xyz.shape[1] == 3:
            xyz_bn3 = xyz.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"bad xyz shape {xyz.shape}")
        if self.normalize_pc_for_affordance:
            xyz_bn3 = pc_normalize_torch(xyz_bn3)
        pts = xyz_bn3.transpose(1, 2).contiguous()  # B,3,N
        out = self.affordance_net(pts)  # B,N,1
        return out.squeeze(-1)

    def xyz_to_pc4(
        self,
        xyz: torch.Tensor,
        *,
        gt_affordance: Optional[torch.Tensor] = None,
        use_gt: bool = False,
    ) -> torch.Tensor:
        """Build (B,N,4) from xyz (+ optional GT affordance).

        PointNet consumes raw ``xyz`` (robot/world frame) concatenated with
        affordance. Affordance itself is predicted on normalized xyz when enabled.
        """
        if xyz.shape[-1] != 3:
            raise ValueError(xyz.shape)
        if use_gt:
            if gt_affordance is None:
                raise ValueError("use_gt=True requires gt_affordance")
            aff = gt_affordance
            if aff.dim() == 3 and aff.shape[-1] == 1:
                aff = aff.squeeze(-1)
        else:
            aff = self.predict_affordance(xyz)
        if self.condition_mode == "no_map":
            aff = torch.zeros_like(aff)
        return torch.cat([xyz, aff.unsqueeze(-1)], dim=-1)

    def encode_obs(
        self,
        point_cloud_4: torch.Tensor,
        state: torch.Tensor,
        contact_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        point_cloud_4: (B, To, N, 4) — To == n_obs_steps (same PCD repeated if static)
        state: (B, To, state_dim)
        returns global_cond: (B, To * 128)
        """
        baseline_cloud = point_cloud_4
        if self.affordance_adapter_mode == "residual":
            baseline_cloud = point_cloud_4.clone()
            baseline_cloud[..., 3] = 0.0
        if self.reuse_static_point_feature:
            # Stage-2 observes one static cloud at every observation step.
            pc_feat = self.obs_encoder.pc_encoder(baseline_cloud[:, 0])
            state_feat = self.obs_encoder.state_encoder(state)
            feat = torch.cat(
                [pc_feat.unsqueeze(1).expand(-1, state.shape[1], -1), state_feat], dim=-1
            )
        else:
            feat = self.obs_encoder(baseline_cloud, state)  # B, To, 128
        if self.affordance_adapter_mode == "residual":
            if self.affordance_adapter is None:
                raise RuntimeError("residual affordance adapter is not initialized")
            adapter_feature = self.affordance_adapter(
                point_cloud_4[..., :3], point_cloud_4[..., 3], state
            )
            point_feature = feat[..., : self.encoder_output_dim] + adapter_feature
            state_feature = feat[..., self.encoder_output_dim :]
            feat = torch.cat([point_feature, state_feature], dim=-1)
        if self.contact_condition == "coordinate":
            if contact_token is None or self.contact_encoder is None:
                raise ValueError("coordinate contact conditioning requires contact_token")
            contact_feature = self.contact_encoder(contact_token)
            point_feature = feat[..., : self.encoder_output_dim]
            state_feature = feat[..., self.encoder_output_dim :]
            point_feature = point_feature + contact_feature.unsqueeze(1)
            feat = torch.cat([point_feature, state_feature], dim=-1)
        return feat.reshape(feat.shape[0], -1)

    def build_contact_token(
        self, xyz: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if self.contact_condition == "none":
            return None
        required = ("contact_xyz_world", "contact_visible_5cm", "contact_valid")
        if any(key not in batch for key in required):
            raise ValueError(f"contact-conditioned batch requires {required}")
        contact = batch["contact_xyz_world"].reshape(-1, 3)
        visible = batch["contact_visible_5cm"].reshape(-1, 1).float()
        valid = batch["contact_valid"].reshape(-1, 1).float()
        center = xyz.mean(dim=1)
        radius = torch.linalg.vector_norm(xyz - center.unsqueeze(1), dim=-1).amax(dim=1, keepdim=True)
        relative = (contact - center) / radius.clamp_min(1e-6)
        relative = relative * valid
        missing = 1.0 - valid
        return torch.cat([relative, visible, valid, missing], dim=-1)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch: Dict[str, torch.Tensor], use_gt_affordance: bool = False):
        """DDP-friendly entry: same as compute_loss."""
        return self.compute_loss(batch, use_gt_affordance=use_gt_affordance)

    def compute_affordance_auxiliary_loss(
        self, point_cloud_4: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.affordance_aux_head is None:
            raise RuntimeError("affordance auxiliary head is not initialized")
        if self.obs_encoder_variant != "pointnet":
            raise RuntimeError("affordance auxiliary head requires pointnet")
        point_tokens = self.obs_encoder.pc_encoder.encode_points(point_cloud_4)
        logits = self.affordance_aux_head(point_tokens).squeeze(-1)
        target = target.reshape_as(logits).clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits)
        intersection = (probability * target).sum(dim=1)
        denominator = probability.sum(dim=1) + target.sum(dim=1)
        dice = (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        return bce + dice

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        use_gt_affordance: bool = False,
        include_auxiliary: bool = True,
    ):
        """
        batch keys:
          point_cloud_xyz: (B, N, 3) shared replay cloud (static over horizon)
          affordance_gt:   (B, N) optional
          state:  (B, horizon, state_dim)  — full sequence; uses first n_obs_steps
          action: (B, horizon, action_dim)
        """
        nobs = self.normalizer.normalize(
            {
                "state": batch["state"],
                "action": batch["action"],
            }
        )
        state = nobs["state"]
        action = nobs["action"]
        B = action.shape[0]

        xyz = batch["point_cloud_xyz"]
        gt = batch.get("affordance_gt", None)
        # stage1 freeze: no grad through affordance; stage2 unfreeze: grad flows
        pc4 = self.xyz_to_pc4(xyz, gt_affordance=gt, use_gt=use_gt_affordance)  # B,N,4

        To = self.n_obs_steps
        pc4_t = pc4.unsqueeze(1).expand(B, To, -1, -1).contiguous()
        state_t = state[:, :To]
        contact_token = self.build_contact_token(xyz, batch)
        global_cond = self.encode_obs(pc4_t, state_t, contact_token=contact_token)

        trajectory = action
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=trajectory.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy, timesteps, local_cond=None, global_cond=global_cond)
        target = trajectory
        loss_mask = ~condition_mask
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.float()
        diffusion_loss = loss.mean()
        if (
            include_auxiliary
            and self.affordance_aux_weight > 0.0
            and self.affordance_aux_head is not None
        ):
            if gt is None:
                raise ValueError("affordance auxiliary supervision requires affordance_gt")
            auxiliary_loss = self.compute_affordance_auxiliary_loss(pc4, gt)
            return diffusion_loss + self.affordance_aux_weight * auxiliary_loss
        return diffusion_loss

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor], *, use_gt_affordance: bool = False):
        state = self.normalizer["state"].normalize(batch["state"])
        B = state.shape[0]
        xyz = batch["point_cloud_xyz"]
        gt = batch.get("affordance_gt", None)
        pc4 = self.xyz_to_pc4(xyz, gt_affordance=gt, use_gt=use_gt_affordance)
        To = self.n_obs_steps
        pc4_t = pc4.unsqueeze(1).expand(B, To, -1, -1).contiguous()
        state_t = state[:, :To]
        contact_token = self.build_contact_token(xyz, batch)
        global_cond = self.encode_obs(pc4_t, state_t, contact_token=contact_token)

        shape = (B, self.horizon, self.action_dim)
        cond_data = torch.zeros(shape, device=state.device, dtype=state.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        trajectory = torch.randn(size=shape, device=state.device, dtype=state.dtype)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            trajectory[cond_mask] = cond_data[cond_mask]
            model_out = self.model(trajectory, t, local_cond=None, global_cond=global_cond)
            trajectory = self.noise_scheduler.step(model_out, t, trajectory).prev_sample
        trajectory[cond_mask] = cond_data[cond_mask]
        action_pred = self.normalizer["action"].unnormalize(trajectory)
        start = To - 1
        end = start + self.n_action_steps
        return {"action": action_pred[:, start:end], "action_pred": action_pred}

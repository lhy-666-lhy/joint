"""Point-M2AE segmentation backbone for per-point affordance regression.

Same structure as Point_M2AE_SEG (including 16-d category conditioning).
Only the prediction head differs:
  - 1-channel output
  - sigmoid -> [0, 1]
  - loss: MSE or PeakFocused (from 3DAffordanceNet)
"""

from __future__ import annotations

from . import bootstrap  # noqa: F401  — apply ops_fallback if CUDA ops missing

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .modules import Encoder_Block, Group, Token_Embed
from .pointnet2_utils import PointNetFeaturePropagation_
from .utils.logger import print_log


class H_Encoder_seg(nn.Module):
    """Copied from Point_M2AE_SEG to avoid importing chamfer CUDA extension."""

    def __init__(
        self,
        encoder_depths=None,
        num_heads=6,
        encoder_dims=None,
        local_radius=None,
    ):
        super().__init__()
        if encoder_depths is None:
            encoder_depths = [5, 5, 5]
        if encoder_dims is None:
            encoder_dims = [96, 192, 384]
        if local_radius is None:
            local_radius = [0.32, 0.64, 1.28]

        self.encoder_depths = encoder_depths
        self.encoder_num_heads = num_heads
        self.encoder_dims = encoder_dims
        self.local_radius = local_radius

        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(
                    Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i])
                )
            self.encoder_pos_embeds.append(
                nn.Sequential(
                    nn.Linear(3, self.encoder_dims[i]),
                    nn.GELU(),
                    nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                )
            )

        self.encoder_blocks = nn.ModuleList()
        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(
                Encoder_Block(
                    embed_dim=self.encoder_dims[i],
                    depth=self.encoder_depths[i],
                    drop_path_rate=dpr[depth_count : depth_count + self.encoder_depths[i]],
                    num_heads=self.encoder_num_heads,
                )
            )
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList(
            [nn.LayerNorm(self.encoder_dims[i]) for i in range(len(self.encoder_depths))]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        del eval
        x_vis_list = []
        xyz_dist = None
        x_vis = None
        for i in range(len(centers)):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(centers[i], self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius
            else:
                mask_vis_att = None

            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask_vis_att)
            x_vis_list.append(x_vis)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i]).transpose(-1, -2).contiguous()
        return x_vis_list


class Point_M2AE_Afford(nn.Module):
    """Point-M2AE segmentation backbone with affordance head(s).

    Modes:
      - dual_head=False (legacy): single Conv1d + sigmoid -> (B,N,1) in [0,1]
      - dual_head=True: classification head (sigmoid) + value head (ReLU).
        Default forward returns (prob * value) as (B,N,1); use return_parts=True
        to get (prob, value) each (B,N).
    """

    def __init__(
        self,
        cls_dim: int = 1,
        num_categories: int = 16,
        dual_head: bool = False,
        value_activation: str = "relu",
    ):
        super().__init__()
        self.trans_dim = 384
        self.group_sizes = [16, 8, 8]
        self.num_groups = [512, 256, 64]
        self.cls_dim = int(cls_dim)
        self.num_categories = int(num_categories)
        self.encoder_dims = [96, 192, 384]
        self.dual_head = bool(dual_head)
        self.value_activation = str(value_activation).lower()
        if self.value_activation not in {"relu", "sigmoid"}:
            raise ValueError(f"unknown value_activation={value_activation}")

        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        self.h_encoder = H_Encoder_seg()

        self.label_conv = nn.Sequential(
            nn.Conv1d(self.num_categories, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )

        self.propagations = nn.ModuleList()
        for i in range(3):
            self.propagations.append(
                PointNetFeaturePropagation_(
                    in_channel=self.encoder_dims[i] + 3,
                    mlp=[self.trans_dim * 4, 1024],
                )
            )

        self.convs1 = nn.Conv1d(6208, 1024, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(1024, 512, 1)
        self.convs3 = nn.Conv1d(512, 256, 1)
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if self.dual_head:
            self.convs4_cls = nn.Conv1d(256, 1, 1)
            self.convs4_val = nn.Conv1d(256, 1, 1)
        else:
            self.convs4 = nn.Conv1d(256, self.cls_dim, 1)

    def load_model_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict):
            if "base_model" in state_dict:
                weights = state_dict["base_model"]
            elif "model_state_dict" in state_dict:
                weights = state_dict["model_state_dict"]
            elif "model" in state_dict:
                weights = state_dict["model"]
            else:
                weights = state_dict
        else:
            weights = state_dict

        incompatible = self.load_state_dict(weights, strict=False)
        if incompatible.missing_keys:
            print_log("missing_keys", logger="Point_M2AE_Afford")
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger="Point_M2AE_Afford")
        if incompatible.unexpected_keys:
            print_log("unexpected_keys", logger="Point_M2AE_Afford")
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger="Point_M2AE_Afford",
            )

    def _backbone_feat(self, pts, cls_label=None):
        """Shared trunk up to 256-d per-point features. pts: (B,3,N) -> (B,256,N)."""
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2).contiguous()  # B N 3

        if cls_label is None:
            cls_label = torch.zeros(B, self.num_categories, device=pts.device, dtype=pts.dtype)
            cls_label[:, 0] = 1.0
        else:
            cls_label = cls_label.to(device=pts.device, dtype=pts.dtype)
            if cls_label.dim() == 1:
                eye = torch.eye(self.num_categories, device=pts.device, dtype=pts.dtype)
                cls_label = eye[cls_label.long()]

        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)

        x_vis_list = self.h_encoder(neighborhoods, centers, idxs, eval=True)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.propagations[i](
                pts.transpose(-1, -2),
                centers[i].transpose(-1, -2),
                pts.transpose(-1, -2),
                x_vis_list[i],
            )

        x = torch.cat((x_vis_list[0], x_vis_list[1], x_vis_list[2]), dim=1)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, self.num_categories, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature + x_avg_feature, cls_label_feature), 1)

        x = torch.cat((x_global_feature, x), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.relu(self.bns3(self.convs3(x)))
        return x  # B,256,N

    def forward(self, pts, cls_label=None, return_parts=False):
        """
        Args:
            pts: (B, 3, N)
            cls_label: (B, num_categories) one-hot. If None, use class-0 one-hot.
            return_parts: if dual_head, return (prob, value) each (B,N)
        Returns:
            dual_head + return_parts: (prob, value)
            dual_head: (B, N, 1) = prob * value (may be > 1)
            legacy: (B, N, 1) in [0, 1]
        """
        feat = self._backbone_feat(pts, cls_label=cls_label)
        if self.dual_head:
            prob = torch.sigmoid(self.convs4_cls(feat)).squeeze(1)  # B,N
            value_logits = self.convs4_val(feat)
            if self.value_activation == "sigmoid":
                value = torch.sigmoid(value_logits).squeeze(1)
            else:
                value = F.relu(value_logits).squeeze(1)
            if return_parts:
                return prob, value
            return (prob * value).unsqueeze(-1).contiguous()

        x = self.convs4(feat)
        x = torch.sigmoid(x)
        return x.permute(0, 2, 1).contiguous()


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_parts = {}

    def forward(self, pred, target):
        loss = F.mse_loss(pred.reshape(-1), target.reshape(-1))
        self.last_parts = {"mse": float(loss.detach()), "total": float(loss.detach())}
        return loss


class PeakFocusedLoss(nn.Module):
    """Weighted MSE + Pearson penalty + peak-region MSE (3DAffordanceNet)."""

    def __init__(
        self,
        fg_alpha=20.0,
        mse_weight=1.0,
        corr_weight=0.5,
        peak_weight=0.3,
        peak_threshold=0.3,
        eps=1e-6,
    ):
        super().__init__()
        self.fg_alpha = float(fg_alpha)
        self.mse_weight = float(mse_weight)
        self.corr_weight = float(corr_weight)
        self.peak_weight = float(peak_weight)
        self.peak_threshold = float(peak_threshold)
        self.eps = float(eps)
        self.last_parts = {}

    @staticmethod
    def _pearson_loss(pred, target, eps):
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        num = (pred * target).sum(dim=1)
        den = torch.sqrt((pred * pred).sum(dim=1) * (target * target).sum(dim=1) + eps)
        corr = num / den
        return (1.0 - corr).mean()

    def forward(self, pred, target):
        # pred/target: (B, N, 1) or (B, N)
        weights = 1.0 + self.fg_alpha * (target**2)
        mse = (weights * (pred - target) ** 2).mean()

        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        corr = self._pearson_loss(pred_flat, target_flat, self.eps)

        peak_mask = target_flat > self.peak_threshold
        if peak_mask.any():
            peak = ((pred_flat[peak_mask] - target_flat[peak_mask]) ** 2).mean()
        else:
            peak = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = self.mse_weight * mse + self.corr_weight * corr + self.peak_weight * peak
        self.last_parts = {
            "mse": float(mse.detach()),
            "corr": float(corr.detach()),
            "peak": float(peak.detach()),
            "total": float(total.detach()),
        }
        return total


class CEDiceLoss(nn.Module):
    """BCE + Dice (+ optional MSE on predicted-positive points).

    CE/Dice use binarized GT by default. Optional pos_mse compares continuous
    pred vs continuous GT on points where pred >= pos_mse_thresh.
    """

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        binarize_gt=True,
        gt_thresh=0.3,
        pos_mse_weight=0.0,
        pos_mse_thresh=0.3,
        eps=1e-6,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.binarize_gt = bool(binarize_gt)
        self.gt_thresh = float(gt_thresh)
        self.pos_mse_weight = float(pos_mse_weight)
        self.pos_mse_thresh = float(pos_mse_thresh)
        self.eps = float(eps)
        self.last_parts = {}

    def forward(self, pred, target):
        # pred already sigmoid'ed to [0,1]
        pred = pred.reshape(pred.shape[0], -1)
        target_cont = target.reshape(target.shape[0], -1).clamp(0.0, 1.0)
        target_ce = (target_cont >= self.gt_thresh).float() if self.binarize_gt else target_cont

        ce = F.binary_cross_entropy(pred, target_ce)

        inter = (pred * target_ce).sum(dim=1)
        denom = pred.sum(dim=1) + target_ce.sum(dim=1)
        dice = (1.0 - (2.0 * inter + self.eps) / (denom + self.eps)).mean()

        if self.pos_mse_weight > 0:
            mask = pred >= self.pos_mse_thresh  # 预测为正
            if mask.any():
                pos_mse = ((pred[mask] - target_cont[mask]) ** 2).mean()
            else:
                pos_mse = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            pos_mse = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.pos_mse_weight * pos_mse
        )
        self.last_parts = {
            "ce": float(ce.detach()),
            "dice": float(dice.detach()),
            "pos_mse": float(pos_mse.detach()),
            "total": float(total.detach()),
        }
        return total


class DualHeadLoss(nn.Module):
    """Two-head affordance loss.

    - Classification head (prob in [0,1]): CE + Dice vs binary GT (thresh).
    - Value head (ReLU): MSE vs continuous GT, only on GT-positive points.
    """

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        mse_weight=10.0,
        gt_thresh=0.3,
        eps=1e-6,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.mse_weight = float(mse_weight)
        self.gt_thresh = float(gt_thresh)
        self.eps = float(eps)
        self.last_parts = {}

    def forward(self, prob, value, target):
        prob = prob.reshape(prob.shape[0], -1)
        value = value.reshape(value.shape[0], -1)
        target = target.reshape(target.shape[0], -1).clamp(0.0, 1.0)
        gt_bin = (target >= self.gt_thresh).float()

        ce = F.binary_cross_entropy(prob, gt_bin)
        inter = (prob * gt_bin).sum(dim=1)
        denom = prob.sum(dim=1) + gt_bin.sum(dim=1)
        dice = (1.0 - (2.0 * inter + self.eps) / (denom + self.eps)).mean()

        mask = gt_bin > 0.5
        if mask.any():
            mse = ((value[mask] - target[mask]) ** 2).mean()
        else:
            mse = torch.zeros((), device=prob.device, dtype=prob.dtype)

        total = self.ce_weight * ce + self.dice_weight * dice + self.mse_weight * mse
        self.last_parts = {
            "ce": float(ce.detach()),
            "dice": float(dice.detach()),
            "mse": float(mse.detach()),
            "total": float(total.detach()),
        }
        return total


def get_loss(loss_type="peak_focused", **kwargs):
    """Factory: loss_type in {mse, peak_focused, ce_dice, dual_head}."""
    loss_type = str(loss_type).lower()
    if loss_type in ("mse", "l2"):
        return MSELoss()
    if loss_type in ("peak_focused", "peak", "focus_peak"):
        keys = ("fg_alpha", "mse_weight", "corr_weight", "peak_weight", "peak_threshold", "eps")
        return PeakFocusedLoss(**{k: kwargs[k] for k in keys if k in kwargs})
    if loss_type in ("ce_dice", "bce_dice", "dice_ce"):
        keys = (
            "ce_weight",
            "dice_weight",
            "binarize_gt",
            "gt_thresh",
            "pos_mse_weight",
            "pos_mse_thresh",
            "eps",
        )
        return CEDiceLoss(**{k: kwargs[k] for k in keys if k in kwargs})
    if loss_type in ("dual_head", "dual", "cls_val"):
        keys = ("ce_weight", "dice_weight", "mse_weight", "gt_thresh", "eps")
        return DualHeadLoss(**{k: kwargs[k] for k in keys if k in kwargs})
    raise ValueError(f"unknown loss_type={loss_type}")

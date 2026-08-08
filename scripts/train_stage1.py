#!/usr/bin/env python3
"""Stage-1: fine-tune Point-M2AE affordance with dual heads.

Protocol:
  - Head-cls (sigmoid): binary label @ --iou_gt_thresh, loss = CE + Dice
  - Head-val (ReLU): continuous affordance on GT-positive points only, loss = MSE * weight
  - Val / best: score = prob * value (may exceed 1), select by IoU@iou_gt_thresh
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_train.data.zarr_datasets import AffordanceCloudDataset
from vendor.point_m2ae import provider
from vendor.point_m2ae.Point_M2AE_Afford import Point_M2AE_Afford, get_loss


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Same param grouping as Point-M2AE main_pcd_affordance."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "token" in name
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def compute_iou_at_thresh(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.3) -> float:
    pred = pred.reshape(pred.shape[0], -1)
    gt = gt.reshape(gt.shape[0], -1)
    gt_bin = (gt >= thresh).astype(np.int32)
    pred_bin = (pred >= thresh).astype(np.int32)
    ious = []
    for i in range(pred.shape[0]):
        t = gt_bin[i]
        p = pred_bin[i]
        if t.sum() == 0:
            continue
        inter = np.sum(p & t)
        union = np.sum(p | t)
        ious.append(float(inter) / float(union + 1e-8))
    return float(np.mean(ious)) if ious else float("nan")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", type=str, default=str(ROOT / "data" / "joint_door.zarr"))
    p.add_argument("--ckpts", type=str, default=str(ROOT / "ckpts" / "pre-train.pth"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "runs" / "stage1_dual"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_points", type=int, default=None)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--warmup_epoch", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=4e-4)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--iou_gt_thresh", type=float, default=0.3)
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--mse_weight", type=float, default=10.0, help="GT-positive value MSE weight")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument(
        "--view_mode",
        choices=("primary", "augmentation", "combined"),
        default="primary",
        help="training views: primary, target-aware augmentation, or both",
    )
    p.add_argument(
        "--val_view_mode",
        choices=("primary", "augmentation", "combined"),
        default="primary",
        help="validation views; primary keeps IoU comparable to the original single-view baseline",
    )
    p.add_argument("--label_source", choices=("updated", "initial"), default="updated")
    p.add_argument("--seed", type=int, default=20260726)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresh: float):
    model.eval()
    losses, ious = [], []
    parts_sum = {"ce": 0.0, "dice": 0.0, "mse": 0.0}
    n = 0
    for batch in loader:
        xyz = batch["xyz"].to(device)
        gt = batch["affordance"].to(device)
        pts = xyz.transpose(1, 2).contiguous()
        prob, value = model(pts, return_parts=True)
        pred = prob * value
        loss = criterion(prob, value, gt)
        losses.append(float(loss.detach()))
        parts = getattr(criterion, "last_parts", {}) or {}
        for k in parts_sum:
            if k in parts:
                parts_sum[k] += float(parts[k])
        n += 1
        ious.append(
            compute_iou_at_thresh(
                pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), thresh
            )
        )
    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "iou": float(np.nanmean(ious)) if ious else float("nan"),
    }
    if n > 0:
        for k, v in parts_sum.items():
            out[k] = v / n
    return out


def main():
    args = parse_args()
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    train_set = AffordanceCloudDataset(
        args.zarr, split="train", augment=not args.no_augment, view_mode=args.view_mode,
        label_source=args.label_source, num_points=args.num_points
    )
    val_set = AffordanceCloudDataset(
        args.zarr, split="val", augment=False, view_mode=args.val_view_mode,
        label_source=args.label_source, num_points=args.num_points
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Point_M2AE_Afford(cls_dim=1, num_categories=16, dual_head=True).to(device)
    model.apply(inplace_relu)
    if args.ckpts and Path(args.ckpts).exists():
        print(f"load pretrain: {args.ckpts}", flush=True)
        model.load_model_from_ckpt(args.ckpts)
    else:
        print(f"[warn] pretrain missing: {args.ckpts}", flush=True)

    criterion = get_loss(
        "dual_head",
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        mse_weight=args.mse_weight,
        gt_thresh=args.iou_gt_thresh,
    )
    print(
        f"dual_head: cls=CE+Dice({args.ce_weight}/{args.dice_weight}) "
        f"val=GT+MSE*{args.mse_weight} thresh={args.iou_gt_thresh} "
        f"score=prob*value epochs={args.epoch}",
        flush=True,
    )

    optimizer = optim.AdamW(
        add_weight_decay(model, weight_decay=0.05),
        lr=args.learning_rate,
        weight_decay=0.05,
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epoch,
        lr_min=1e-6,
        cycle_mul=1.0,
        cycle_decay=0.1,
        warmup_lr_init=1e-6,
        warmup_t=args.warmup_epoch,
        cycle_limit=1,
        t_in_epochs=True,
    )

    best_iou = -1.0
    history = []
    for epoch in range(args.epoch):
        model.train()
        train_losses = []
        train_parts_sum = {"ce": 0.0, "dice": 0.0, "mse": 0.0}
        pbar = tqdm(train_loader, desc=f"ep{epoch+1}/{args.epoch}[dual]")
        for batch in pbar:
            points_np = batch["xyz"].numpy()
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
            xyz = torch.from_numpy(points_np).float().to(device)
            gt = batch["affordance"].to(device)
            pts = xyz.transpose(1, 2).contiguous()
            prob, value = model(pts, return_parts=True)
            loss = criterion(prob, value, gt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            parts = getattr(criterion, "last_parts", {}) or {}
            for k in train_parts_sum:
                if k in parts:
                    train_parts_sum[k] += float(parts[k])
            pbar.set_postfix(
                loss=f"{train_losses[-1]:.4f}",
                ce=f"{parts.get('ce', float('nan')):.3f}",
                dice=f"{parts.get('dice', float('nan')):.3f}",
                mse=f"{parts.get('mse', float('nan')):.3f}",
            )

        n_train = max(len(train_losses), 1)
        train_parts = {k: v / n_train for k, v in train_parts_sum.items()}
        scheduler.step(epoch)
        val = evaluate(model, val_loader, criterion, device, args.iou_gt_thresh)
        row = {
            "epoch": epoch + 1,
            "phase": "dual_head",
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_parts.items()},
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        print(
            f"[epoch {epoch+1}] "
            f"train={row['train_loss']:.4f} "
            f"(ce={train_parts['ce']:.4f} dice={train_parts['dice']:.4f} mse={train_parts['mse']:.4f}) | "
            f"val={val['loss']:.4f} "
            f"(ce={val.get('ce', float('nan')):.4f} dice={val.get('dice', float('nan')):.4f} "
            f"mse={val.get('mse', float('nan')):.4f}) | "
            f"iou@{args.iou_gt_thresh:g}={val['iou']:.4f} lr={row['lr']:.2e}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
                "dual_head": True,
            },
            out_dir / "last.pth",
        )
        if val["iou"] == val["iou"] and val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_iou": best_iou,
                    "args": vars(args),
                    "dual_head": True,
                },
                out_dir / "best.pth",
            )
            print(f"  -> new best iou={best_iou:.4f}", flush=True)

        (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    print(f"DONE best_iou={best_iou:.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

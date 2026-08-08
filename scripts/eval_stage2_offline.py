#!/usr/bin/env python3
"""Offline GT-affordance metrics for a Stage-2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_train.data.zarr_datasets import JointActionDataset
from joint_train.models.joint_policy import JointDiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", type=str, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--max_val_objects", type=int, default=4)
    p.add_argument("--val_traj_per_object", type=int, default=100000)
    p.add_argument("--max_batches", type=int, default=0, help="0=all batches")
    p.add_argument("--skip_action_metrics", action="store_true", help="evaluate diffusion loss only")
    p.add_argument("--val_manifest", type=Path, default=None, help="JSON manifest with val.episode_ids")
    return p.parse_args()


def load_policy(ckpt_path: Path, device: torch.device) -> JointDiffusionPolicy:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    dual = bool(ckpt.get("dual_head", False)) if isinstance(ckpt, dict) else False
    if not dual and isinstance(state, dict):
        dual = any("convs4_cls" in key for key in state)
    policy = JointDiffusionPolicy(
        action_dim=9,
        state_dim=11,
        horizon=int(saved_args.get("horizon", 16)),
        n_obs_steps=int(saved_args.get("n_obs_steps", 2)),
        n_action_steps=int(saved_args.get("n_action_steps", 8)),
        down_dims=tuple(saved_args.get("down_dims", (512, 1024, 2048))),
        reuse_static_point_feature=bool(saved_args.get("reuse_static_point_feature", False)),
        dual_head=dual,
    )
    weights = ckpt.get("ema") if isinstance(ckpt, dict) and ckpt.get("ema") is not None else state
    policy.load_state_dict(weights, strict=False)
    if isinstance(ckpt, dict) and ckpt.get("normalizer") is not None:
        policy.normalizer.load_state_dict(ckpt["normalizer"])
    return policy.to(device).eval()


def load_manifest_episode_ids(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    try:
        return [int(item) for item in payload["val"]["episode_ids"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} has no val.episode_ids") from exc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = JointActionDataset(
        args.zarr,
        split="val",
        seed=args.seed,
        max_val_objects=args.max_val_objects,
        val_traj_per_object=args.val_traj_per_object,
        random_val_objects=True,
        episode_ids=load_manifest_episode_ids(args.val_manifest),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    policy = load_policy(args.ckpt, device)
    total = sequence_count = action_steps = loss_batches = loss_sequences = 0
    loss_sum = mae_sum = sq_sum = joint_mae_sum = gripper_mae_sum = 0.0
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            if args.max_batches > 0 and batch_i >= args.max_batches:
                break
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            loss = policy.compute_loss(batch, use_gt_affordance=True)
            batch_sequences = batch["action"].shape[0]
            loss_sequences += batch_sequences
            if args.skip_action_metrics:
                loss_batches += 1
                loss_sum += float(loss) * batch_sequences
                continue
            pred = policy.predict_action(batch, use_gt_affordance=True)["action"]
            start = policy.n_obs_steps - 1
            target = batch["action"][:, start : start + policy.n_action_steps]
            diff = pred - target
            n = diff.numel()
            total += n
            sequence_count += pred.shape[0]
            action_steps += pred.shape[0] * pred.shape[1]
            loss_batches += 1
            loss_sum += float(loss) * pred.shape[0]
            mae_sum += float(diff.abs().sum())
            sq_sum += float(diff.square().sum())
            joint_mae_sum += float(diff[..., :7].abs().sum())
            gripper_mae_sum += float(diff[..., 7:9].abs().sum())
            if (batch_i + 1) % 20 == 0:
                print(f"offline batch {batch_i+1}/{len(loader)}", flush=True)
    loss_batches = max(1, loss_batches)
    loss_weight = max(1, loss_sequences)
    action_steps = max(1, action_steps)
    metrics = {
        "affordance_source": "gt",
        "n_sequences": loss_sequences,
        "n_episodes": dataset.subset_info["n_episodes"],
        "targets": [
            {"obj": item["obj"], "n_traj": item.get("n_traj", 1)}
            for item in dataset.subset_detail
        ],
        "diffusion_loss_mean": loss_sum / loss_weight,
        "action_mae": None if args.skip_action_metrics else mae_sum / total,
        "action_rmse": None if args.skip_action_metrics else float(np.sqrt(sq_sum / total)),
        "joint_mae": None if args.skip_action_metrics else joint_mae_sum / (action_steps * 7),
        "gripper_mae": None if args.skip_action_metrics else gripper_mae_sum / (action_steps * 2),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "offline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

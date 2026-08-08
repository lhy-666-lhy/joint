#!/usr/bin/env python3
"""Stage-2: train DP3 diffusion head (+ optional Point-M2AE finetune).

Supports single-GPU and multi-GPU DDP via torchrun.

Modes:
  --affordance_source infer   (default): freeze stage1, use predicted affordance
  --affordance_source gt      : freeze stage1, use zarr GT affordance
  --unfreeze_stage1           : also finetune Point-M2AE at --stage1_lr (default 1e-5)

Batch: --batch_size is PER-GPU. Global batch = batch_size * world_size.
  e.g. 4 GPUs + --batch_size 64  ->  global batch 256
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_train.data.zarr_datasets import JointActionDataset
from joint_train.models.joint_policy import JointDiffusionPolicy
from vendor.dp3.ema_model import EMAModel
from vendor.dp3.lr_scheduler import get_scheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", type=str, default=str(ROOT / "data" / "joint_door.zarr"))
    p.add_argument("--stage1_ckpt", type=str, default=str(ROOT / "runs" / "stage1_dual" / "best.pth"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "runs" / "stage2_tiny100"))
    p.add_argument(
        "--affordance_source",
        type=str,
        default="infer",
        choices=["infer", "gt"],
        help="infer: stage1 predicted affordance; gt: zarr GT affordance",
    )
    p.add_argument(
        "--condition_variant",
        choices=("updated", "initial", "no_map"),
        default="updated",
        help="policy condition: updated GT/prediction, initial GT map, or zero map",
    )
    p.add_argument(
        "--unfreeze_stage1",
        action="store_true",
        help="finetune Point-M2AE with --stage1_lr (implies affordance_source=infer)",
    )
    p.add_argument("--stage1_lr", type=float, default=1e-5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="per-GPU batch size; global_batch = batch_size * n_gpus",
    )
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="stop after this many optimizer updates (0=use num_epochs)",
    )
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_obs_steps", type=int, default=2)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES, e.g. '0,1,2,3' for 4-GPU",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument(
        "--val_every_steps",
        type=int,
        default=0,
        help="validate every N updates (0=use --val_every epochs)",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="stop after this many validation checks without improvement (0=disabled)",
    )
    p.add_argument("--early_stop_min_epochs", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument(
        "--max_val_episodes",
        type=int,
        default=2,
        help="only first N real-val episodes for quick val (0=full val)",
    )
    p.add_argument("--max_train_episodes", type=int, default=None)
    p.add_argument(
        "--max_train_objects",
        type=int,
        default=10,
        help="use first K train objects (0=all)",
    )
    p.add_argument(
        "--random_train_objects",
        action="store_true",
        help="sample train objects with --seed instead of taking dataset order",
    )
    p.add_argument(
        "--train_trajectory_fraction",
        type=float,
        default=1.0,
        help="randomly retain this trajectory fraction within each selected train object",
    )
    p.add_argument("--sequence_stride", type=int, default=1, help="keep every K-th train window per episode")
    p.add_argument("--max_val_objects", type=int, default=0, help="cap validation to distinct objects (0=all)")
    p.add_argument("--val_traj_per_object", type=int, default=1)
    p.add_argument("--random_val_objects", action="store_true")
    p.add_argument(
        "--down_dims",
        type=str,
        default="512,1024,2048",
        help="comma-separated diffusion U-Net widths",
    )
    p.add_argument(
        "--reuse_static_point_feature",
        action="store_true",
        help="encode the repeated static observation cloud once per sample",
    )
    p.add_argument(
        "--obs_encoder_variant",
        choices=("pointnet", "token_cross_attention"),
        default="pointnet",
        help="point-cloud aggregation used to construct the DP3 global condition",
    )
    p.add_argument(
        "--affordance_adapter",
        choices=("none", "residual"),
        default="none",
        help="optional explicit map-biased residual branch on the stable PointNet path",
    )
    p.add_argument(
        "--affordance_aux_weight",
        type=float,
        default=0.0,
        help="training-only BCE+soft-Dice weight on shared PointNet per-point features",
    )
    p.add_argument(
        "--contact_condition",
        choices=("none", "coordinate"),
        default="none",
        help="optional fixed trajectory-level contact token",
    )
    p.add_argument(
        "--contact_sidecar",
        type=Path,
        default=None,
        help="episode-index-aligned contact sidecar required by coordinate conditioning",
    )
    p.add_argument(
        "--traj_per_object",
        type=int,
        default=10,
        help="random trajs per object (cap); 10 objs × 10 = 100 max",
    )
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--view_mode", choices=("primary", "augmentation", "combined"), default="primary")
    p.add_argument("--val_view_mode", choices=("primary", "augmentation", "combined"), default="primary")
    p.add_argument("--train_manifest", type=Path, default=None, help="JSON manifest with train.episode_ids")
    p.add_argument("--val_manifest", type=Path, default=None, help="JSON manifest with val.episode_ids")
    p.add_argument(
        "--sampler",
        choices=("sequence_uniform", "target_balanced"),
        default="sequence_uniform",
        help="training sampling distribution",
    )
    p.add_argument("--local_rank", type=int, default=-1, help="set by torchrun")
    return p.parse_args()


def setup_distributed(args):
    """Init DDP if launched by torchrun / torch.distributed."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        # CUDA_VISIBLE_DEVICES should be set by the launcher, e.g.
        # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        if args.gpu is not None and str(args.gpu).strip() != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def build_optimizer(policy: JointDiffusionPolicy, args):
    stage1_params = list(policy.affordance_net.parameters())
    other_params = [
        p
        for n, p in policy.named_parameters()
        if not n.startswith("affordance_net.") and p.requires_grad
    ]
    groups = [{"params": other_params, "lr": args.lr}]
    if args.unfreeze_stage1:
        groups.append({"params": stage1_params, "lr": args.stage1_lr})
    return torch.optim.AdamW(groups, betas=(0.95, 0.999), eps=1e-8, weight_decay=args.weight_decay)


def save_ckpt(path: Path, policy, ema_model, args, epoch, best_val=None):
    raw = unwrap(policy)
    payload = {
        "model": raw.state_dict(),
        "ema": ema_model.averaged_model.state_dict() if ema_model else None,
        "normalizer": raw.normalizer.state_dict(),
        "epoch": epoch,
        "args": json.loads(json.dumps(vars(args), default=str)),
        "dual_head": bool(getattr(raw, "dual_head", False)),
    }
    if best_val is not None:
        payload["best_val"] = best_val
    torch.save(payload, path)


def compute_train_loss(policy, batch, args, use_gt: bool):
    """Compute loss through DDP wrapper so gradients sync correctly."""
    raw = unwrap(policy)
    if args.unfreeze_stage1:
        return policy(batch, use_gt_affordance=False)
    if use_gt:
        return policy(batch, use_gt_affordance=True)
    with torch.no_grad():
        aff = raw.predict_affordance(batch["point_cloud_xyz"])
    batch_gt = dict(batch)
    batch_gt["affordance_gt"] = aff
    return policy(batch_gt, use_gt_affordance=True)


def load_manifest_episode_ids(path: Path | None, split: str) -> list[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    try:
        return [int(item) for item in payload[split]["episode_ids"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} has no {split}.episode_ids") from exc


def parse_down_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(x.strip()) for x in str(value).split(",") if x.strip())
    if len(dims) < 2 or any(x <= 0 for x in dims):
        raise ValueError(f"--down_dims must contain at least two positive widths, got {value!r}")
    return dims


class TargetBalancedDistributedSampler(Sampler[int]):
    """Sample target, episode, and sequence uniformly, then shard across ranks."""

    def __init__(self, dataset: JointActionDataset, num_replicas: int, rank: int, seed: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(np.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        episode_ids = np.searchsorted(
            dataset.episode_ends,
            np.asarray(dataset.indices[:, 0], dtype=np.int64),
            side="right",
        )
        windows_per_episode = np.bincount(episode_ids, minlength=len(dataset.episode_ends))
        target_by_episode = [
            dataset.replay_obj_keys[int(dataset.episode_replay_ids[episode_id])]
            for episode_id in range(len(dataset.episode_ends))
        ]
        selected_episodes = np.nonzero(windows_per_episode > 0)[0]
        episodes_per_target: dict[str, int] = {}
        for episode_id in selected_episodes.tolist():
            target = target_by_episode[episode_id]
            episodes_per_target[target] = episodes_per_target.get(target, 0) + 1
        weights = [
            1.0
            / (
                episodes_per_target[target_by_episode[episode_id]]
                * int(windows_per_episode[episode_id])
            )
            for episode_id in episode_ids.tolist()
        ]
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def main():
    args = parse_args()
    args.down_dims = parse_down_dims(args.down_dims)
    if args.condition_variant == "initial" and args.affordance_source != "gt":
        raise ValueError("--condition_variant initial requires --affordance_source gt")
    if args.contact_condition != "none" and args.contact_sidecar is None:
        raise ValueError("--contact_condition requires --contact_sidecar")
    if args.no_ema:
        args.use_ema = False

    distributed, rank, local_rank, world_size, device = setup_distributed(args)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    pad_before = args.n_obs_steps - 1
    pad_after = args.n_action_steps - 1
    max_objs = args.max_train_objects if args.max_train_objects and args.max_train_objects > 0 else None
    train_episode_ids = load_manifest_episode_ids(args.train_manifest, "train")
    val_episode_ids = load_manifest_episode_ids(args.val_manifest, "val")
    train_set = JointActionDataset(
        args.zarr,
        horizon=args.horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        split="train",
        seed=args.seed,
        max_train_episodes=args.max_train_episodes,
        max_train_objects=max_objs,
        traj_per_object=args.traj_per_object,
        random_train_objects=args.random_train_objects,
        train_trajectory_fraction=args.train_trajectory_fraction,
        sequence_stride=args.sequence_stride,
        view_mode=args.view_mode,
        episode_ids=train_episode_ids,
        affordance_label_source=("initial" if args.condition_variant == "initial" else "updated"),
        contact_sidecar=(str(args.contact_sidecar) if args.contact_sidecar is not None else None),
    )
    val_set = JointActionDataset(
        args.zarr,
        horizon=args.horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        split="val",
        seed=args.seed,
        max_val_episodes=(args.max_val_episodes if args.max_val_episodes > 0 else None),
        max_val_objects=(args.max_val_objects if args.max_val_objects > 0 else None),
        val_traj_per_object=args.val_traj_per_object,
        random_val_objects=args.random_val_objects,
        view_mode=args.val_view_mode,
        episode_ids=val_episode_ids,
        affordance_label_source=("initial" if args.condition_variant == "initial" else "updated"),
        contact_sidecar=(str(args.contact_sidecar) if args.contact_sidecar is not None else None),
    )
    global_batch = args.batch_size * world_size
    if is_main(rank):
        print(
            f"DDP world_size={world_size} per_gpu_batch={args.batch_size} "
            f"global_batch={global_batch}",
            flush=True,
        )
        print(
            f"train episodes={train_set.subset_info['n_episodes']} "
            f"sequences={len(train_set)} | "
            f"val episodes={val_set.subset_info['n_episodes']} "
            f"sequences={len(val_set)} (max_val_episodes={args.max_val_episodes})",
            flush=True,
        )
        if train_set.subset_detail:
            detail = ", ".join(f"{d['obj']}x{d['n_traj']}" for d in train_set.subset_detail)
            print(
                f"subset objects={len(train_set.subset_detail)} "
                f"traj_per_object<={args.traj_per_object}: {detail}",
                flush=True,
            )
            (out_dir / "train_subset.json").write_text(
                json.dumps(train_set.subset_detail, indent=2) + "\n"
            )
        if val_set.subset_detail:
            print(
                "val quick episodes: "
                + ", ".join(
                    f"{d['obj']}(ep{d['episode_id']})"
                    if "episode_id" in d
                    else f"{d['obj']}x{d.get('n_traj', 1)}"
                    for d in val_set.subset_detail
                ),
                flush=True,
            )
            (out_dir / "val_subset.json").write_text(
                json.dumps(val_set.subset_detail, indent=2) + "\n"
            )

    if args.sampler == "target_balanced":
        train_sampler = TargetBalancedDistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
    elif distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    # Val only on rank0 to avoid heavy multi-rank IO on full val
    val_loader = None
    if is_main(rank):
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
        )

    use_gt = args.affordance_source == "gt"
    ckpt_path = Path(args.stage1_ckpt)
    dual_head = False
    stage1_state = None
    if not use_gt and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        stage1_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        dual_head = bool(ckpt.get("dual_head", False)) if isinstance(ckpt, dict) else False
        if not dual_head and isinstance(stage1_state, dict):
            dual_head = any(k.startswith("convs4_cls") for k in stage1_state)

    policy = JointDiffusionPolicy(
        action_dim=9,
        state_dim=11,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        down_dims=args.down_dims,
        reuse_static_point_feature=args.reuse_static_point_feature,
        condition_mode=("no_map" if args.condition_variant == "no_map" else "affordance"),
        obs_encoder_variant=args.obs_encoder_variant,
        affordance_adapter=args.affordance_adapter,
        affordance_aux_weight=args.affordance_aux_weight,
        contact_condition=args.contact_condition,
        dual_head=dual_head,
    ).to(device)

    if stage1_state is not None:
        missing, unexpected = policy.affordance_net.load_state_dict(stage1_state, strict=False)
        if is_main(rank):
            print(
                f"loaded stage1 from {ckpt_path} dual_head={dual_head} "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
    elif not use_gt and is_main(rank):
        print(f"[warn] stage1 ckpt missing: {ckpt_path}", flush=True)

    if use_gt and args.unfreeze_stage1:
        if is_main(rank):
            print("[warn] gt affordance + unfreeze_stage1: freezing stage1.", flush=True)
        args.unfreeze_stage1 = False

    if args.unfreeze_stage1:
        set_requires_grad(policy.affordance_net, True)
        policy.affordance_net.train()
    else:
        set_requires_grad(policy.affordance_net, False)
        policy.affordance_net.eval()

    normalizer = train_set.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.normalizer.to(device)

    if distributed:
        # frozen affordance params get no grad -> need find_unused_parameters
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=not args.unfreeze_stage1,
        )

    optimizer = build_optimizer(unwrap(policy), args)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(args.max_train_steps if args.max_train_steps > 0 else args.num_epochs * len(train_loader)),
    )

    ema_model = None
    if args.use_ema and is_main(rank):
        ema_model = EMAModel(
            model=copy.deepcopy(unwrap(policy)),
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            max_value=0.9999,
        )

    history = []
    global_step = 0
    best_val = float("inf")
    no_improve_checks = 0
    start_time = time.perf_counter()
    stop_training = False
    next_val_step = args.val_every_steps if args.val_every_steps > 0 else None
    total_train_steps = (
        min(args.num_epochs * len(train_loader), args.max_train_steps)
        if args.max_train_steps > 0
        else args.num_epochs * len(train_loader)
    )
    progress = tqdm(total=total_train_steps, desc="train updates", disable=not is_main(rank))

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        unwrap(policy).train()
        if not args.unfreeze_stage1:
            unwrap(policy).affordance_net.eval()

        train_losses = []
        for batch in train_loader:
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss = compute_train_loss(policy, batch, args, use_gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if ema_model is not None:
                ema_model.step(unwrap(policy))
            train_losses.append(float(loss.detach()))
            global_step += 1
            if is_main(rank):
                progress.update(1)
                progress.set_postfix(
                    loss=f"{train_losses[-1]:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    gbs=global_batch,
                )
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break

        # average train loss across ranks
        local_sum = float(np.sum(train_losses)) if train_losses else 0.0
        local_n = float(len(train_losses))
        if distributed:
            t = torch.tensor([local_sum, local_n], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mean_loss = float(t[0] / max(t[1], 1.0))
        else:
            mean_loss = local_sum / max(local_n, 1.0)

        row = {
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "global_step": global_step,
            "global_batch": global_batch,
            "world_size": world_size,
            "elapsed_seconds": time.perf_counter() - start_time,
            "seconds_per_step": (time.perf_counter() - start_time) / max(global_step, 1),
        }

        should_validate = (
            args.val_every_steps > 0
            and (stop_training or (next_val_step is not None and global_step >= next_val_step))
        ) or (args.val_every_steps <= 0 and (epoch + 1) % args.val_every == 0)
        if is_main(rank) and val_loader is not None and should_validate and len(val_loader) > 0:
            eval_policy = ema_model.averaged_model if ema_model is not None else unwrap(policy)
            eval_policy.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                    if use_gt:
                        vloss = eval_policy.compute_loss(
                            batch,
                            use_gt_affordance=True,
                            include_auxiliary=False,
                        )
                    else:
                        aff = eval_policy.predict_affordance(batch["point_cloud_xyz"])
                        batch_gt = dict(batch)
                        batch_gt["affordance_gt"] = aff
                        vloss = eval_policy.compute_loss(
                            batch_gt,
                            use_gt_affordance=True,
                            include_auxiliary=False,
                        )
                    val_losses.append(float(vloss))
            row["val_loss"] = float(np.mean(val_losses)) if val_losses else float("nan")
            if row["val_loss"] < best_val - args.early_stop_min_delta:
                best_val = row["val_loss"]
                no_improve_checks = 0
                save_ckpt(out_dir / "best.pth", policy, ema_model, args, epoch + 1, best_val=best_val)
                print(f"  -> new best val_loss={best_val:.6f}", flush=True)
            else:
                no_improve_checks += 1
            row["no_improve_checks"] = no_improve_checks
        early_stop = bool(
            is_main(rank)
            and args.early_stop_patience > 0
            and should_validate
            and (epoch + 1) >= args.early_stop_min_epochs
            and no_improve_checks >= args.early_stop_patience
        )
        if distributed:
            signal = torch.tensor([int(early_stop)], device=device, dtype=torch.int32)
            dist.broadcast(signal, src=0)
            early_stop = bool(signal.item())
        if early_stop:
            stop_training = True
            if is_main(rank):
                print(
                    f"early stop at epoch={epoch+1} after {no_improve_checks} validation checks",
                    flush=True,
                )
        if next_val_step is not None and should_validate:
            while next_val_step <= global_step:
                next_val_step += args.val_every_steps

        if is_main(rank):
            history.append(row)
            print(
                f"[epoch {epoch+1}] train={row['train_loss']:.6f} "
                f"val={row.get('val_loss', float('nan')):.6f} "
                f"global_batch={global_batch}",
                flush=True,
            )
            if (epoch + 1) % args.checkpoint_every == 0:
                save_ckpt(out_dir / f"epoch_{epoch+1}.pth", policy, ema_model, args, epoch + 1)
            save_ckpt(out_dir / "last.pth", policy, ema_model, args, epoch + 1)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

        if distributed:
            dist.barrier()
        if stop_training:
            break

    progress.close()

    if is_main(rank):
        result = {
            "status": "done",
            "best_val_loss": best_val,
            "global_step": global_step,
            "epochs": len(history),
            "elapsed_seconds": time.perf_counter() - start_time,
            "condition_variant": args.condition_variant,
            "obs_encoder_variant": args.obs_encoder_variant,
            "affordance_adapter": args.affordance_adapter,
            "affordance_aux_weight": args.affordance_aux_weight,
            "contact_condition": args.contact_condition,
            "sampler": args.sampler,
            "seed": args.seed,
            "total_parameters": sum(p.numel() for p in unwrap(policy).parameters()),
            "trainable_parameters": sum(
                p.numel() for p in unwrap(policy).parameters() if p.requires_grad
            ),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"DONE best_val={best_val:.6f} -> {out_dir}", flush=True)
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()

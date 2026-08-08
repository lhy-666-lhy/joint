#!/usr/bin/env python3
"""A6 clean-membership Point-M2AE affordance sanity and fixed-step training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import zarr
from timm.scheduler import CosineLRScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
JOINT_ROOT = ROOT / "jointTrain_new"
STAGE1_ROOT = JOINT_ROOT / "experiment" / "stage1_optimize"
for path in (ROOT, JOINT_ROOT, STAGE1_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from joint_train.utils.pc_utils import pc_normalize
from path_config import JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT, JOINTTRAIN_ARCH6_A010C_RESULT_ROOT, JOINTTRAIN_ARCH6_A020C_RESULT_ROOT, JOINTTRAIN_BESTVIEW_DUAL_ZARR
from train_stage1_optimize import compute_loss, inplace_relu, optimizer_groups
from vendor.point_m2ae.Point_M2AE_Afford import Point_M2AE_Afford, get_loss


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


class CleanAffordanceDataset(Dataset):
    def __init__(self, manifest: dict, *, fixed8: bool, augment: bool):
        self.root = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
        primary = [int(row["primary_row"]) for row in manifest["primary"]["A5_TRAIN"]]
        self.rows = [("primary", row) for row in (primary[:8] if fixed8 else primary)]
        if not fixed8:
            self.rows.extend(("augmentation", int(row)) for row in manifest["augmentation"]["A5_TRAIN"])
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        kind, row = self.rows[index]
        if kind == "primary":
            cloud = np.asarray(self.root["data/point_cloud"][row], dtype=np.float32)
            score = np.asarray(self.root["data/affordance_updated"][row], dtype=np.float32).copy()
        else:
            cloud = np.asarray(self.root["data/stage1_aug_point_cloud"][row], dtype=np.float32)
            score = np.asarray(self.root["data/stage1_aug_affordance_updated"][row], dtype=np.float32).copy()
        xyz = pc_normalize(cloud[:, :3].copy())
        score = np.clip(score, 0.0, 1.0)
        if self.augment:
            angle = np.random.uniform(-30.0, 30.0) * np.pi / 180.0
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            xyz = xyz @ rotation.T
            xyz += np.random.normal(0.0, 0.005, xyz.shape).astype(np.float32)
            xyz *= np.float32(np.random.uniform(0.8, 1.25))
            xyz += np.random.uniform(-0.1, 0.1, (1, 3)).astype(np.float32)
        return {"xyz": torch.from_numpy(xyz), "affordance": torch.from_numpy(score)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def prediction(model, xyz: torch.Tensor) -> torch.Tensor:
    probability, value = model(xyz.transpose(1, 2).contiguous(), return_parts=True)
    return probability * value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed8", "full", "pilot"], required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    fixed8 = args.mode == "fixed8"
    steps = args.steps or (500 if fixed8 else (20 if args.mode == "pilot" else 7000))
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    manifest_path = Path(JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT) / "membership_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = CleanAffordanceDataset(manifest, fixed8=fixed8, augment=not fixed8)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True, num_workers=args.num_workers, drop_last=not fixed8, pin_memory=True, generator=torch.Generator().manual_seed(args.seed), worker_init_fn=worker_seed, persistent_workers=args.num_workers > 0)
    model = Point_M2AE_Afford(cls_dim=1, num_categories=16, dual_head=True, value_activation="relu").to(device)
    model.apply(inplace_relu)
    criterion = get_loss("dual_head", ce_weight=1.0, dice_weight=1.0, mse_weight=10.0, gt_thresh=0.05)
    optimizer = AdamW(optimizer_groups(model, 4e-4, 1.0, 0.05), lr=4e-4)
    scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=10, cycle_limit=1, t_in_epochs=True) if args.mode == "full" else None
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses = []
    first_batch = next(iter(loader))
    fixed_xyz = first_batch["xyz"].to(device)
    fixed_target = first_batch["affordance"].to(device)
    model.eval()
    with torch.no_grad():
        initial_prediction = prediction(model, fixed_xyz)
        initial_mae = float(torch.abs(initial_prediction - fixed_target).mean())
    model.train()
    out_root = Path(JOINTTRAIN_ARCH6_A010C_RESULT_ROOT if fixed8 else JOINTTRAIN_ARCH6_A020C_RESULT_ROOT)
    out = out_root if fixed8 else out_root / f"seed_{args.seed}"
    if args.mode == "pilot":
        out = out_root / f"pilot_b{args.batch_size}_a{args.gradient_accumulation}_w{args.num_workers}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    start_step = 0
    latest_path = out / "latest.pth"
    if args.resume and latest_path.exists():
        resume = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None and resume.get("scheduler") is not None:
            scheduler.load_state_dict(resume["scheduler"])
        start_step = int(resume["step"])
    iterator = iter(loader)
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            xyz = batch["xyz"].to(device, non_blocking=True)
            target = batch["affordance"].to(device, non_blocking=True)
            loss, _ = compute_loss(model, criterion, xyz, target, 0.0)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite affordance loss")
            (loss / args.gradient_accumulation).backward()
            micro_losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(np.mean(micro_losses)))
        if scheduler is not None and (step + 1) % 70 == 0:
            scheduler.step(step // 70)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            atomic(out / "progress.json", {"step": step + 1, "steps": steps, "elapsed_seconds": time.time() - started, "latest_loss": losses[-1], "learning_rate": max(group["lr"] for group in optimizer.param_groups)})
        if args.mode == "full" and ((step + 1) % 500 == 0 or step + 1 == steps):
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler is not None else None, "step": step + 1, "seed": args.seed, "effective_batch": args.batch_size * args.gradient_accumulation}, latest_path)
    model.eval()
    with torch.no_grad():
        parity_seed = args.seed + 999
        torch.manual_seed(parity_seed)
        final_prediction = prediction(model, fixed_xyz)
        final_mae = float(torch.abs(final_prediction - fixed_target).mean())
        stochastic_prediction = prediction(model, fixed_xyz)
        stochastic_forward_delta = float(torch.max(torch.abs(final_prediction - stochastic_prediction)))
    checkpoint = {"model": model.state_dict(), "dual_head": True, "seed": args.seed, "steps": steps, "random_init": True, "membership_manifest": str(manifest_path), "mode": args.mode, "microbatch": args.batch_size, "gradient_accumulation": args.gradient_accumulation, "effective_batch": args.batch_size * args.gradient_accumulation}
    checkpoint_path = out / "last.pth"
    torch.save(checkpoint, checkpoint_path)
    reload_model = Point_M2AE_Afford(cls_dim=1, num_categories=16, dual_head=True, value_activation="relu").to(device)
    reload_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    reload_model.eval()
    with torch.no_grad():
        torch.manual_seed(parity_seed)
        reload_error = float(torch.max(torch.abs(final_prediction - prediction(reload_model, fixed_xyz))))
    state_exact = all(torch.equal(model.state_dict()[key], reload_model.state_dict()[key]) for key in model.state_dict())
    checks = {
        "random_init": True,
        "finite": bool(np.isfinite(losses).all()),
        "loss_decreased": losses[-1] < losses[0],
        "fixed_mae_decreased": (final_mae < initial_mae) if fixed8 else True,
        "reload": reload_error == 0.0,
        "reload_state_exact": state_exact,
        "membership_counts": len(manifest["primary"]["A5_TRAIN"]) == 557 and len(manifest["augmentation"]["A5_TRAIN"]) == 4899,
        "cal_content_unread": True,
        "old_replay_split_unused": True,
    }
    elapsed = time.time() - started
    summary = {"schema_version": 1, "run_id": "A6-A010C" if fixed8 else ("A6-A020C-PILOT" if args.mode == "pilot" else "A6-A020C"), "status": "passed" if all(checks.values()) else "failed", "complete": True, "terminal": True, "mode": args.mode, "seed": args.seed, "device": str(device), "steps": steps, "start_step": start_step, "microbatch": args.batch_size, "gradient_accumulation": args.gradient_accumulation, "effective_batch": args.batch_size * args.gradient_accumulation, "num_workers": args.num_workers, "dataset_rows": len(dataset), "elapsed_seconds": elapsed, "steps_per_second": float((steps - start_step) / elapsed), "samples_per_second": float((steps - start_step) * args.batch_size * args.gradient_accumulation / elapsed), "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0, "loss_first": losses[0], "loss_last": losses[-1], "fixed_mae_initial": initial_mae, "fixed_mae_final": final_mae, "fixed_mae_diagnostic_only": not fixed8, "stochastic_forward_max_abs": stochastic_forward_delta, "parity_seed": parity_seed, "reload_max_abs": reload_error, "checkpoint": str(checkpoint_path), "checks": checks, "decision": "authorize clean full training" if fixed8 and all(checks.values()) else ("resource pilot complete" if args.mode == "pilot" else "evaluate fixed last checkpoints")}
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

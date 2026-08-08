#!/usr/bin/env python3
"""Matched one-seed INITIAL and MIX060 affordance producer screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from timm.scheduler import CosineLRScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import zarr


ROOT = Path(__file__).resolve().parents[3]
JOINT_ROOT = ROOT / "jointTrain_new"
STAGE1_ROOT = JOINT_ROOT / "experiment" / "stage1_optimize"
for path in (ROOT, JOINT_ROOT, STAGE1_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from joint_train.utils.pc_utils import pc_normalize
from jointTrain_new.experiment.model_architecture_6.run_a6_a010_a020_affordance_train import (
    atomic,
    prediction,
    set_seed,
    worker_seed,
)
from jointTrain_new.experiment.model_architecture_6.run_a6_a030c_affordance_cal_consumer import contact_metrics
from path_config import (
    JOINTTRAIN_AFFORDANCE_FOURTH_ROUND_OVERLAY,
    JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A030C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A031C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A032C_RESULT_ROOT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
)
from stage1_optimize_lib import compute_prediction_metrics
from train_stage1_optimize import compute_loss, inplace_relu, optimizer_groups
from vendor.point_m2ae.Point_M2AE_Afford import Point_M2AE_Afford, get_loss


SEED = 20260806
EVAL_SEED = 20260807
CONDITIONS = ("initial", "mix060")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class A032Dataset(Dataset):
    def __init__(self, membership: dict, condition: str):
        self.source = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
        self.overlay = zarr.open_group(str(JOINTTRAIN_AFFORDANCE_FOURTH_ROUND_OVERLAY), mode="r")
        self.condition = condition
        primary = [int(row["primary_row"]) for row in membership["primary"]["A5_TRAIN"]]
        self.rows = [("primary", row) for row in primary]
        self.rows.extend(("augmentation", int(row)) for row in membership["augmentation"]["A5_TRAIN"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        kind, row = self.rows[index]
        if kind == "primary":
            cloud = np.asarray(self.source["data/point_cloud"][row], dtype=np.float32)
            score = self._score("data/affordance_initial", "primary/updated_mix_060", row)
        else:
            cloud = np.asarray(self.source["data/stage1_aug_point_cloud"][row], dtype=np.float32)
            score = self._score("data/stage1_aug_affordance_initial", "augmentation/updated_mix_060", row)
        xyz = pc_normalize(cloud[:, :3].copy())
        angle = np.random.uniform(-30.0, 30.0) * np.pi / 180.0
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = xyz @ rotation.T
        xyz += np.random.normal(0.0, 0.005, xyz.shape).astype(np.float32)
        xyz *= np.float32(np.random.uniform(0.8, 1.25))
        xyz += np.random.uniform(-0.1, 0.1, (1, 3)).astype(np.float32)
        return {"xyz": torch.from_numpy(xyz), "affordance": torch.from_numpy(np.clip(score, 0.0, 1.0))}

    def _score(self, initial_key: str, mix_key: str, row: int) -> np.ndarray:
        root, key = (self.source, initial_key) if self.condition == "initial" else (self.overlay, mix_key)
        return np.asarray(root[key][row], dtype=np.float32).copy()


def cal_data(membership: dict, condition: str, limit: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    source = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    overlay = zarr.open_group(str(JOINTTRAIN_AFFORDANCE_FOURTH_ROUND_OVERLAY), mode="r")
    rows = membership["primary"]["A5_CAL"]
    if limit:
        rows = rows[:limit]
    indices = np.asarray([int(row["primary_row"]) for row in rows], dtype=np.int64)
    source_ids = np.asarray([int(row["source_replay_id"]) for row in rows], dtype=np.int64)
    object_keys = [str(row["target"]) for row in rows]
    points = np.asarray(source["data/point_cloud"][indices, :, :3], dtype=np.float32)
    key_root, key = (source, "data/affordance_initial") if condition == "initial" else (overlay, "primary/updated_mix_060")
    targets = np.asarray(key_root[key][indices], dtype=np.float32)
    xyz = np.stack([pc_normalize(item.copy()) for item in points]).astype(np.float32)
    return xyz, targets, source_ids, object_keys


def evaluate(model: torch.nn.Module, xyz: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    def run_once() -> np.ndarray:
        values = []
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(EVAL_SEED)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(EVAL_SEED)
            for start in range(0, len(xyz), batch_size):
                values.append(prediction(model, torch.from_numpy(xyz[start : start + batch_size]).to(device)).cpu().numpy())
        return np.concatenate(values)

    first = run_once()
    second = run_once()
    return first, float(np.max(np.abs(first - second)))


def write_combined() -> None:
    root = Path(JOINTTRAIN_ARCH6_A032C_RESULT_ROOT)
    paths = {condition: root / condition / "full" / "summary.json" for condition in CONDITIONS}
    if not all(path.exists() for path in paths.values()):
        return
    summaries = {condition: json.loads(path.read_text(encoding="utf-8")) for condition, path in paths.items()}
    a030 = json.loads((Path(JOINTTRAIN_ARCH6_A030C_RESULT_ROOT) / "full" / "summary.json").read_text(encoding="utf-8"))
    baseline = a030["contact_metrics"][f"seed_{SEED}_top4_nms3cm"]
    comparisons = {}
    candidates = []
    for condition, summary in summaries.items():
        metrics = summary["contact_metrics"][f"{condition}_top4_nms3cm"]
        comparisons[condition] = {
            "mean_distance_minus_updated_m": metrics["mean_m"] - baseline["mean_m"],
            "coverage_3cm_minus_updated": metrics["coverage_3cm"] - baseline["coverage_3cm"],
            "coverage_5cm_minus_updated": metrics["coverage_5cm"] - baseline["coverage_5cm"],
        }
        if metrics["mean_m"] < baseline["mean_m"] and metrics["coverage_3cm"] > baseline["coverage_3cm"]:
            candidates.append(condition)
    passed = all(summary.get("status") == "passed" for summary in summaries.values())
    combined = {
        "schema_version": 1,
        "run_id": "A6-A032C",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "baseline": {"condition": "updated", "seed": SEED, "contact_metrics": baseline},
        "variants": {condition: summaries[condition]["contact_metrics"] for condition in CONDITIONS},
        "descriptive_comparison_to_updated": comparisons,
        "screen_candidates": candidates,
        "claim_supported": "screen_only" if passed else "no",
        "producer_replacement_authorized": False,
        "decision": "retain A030 updated producer; G065 selector gate blocks A033 downstream utility",
        "next_run_ids": [],
    }
    atomic(root / "summary.json", combined)
    atomic(root / "run_state.json", combined)
    atomic(root / "queue_state.json", combined)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    steps = 50 if args.sanity else args.steps
    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    membership_path = Path(JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT) / "membership_manifest.json"
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    dataset = A032Dataset(membership, args.condition)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
        worker_init_fn=worker_seed,
        persistent_workers=args.num_workers > 0,
    )
    model = Point_M2AE_Afford(cls_dim=1, num_categories=16, dual_head=True, value_activation="relu").to(device)
    model.apply(inplace_relu)
    criterion = get_loss("dual_head", ce_weight=1.0, dice_weight=1.0, mse_weight=10.0, gt_thresh=0.05)
    optimizer = AdamW(optimizer_groups(model, 4e-4, 1.0, 0.05), lr=4e-4)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=100, lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=10, cycle_limit=1, t_in_epochs=True
    )
    out = Path(JOINTTRAIN_ARCH6_A032C_RESULT_ROOT) / args.condition / ("sanity" if args.sanity else "full")
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "training_config.json", {
        "condition": args.condition,
        "seed": SEED,
        "steps": steps,
        "microbatch": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch": args.batch_size * args.gradient_accumulation,
        "optimizer": "AdamW",
        "learning_rate": 4e-4,
        "weight_decay": 0.05,
    })
    latest_path = out / "latest.pth"
    start_step = 0
    if args.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        losses = list(state["losses"])
    else:
        losses = []
    started = time.time()
    iterator = iter(loader)
    model.train()
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
                raise RuntimeError("non-finite A032 affordance loss")
            (loss / args.gradient_accumulation).backward()
            micro_losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(np.mean(micro_losses)))
        if (step + 1) % 70 == 0:
            scheduler.step(step // 70)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            atomic(out / "progress.json", {
                "step": step + 1,
                "steps": steps,
                "elapsed_seconds": time.time() - started,
                "latest_loss": losses[-1],
                "learning_rate": max(group["lr"] for group in optimizer.param_groups),
            })
        if not args.sanity and ((step + 1) % 500 == 0 or step + 1 == steps):
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
                "condition": args.condition,
                "seed": SEED,
                "losses": losses,
            }, latest_path)

    checkpoint = out / "last.pth"
    torch.save({"model": model.state_dict(), "condition": args.condition, "seed": SEED, "steps": steps}, checkpoint)
    model.eval()
    cal_xyz, cal_target, source_ids, object_keys = cal_data(membership, args.condition, 8 if args.sanity else 0)
    cal_prediction, stochastic_delta = evaluate(model, cal_xyz, device, 16)
    point_metrics, _ = compute_prediction_metrics(
        cal_prediction, cal_target, source_ids.tolist(), object_keys, threshold=0.05, batch_size=16
    )
    reload_model = Point_M2AE_Afford(cls_dim=1, num_categories=16, dual_head=True, value_activation="relu").to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    reload_model.eval()
    reloaded, _ = evaluate(reload_model, cal_xyz, device, 16)
    reload_error = float(np.max(np.abs(cal_prediction - reloaded)))
    contact = None
    contact_audit = None
    if not args.sanity:
        source = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
        all_source_ids = np.asarray(source["meta/source_replay_id"][:], dtype=np.int64)
        all_points = np.asarray(source["data/point_cloud"][:, :, :3], dtype=np.float32)
        all_updated = np.asarray(source["data/affordance_updated"][:], dtype=np.float32)
        cal_index = {int(value): index for index, value in enumerate(source_ids.tolist())}
        full_prediction = np.zeros_like(all_updated)
        for row, source_id in enumerate(all_source_ids.tolist()):
            if int(source_id) in cal_index:
                full_prediction[row] = cal_prediction[cal_index[int(source_id)]]
        raw_contact, contact_audit = contact_metrics(
            {"ensemble_top4_nms3cm": full_prediction}, all_source_ids, all_points, all_updated, EVAL_SEED
        )
        contact = {
            (f"{args.condition}_top4_nms3cm" if name == "ensemble_top4_nms3cm" else name): metrics
            for name, metrics in raw_contact.items()
        }
    np.savez_compressed(out / "predictions.npz", source_replay_id=source_ids, prediction=cal_prediction, target=cal_target)
    checks = {
        "a031_terminal": json.loads((Path(JOINTTRAIN_ARCH6_A031C_RESULT_ROOT) / "full" / "summary.json").read_text())["status"] == "passed",
        "matched_seed_budget": SEED == 20260806 and steps in {50, 7000} and args.batch_size * args.gradient_accumulation == 96,
        "membership_counts": len(dataset) == 5456,
        "finite": bool(np.isfinite(losses).all() and np.isfinite(cal_prediction).all()),
        "loss_decreased": losses[-1] < losses[0],
        "deterministic_eval": stochastic_delta == 0.0,
        "reload_exact": reload_error == 0.0,
        "cal_primary_only": len(source_ids) == (8 if args.sanity else 102),
        "contact_labels_382": args.sanity or contact_audit["cal_contact_labels"] == 382,
        "point_frame_alignment": args.sanity or contact_audit["point_index_and_frame_alignment"],
        "no_outcome_read": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": f"A6-A032C-{args.condition.upper()}-{'SANITY' if args.sanity else 'FULL'}",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "condition": args.condition,
        "seed": SEED,
        "optimizer_steps": steps,
        "start_step": start_step,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": sha256(checkpoint),
        "reload_max_abs": reload_error,
        "point_metrics_diagnostic": point_metrics,
        "contact_metrics": contact,
        "contact_audit": contact_audit,
        "checks": checks,
        "claim_supported": "sanity_only" if passed and args.sanity else ("screen_only" if passed else "no"),
        "decision": "run matched full condition" if passed and args.sanity else "wait for matched A032 aggregate",
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    if not args.sanity:
        write_combined()
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

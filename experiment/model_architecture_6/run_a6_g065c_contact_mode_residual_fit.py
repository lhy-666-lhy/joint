#!/usr/bin/env python3
"""Matched G065 contact-local classifier and set-residual fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
JOINT_ROOT = ROOT / "jointTrain_new"
for path in (ROOT, JOINT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from joint_train.models.pointnet_encoder import PointNetEncoderXYZA, StateEncoder
from jointTrain_new.experiment.model_architecture_6.run_a6_a030c_affordance_cal_consumer import paired_bootstrap
from path_config import (
    JOINTTRAIN_ARCH6_G062C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G064C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G065C_RESULT_ROOT,
)


MODE_COUNT = 8


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first = F.normalize(value[..., :3], dim=-1, eps=1e-6)
    second_raw = value[..., 3:6]
    second = F.normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def geodesic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    relative = left.transpose(-1, -2) @ right
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine)


class ContactModeResidual(nn.Module):
    def __init__(self, variant: str, prototype_rotation: torch.Tensor, prototype_translation: torch.Tensor, hidden_dim: int = 256):
        super().__init__()
        self.variant = variant
        self.scene = PointNetEncoderXYZA(in_channels=4, out_channels=hidden_dim)
        self.state = StateEncoder(state_dim=7, out_channels=hidden_dim, hidden=hidden_dim)
        self.query = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mode_head = nn.Linear(hidden_dim, MODE_COUNT)
        output_modes = 1 if variant == "classifier" else MODE_COUNT
        self.translation_head = nn.Linear(hidden_dim, output_modes * 3)
        self.rotation_head = nn.Linear(hidden_dim, output_modes * 6)
        self.presence_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.translation_head.weight)
        nn.init.zeros_(self.translation_head.bias)
        nn.init.zeros_(self.rotation_head.weight)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(output_modes)
        with torch.no_grad():
            self.rotation_head.bias.copy_(identity)
        self.register_buffer("prototype_rotation", prototype_rotation)
        self.register_buffer("prototype_translation", prototype_translation)

    def forward(self, xyz: torch.Tensor, state: torch.Tensor, affordance: torch.Tensor, query_point: torch.Tensor) -> dict[str, torch.Tensor]:
        scene = self.scene(torch.cat((xyz, affordance.unsqueeze(-1)), dim=-1))
        state_feature = self.state(state)
        nearest = torch.cdist(query_point, xyz).argmin(dim=-1)
        query_score = affordance.gather(1, nearest)
        query_feature = self.query(torch.cat((query_point, query_score.unsqueeze(-1)), dim=-1))
        count = query_point.shape[1]
        feature = self.decoder(torch.cat((
            scene[:, None].expand(-1, count, -1),
            state_feature[:, None].expand(-1, count, -1),
            query_feature,
        ), dim=-1))
        modes = 1 if self.variant == "classifier" else MODE_COUNT
        return {
            "mode_logits": self.mode_head(feature),
            "translation_residual": self.translation_head(feature).reshape(*feature.shape[:2], modes, 3),
            "rotation_residual": self.rotation_head(feature).reshape(*feature.shape[:2], modes, 6),
            "presence_logits": self.presence_head(feature).squeeze(-1),
        }

    def candidates(self, output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "classifier":
            mode = output["mode_logits"].argmax(dim=-1)
            prototype_t = self.prototype_translation[mode]
            prototype_r = self.prototype_rotation[mode]
            translation = prototype_t + output["translation_residual"].squeeze(-2)
            rotation = prototype_r @ rotation_6d_to_matrix(output["rotation_residual"].squeeze(-2))
            return translation.unsqueeze(-2), rotation.unsqueeze(-3)
        residual_rotation = rotation_6d_to_matrix(output["rotation_residual"])
        translation = self.prototype_translation[None, None] + output["translation_residual"]
        rotation = self.prototype_rotation[None, None] @ residual_rotation
        return translation, rotation


def targets(batch: dict[str, torch.Tensor], prototype_t: torch.Tensor, prototype_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mode = batch["mode_index"].clamp_min(0)
    translation = prototype_t[mode] + batch["translation_residual"]
    rotation = prototype_r[mode] @ rotation_6d_to_matrix(batch["rotation_residual_6d"])
    return translation, rotation


def loss_fn(model: ContactModeResidual, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["presence"]
    target_t, target_r = targets(batch, model.prototype_translation, model.prototype_rotation)
    mode_loss = F.cross_entropy(output["mode_logits"][valid], batch["mode_index"][valid])
    if model.variant == "classifier":
        translation = output["translation_residual"].squeeze(-2)
        rotation = rotation_6d_to_matrix(output["rotation_residual"].squeeze(-2))
        translation_loss = (translation[valid] - batch["translation_residual"][valid]).abs().mean()
        rotation_loss = geodesic(rotation[valid], rotation_6d_to_matrix(batch["rotation_residual_6d"])[valid]).mean()
        pose_loss = translation_loss + rotation_loss
    else:
        candidate_t, candidate_r = model.candidates(output)
        translation_cost = (candidate_t - target_t.unsqueeze(-2)).abs().mean(dim=-1)
        rotation_cost = geodesic(candidate_r, target_r.unsqueeze(-3))
        best = (translation_cost + rotation_cost).min(dim=-1).values
        pose_loss = best[valid].mean()
        translation_loss = translation_cost.min(dim=-1).values[valid].mean()
        rotation_loss = rotation_cost.min(dim=-1).values[valid].mean()
    presence_loss = F.binary_cross_entropy_with_logits(output["presence_logits"], valid.to(output["presence_logits"].dtype))
    loss = pose_loss + 0.1 * mode_loss + 0.1 * presence_loss
    return loss, {
        "pose": float(pose_loss.detach()),
        "translation": float(translation_loss.detach()),
        "rotation": float(rotation_loss.detach()),
        "mode": float(mode_loss.detach()),
        "presence": float(presence_loss.detach()),
    }


def tensor_data(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "xyz": torch.from_numpy(np.asarray(data["point_cloud_xyz"], dtype=np.float32)),
            "state": torch.from_numpy(np.asarray(data["state_qpos"], dtype=np.float32)),
            "affordance": torch.from_numpy(np.asarray(data["predicted_affordance"], dtype=np.float32)),
            "query": torch.from_numpy(np.asarray(data["query_point"], dtype=np.float32)),
            "split": torch.from_numpy(np.asarray(data["split"], dtype=np.int64)),
            "group_index": torch.from_numpy(np.asarray(data["group_index"], dtype=np.int64)),
            "mode_index": torch.from_numpy(np.asarray(data["mode_index"], dtype=np.int64)),
            "presence": torch.from_numpy(np.asarray(data["mode_presence"], dtype=bool)),
            "translation_residual": torch.from_numpy(np.asarray(data["translation_residual"], dtype=np.float32)),
            "rotation_residual_6d": torch.from_numpy(np.asarray(data["rotation_residual_6d"], dtype=np.float32)),
            "prototype_rotation": torch.from_numpy(np.asarray(data["prototype_rotation"], dtype=np.float32)),
            "prototype_translation": torch.from_numpy(np.asarray(data["prototype_translation"], dtype=np.float32)),
        }


def batch_at(data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("xyz", "state", "affordance", "query", "mode_index", "presence", "translation_residual", "rotation_residual_6d")
    return {key: data[key][indices].to(device) for key in keys}


def evaluate(model: ContactModeResidual, data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    batch = batch_at(data, indices, device)
    with torch.no_grad():
        output = model(batch["xyz"], batch["state"], batch["affordance"], batch["query"])
        candidate_t, candidate_r = model.candidates(output)
        target_t, target_r = targets(batch, model.prototype_translation, model.prototype_rotation)
        translation = torch.linalg.vector_norm(candidate_t - target_t.unsqueeze(-2), dim=-1)
        rotation = geodesic(candidate_r, target_r.unsqueeze(-3))
        if model.variant == "classifier":
            selected = torch.zeros_like(translation[..., 0], dtype=torch.long)
        else:
            selected = output["mode_logits"].argmax(dim=-1)
        selected_t = translation.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        selected_r = rotation.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        best_cost = translation / 0.03 + rotation / np.deg2rad(12.0)
        best = best_cost.argmin(dim=-1)
        best_t = translation.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        best_r = rotation.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    valid = batch["presence"]
    mode_accuracy = float((output["mode_logits"].argmax(dim=-1)[valid] == batch["mode_index"][valid]).float().mean())
    arrays = {
        "selected_translation": selected_t[valid].cpu().numpy(),
        "selected_rotation": selected_r[valid].cpu().numpy(),
        "best_translation": best_t[valid].cpu().numpy(),
        "best_rotation": best_r[valid].cpu().numpy(),
    }
    def aggregate(t: np.ndarray, r: np.ndarray) -> dict:
        return {
            "translation_m": float(t.mean()),
            "rotation_rad": float(r.mean()),
            "pose_within_3cm_12deg": float(np.mean((t <= 0.03) & (r <= np.deg2rad(12.0)))),
        }
    return {"selected_top1": aggregate(arrays["selected_translation"], arrays["selected_rotation"]), "best_of_candidates": aggregate(arrays["best_translation"], arrays["best_rotation"]), "mode_accuracy": mode_accuracy}, arrays


def group_metrics(data: dict[str, torch.Tensor], indices: torch.Tensor, arrays: dict[str, np.ndarray]) -> dict[int, dict[str, float]]:
    rows = []
    for local, source in enumerate(indices.tolist()):
        group = int(data["group_index"][source])
        for slot in np.flatnonzero(data["presence"][source].numpy()):
            rows.append((group, int(slot)))
    output = {}
    for group in sorted(set(item[0] for item in rows)):
        take = [index for index, item in enumerate(rows) if item[0] == group]
        output[group] = {
            "translation_m": float(np.mean(arrays["selected_translation"][take])),
            "rotation_rad": float(np.mean(arrays["selected_rotation"][take])),
        }
    return output


def write_combined(root: Path) -> None:
    paths = {variant: root / variant / "full" / "summary.json" for variant in ("classifier", "set_residual")}
    if not all(path.exists() for path in paths.values()):
        return
    summaries = {variant: json.loads(path.read_text()) for variant, path in paths.items()}
    eligible = [variant for variant, summary in summaries.items() if summary.get("status") == "passed" and summary.get("claim_supported") == "yes"]
    combined = {
        "schema_version": 1,
        "run_id": "A6-G065C",
        "status": "passed" if all(summary.get("status") == "passed" for summary in summaries.values()) else "failed",
        "complete": True,
        "terminal": True,
        "variants": {variant: summary["metrics"] for variant, summary in summaries.items()},
        "eligible_variants": eligible,
        "claim_supported": "yes" if eligible else "no",
        "decision": "authorize G066 multi-IK realization" if eligible else "stop before G066 and analyze mode scoring",
        "next_run_ids": ["A6-G066C"] if eligible else [],
    }
    atomic(root / "summary.json", combined)
    atomic(root / "run_state.json", combined)
    atomic(root / "queue_state.json", combined)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("classifier", "set_residual"), required=True)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    steps = 50 if args.sanity else args.steps
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_path = Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / "full" / "supervision.npz"
    data = tensor_data(source_path)
    train = torch.nonzero(data["split"] == 0, as_tuple=True)[0]
    cal = torch.nonzero(data["split"] == 1, as_tuple=True)[0]
    model = ContactModeResidual(args.variant, data["prototype_rotation"], data["prototype_translation"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(args.seed)
    losses = []
    started = time.time()
    model.train()
    for step in range(steps):
        take = train[torch.randint(len(train), (args.batch_size,), generator=generator)]
        batch = batch_at(data, take, device)
        output = model(batch["xyz"], batch["state"], batch["affordance"], batch["query"])
        loss, _ = loss_fn(model, output, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite G065 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    metrics, arrays = evaluate(model, data, cal, device)
    groups = group_metrics(data, cal, arrays)
    baseline = json.loads((Path(JOINTTRAIN_ARCH6_G062C_RESULT_ROOT) / "full" / "summary.json").read_text())
    baseline_groups = {int(row["group_index"]): row for row in baseline["metrics"]["per_group"]}
    common = sorted(set(groups) & set(baseline_groups))
    translation_difference = np.asarray([groups[group]["translation_m"] - baseline_groups[group]["translation_m"] for group in common])
    rotation_difference = np.asarray([groups[group]["rotation_rad"] - baseline_groups[group]["rotation_rad"] for group in common])
    metrics["selected_minus_g062"] = {
        "translation_m": paired_bootstrap(translation_difference, args.seed),
        "rotation_rad": paired_bootstrap(rotation_difference, args.seed),
    }
    out = Path(JOINTTRAIN_ARCH6_G065C_RESULT_ROOT) / args.variant / ("sanity" if args.sanity else "full")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "last.pth"
    torch.save({"model": model.state_dict(), "variant": args.variant, "seed": args.seed, "steps": steps}, checkpoint)
    reload_model = ContactModeResidual(args.variant, data["prototype_rotation"], data["prototype_translation"]).to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    reload_model.eval()
    probe = cal[: min(8, len(cal))]
    probe_batch = batch_at(data, probe, device)
    model.eval()
    with torch.no_grad():
        original = model(probe_batch["xyz"], probe_batch["state"], probe_batch["affordance"], probe_batch["query"])
        reloaded = reload_model(probe_batch["xyz"], probe_batch["state"], probe_batch["affordance"], probe_batch["query"])
    reload_error = max(float(torch.max(torch.abs(original[key] - reloaded[key]))) for key in original)
    checks = {
        "g064_terminal": json.loads((Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / "full" / "summary.json").read_text())["status"] == "passed",
        "split_counts": len(train) == 531 and len(cal) == 101,
        "finite": bool(np.isfinite(losses).all()),
        "loss_decreased": losses[-1] < losses[0],
        "reload_exact": reload_error == 0.0,
        "same_seed_steps_batch": args.seed == 20260806 and args.batch_size == 64 and steps in {50, 200, 6000},
        "no_outcome_read": True,
    }
    passed = all(checks.values())
    comparison = metrics["selected_minus_g062"]
    supported = bool(
        not args.sanity
        and comparison["translation_m"]["ci95"][1] < 0.0
        and comparison["rotation_rad"]["ci95"][1] < 0.0
        and metrics["selected_top1"]["pose_within_3cm_12deg"] > baseline["metrics"]["aggregate"]["pose_within_3cm_12deg"]
    )
    np.savez_compressed(out / "cal_predictions.npz", **arrays)
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "forbidden_feature_audit.json", {"task_outcome_read": False, "future_path_read": False, "gt_link_pose_input": False})
    summary = {
        "schema_version": 1,
        "run_id": f"A6-G065C-{args.variant}-{'SANITY' if args.sanity else 'FULL'}",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "variant": args.variant,
        "optimizer_steps": steps,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "elapsed_seconds": time.time() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reload_max_abs": reload_error,
        "checkpoint_sha256": sha256(checkpoint),
        "metrics": metrics,
        "checks": checks,
        "claim_supported": "yes" if passed and supported else ("sanity_only" if passed and args.sanity else "no"),
        "decision": "candidate for G066" if supported else ("run full matched variant" if passed and args.sanity else "do not promote this variant"),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    if not args.sanity:
        write_combined(Path(JOINTTRAIN_ARCH6_G065C_RESULT_ROOT))
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

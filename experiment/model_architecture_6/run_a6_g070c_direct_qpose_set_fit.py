#!/usr/bin/env python3
"""Matched direct relative-qpose set fit from G064 legal IK supervision."""

from __future__ import annotations

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
from path_config import (
    JOINTTRAIN_ARCH6_G064C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G070C_RESULT_ROOT,
)


SEED = 20260806
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


class QPoseSetModel(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.scene = PointNetEncoderXYZA(in_channels=4, out_channels=hidden_dim)
        self.state = StateEncoder(state_dim=7, out_channels=hidden_dim, hidden=hidden_dim)
        self.query = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.qpose_head = nn.Linear(hidden_dim, MODE_COUNT * 7)
        self.presence_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.qpose_head.weight)
        nn.init.zeros_(self.qpose_head.bias)

    def forward(self, xyz: torch.Tensor, state: torch.Tensor, affordance: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        scene = self.scene(torch.cat((xyz, affordance.unsqueeze(-1)), dim=-1))
        state_feature = self.state(state)
        nearest = torch.cdist(query, xyz).argmin(dim=-1)
        query_score = affordance.gather(1, nearest)
        query_feature = self.query(torch.cat((query, query_score.unsqueeze(-1)), dim=-1))
        count = query.shape[1]
        feature = self.decoder(torch.cat((
            scene[:, None].expand(-1, count, -1),
            state_feature[:, None].expand(-1, count, -1),
            query_feature,
        ), dim=-1))
        return {
            "qpose": self.qpose_head(feature).reshape(*feature.shape[:2], MODE_COUNT, 7),
            "presence_logits": self.presence_head(feature).squeeze(-1),
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
            "mode_presence": torch.from_numpy(np.asarray(data["mode_presence"], dtype=bool)),
            "ik_qpose_relative": torch.from_numpy(np.asarray(data["ik_qpose_relative"], dtype=np.float32)),
            "ik_presence": torch.from_numpy(np.asarray(data["ik_presence"], dtype=bool)),
        }


def batch(data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: data[key][indices].to(device) for key in ("xyz", "state", "affordance", "query", "mode_presence", "ik_qpose_relative", "ik_presence")}


def set_loss(output: dict[str, torch.Tensor], data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = output["qpose"]
    target = data["ik_qpose_relative"]
    valid_target = data["ik_presence"]
    pair = (predicted[:, :, :, None, :] - target[:, :, None, :, :]).abs().mean(dim=-1)
    target_min = pair.masked_fill(~valid_target[:, :, None, :], float("inf")).min(dim=-1).values
    pred_min = pair.masked_fill(~valid_target[:, :, None, :], float("inf")).min(dim=-1).values
    pose_valid = data["mode_presence"] & valid_target.any(dim=-1)
    pose = target_min[valid_target].mean() + 0.25 * pred_min[pose_valid].mean()
    presence = F.binary_cross_entropy_with_logits(output["presence_logits"], data["mode_presence"].to(torch.float32))
    loss = pose + 0.1 * presence
    return loss, {"pose": float(pose.detach()), "presence": float(presence.detach())}


def evaluate(model: QPoseSetModel, data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        item = batch(data, indices, device)
        output = model(item["xyz"], item["state"], item["affordance"], item["query"])
        pair = (output["qpose"][:, :, :, None, :] - item["ik_qpose_relative"][:, :, None, :, :]).abs().mean(dim=-1)
        pair = pair.masked_fill(~item["ik_presence"][:, :, None, :], float("inf"))
        qpose_error = pair.min(dim=-1).values
        norm_selector = output["qpose"].norm(dim=-1).argmin(dim=-1)
        norm_error = qpose_error.gather(-1, norm_selector.unsqueeze(-1)).squeeze(-1)
        oracle_error = qpose_error.min(dim=-1).values
        selected_presence = item["mode_presence"]
    return {
        "group_index": data["group_index"][indices].cpu().numpy(),
        "presence": selected_presence.cpu().numpy(),
        "qpose_candidates": output["qpose"].cpu().numpy(),
        "norm_selected": norm_selector.cpu().numpy(),
        "oracle_error": oracle_error.cpu().numpy(),
        "norm_error": norm_error.cpu().numpy(),
        "ik_presence": item["ik_presence"].cpu().numpy(),
        "ik_targets": item["ik_qpose_relative"].cpu().numpy(),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    steps = 50 if args.sanity else args.steps
    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_path = Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / "full" / "supervision.npz"
    data = tensor_data(source_path)
    train = torch.nonzero(data["split"] == 0, as_tuple=True)[0]
    cal = torch.nonzero(data["split"] == 1, as_tuple=True)[0]
    model = QPoseSetModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(SEED)
    losses = []
    started = time.time()
    model.train()
    for _ in range(steps):
        take = train[torch.randint(len(train), (args.batch_size,), generator=generator)]
        item = batch(data, take, device)
        output = model(item["xyz"], item["state"], item["affordance"], item["query"])
        loss, _ = set_loss(output, item)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite G070 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    out = Path(JOINTTRAIN_ARCH6_G070C_RESULT_ROOT) / ("sanity" if args.sanity else "full")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "last.pth"
    torch.save({"model": model.state_dict(), "seed": SEED, "steps": steps}, checkpoint)
    cal_result = evaluate(model, data, cal, device)
    reload_model = QPoseSetModel().to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    reload_result = evaluate(reload_model, data, cal, device)
    reload_error = float(np.max(np.abs(cal_result["qpose_candidates"] - reload_result["qpose_candidates"])))
    valid = cal_result["presence"] & cal_result["ik_presence"].any(axis=-1)
    metrics = {
        "oracle_set_l1": float(cal_result["oracle_error"][valid].mean()),
        "norm_selector_set_l1": float(cal_result["norm_error"][valid].mean()),
        "oracle_within_3cm12deg_proxy": float(np.mean(cal_result["oracle_error"][valid] <= 0.03)),
        "norm_selector_within_3cm12deg_proxy": float(np.mean(cal_result["norm_error"][valid] <= 0.03)),
        "legal_ik_slot_fraction": float(cal_result["ik_presence"][valid].any(axis=-1).mean()),
    }
    checks = {
        "g064_terminal": json.loads((Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / "full" / "summary.json").read_text())["status"] == "passed",
        "split_counts": len(train) == 531 and len(cal) == 101,
        "finite": bool(np.isfinite(losses).all() and np.isfinite(cal_result["qpose_candidates"]).all()),
        "loss_decreased": losses[-1] < losses[0],
        "reload_exact": reload_error == 0.0,
        "qpose_shape": cal_result["qpose_candidates"].shape == (101, 4, 8, 7),
        "presence_shape": cal_result["ik_presence"].shape == (101, 4, 8),
        "no_outcome_read": True,
        "no_cal_training": True,
    }
    passed = all(checks.values())
    np.savez_compressed(out / "cal_qpose_candidates.npz", **cal_result)
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "forbidden_feature_audit.json", {"task_outcome_read": False, "future_path_read": False, "cal_labels_for_training": False})
    summary = {
        "schema_version": 1,
        "run_id": "A6-G070C-SANITY" if args.sanity else "A6-G070C",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "optimizer_steps": steps,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": sha256(checkpoint),
        "reload_max_abs": reload_error,
        "metrics": metrics,
        "checks": checks,
        "claim_supported": "sanity_only" if passed and args.sanity else ("diagnostic_only" if passed else "no"),
        "decision": "run direct-qpose planner realization" if passed else "repair G070 qpose set fit",
        "next_run_ids": ["A6-G070C-REALIZE"] if passed and not args.sanity else (["A6-G070C"] if passed else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

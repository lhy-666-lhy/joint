#!/usr/bin/env python3
"""A6-O010 single-GPU fixed-64 memorization for the MLP baseline."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010_RESULT_ROOT,
)


STEPS = 2000
SEED = 20260805
STATE_DIM = 81


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OperationMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.state_mlp = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 32 * 9))

    def forward(self, points: torch.Tensor, target_mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        point_input = torch.cat([points, target_mask.unsqueeze(-1).to(points.dtype)], dim=-1)
        scene = self.point_mlp(point_input).amax(dim=1)
        return self.head(torch.cat([scene, self.state_mlp(state)], dim=-1)).reshape(-1, 32, 9)


def state_at_anchor(data: np.lib.npyio.NpzFile, anchor: int) -> np.ndarray:
    qpos = np.asarray(data["actual_joint_qpos"], dtype=np.float32)
    command = np.asarray(data["joint_command_qpos"], dtype=np.float32)
    indices = np.clip(np.arange(anchor - 3, anchor + 1), 0, qpos.shape[0] - 1)
    history = qpos[indices]
    previous = qpos[np.maximum(indices - 1, 0)]
    qvel_per_tick = history - previous
    return np.concatenate([history.reshape(-1), qvel_per_tick.reshape(-1), command[anchor, :9]], axis=0).astype(np.float32)


def load_fixed_batch() -> tuple[dict[str, torch.Tensor], dict]:
    d020 = Path(JOINTTRAIN_ARCH6_D020_RESULT_ROOT)
    d030 = Path(JOINTTRAIN_ARCH6_D030_RESULT_ROOT)
    labels = np.load(d020 / "fixed_batch.npz")
    rows = json.loads((d020 / "fixed_batch_manifest.json").read_text(encoding="utf-8"))["rows"]
    normalizer = json.loads((d020 / "normalizer.json").read_text(encoding="utf-8"))
    point_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    trajectory_cache: dict[str, dict[str, np.ndarray]] = {}
    materialized_cache: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        target = str(row["target"])
        if target not in materialized_cache:
            with np.load(d030 / "materialized" / f"{target.replace('/', '_')}.npz") as data:
                materialized_cache[target] = {name: np.asarray(data[name]) for name in ("point_cloud", "target_mask", "raw_index")}
        materialized = materialized_cache[target]
        matches = np.flatnonzero(materialized["raw_index"] == int(row["anchor_raw_index"]))
        if matches.size != 1:
            raise ValueError(f"anchor join mismatch: {target} {row['anchor_raw_index']}")
        point_rows.append(materialized["point_cloud"][matches[0]])
        mask_rows.append(materialized["target_mask"][matches[0]])
        relative = str(row["trajectory_relative_path"])
        if relative not in trajectory_cache:
            with np.load(Path(ARTICU_COLLECTION_ROOT) / relative, allow_pickle=False) as data:
                trajectory_cache[relative] = {"actual_joint_qpos": np.asarray(data["actual_joint_qpos"]), "joint_command_qpos": np.asarray(data["joint_command_qpos"])}
        state_rows.append(state_at_anchor(trajectory_cache[relative], int(row["anchor_raw_index"])))
    mean = np.asarray(normalizer["mean"], dtype=np.float32).reshape(1, 1, 9)
    std = np.asarray(normalizer["std"], dtype=np.float32).reshape(1, 1, 9)
    target = (np.asarray(labels["action"], dtype=np.float32) - mean) / std
    batch = {"points": torch.from_numpy(np.stack(point_rows).astype(np.float32)), "target_mask": torch.from_numpy(np.stack(mask_rows).astype(bool)), "state": torch.from_numpy(np.stack(state_rows).astype(np.float32)), "target": torch.from_numpy(target.astype(np.float32)), "valid": torch.from_numpy(np.asarray(labels["valid_mask"], dtype=bool))}
    lineage = {"fixed_batch_sha256": sha256_file(d020 / "fixed_batch.npz"), "fixed_manifest_sha256": sha256_file(d020 / "fixed_batch_manifest.json"), "normalizer_sha256": sha256_file(d020 / "normalizer.json"), "materialization_manifest_sha256": sha256_file(d030 / "materialization_manifest.json"), "qvel_source": "actual_joint_qpos finite difference per raw tick", "result_json_read": False, "affordance_channel": "ZERO"}
    return batch, lineage


def masked_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = valid.unsqueeze(-1).to(prediction.dtype)
    denominator = mask.expand_as(prediction).sum().clamp_min(1.0)
    mse = (torch.square(prediction - target) * mask).sum() / denominator
    mae = (torch.abs(prediction - target) * mask).sum() / denominator
    return mse, mae


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O010_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("O010 requires one GPU")
    batch, lineage = load_fixed_batch()
    batch = {key: value.to(device) for key, value in batch.items()}
    atomic_json(out_dir / "run_manifest.json", {"schema_version": 1, "run_id": "a6_o010_mlp_fixed64_v1", "model": "O-MLP-ABS", "steps": STEPS, "seed": SEED, "batch_size": 64, "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 1e-6}, "checkpoint_rule": "last_step_2000_only", "lineage": lineage})
    probe = OperationMLP().to(device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=1e-6)
    torch.cuda.reset_peak_memory_stats(device)
    probe_start = time.perf_counter()
    for _ in range(10):
        probe_optimizer.zero_grad(set_to_none=True)
        probe_loss, _ = masked_metrics(probe(batch["points"], batch["target_mask"], batch["state"]), batch["target"], batch["valid"])
        probe_loss.backward()
        probe_optimizer.step()
    probe_wall = time.perf_counter() - probe_start
    probe_passed = bool(np.isfinite(probe_wall) and probe_wall < 60.0)
    atomic_json(Path("experiment_loop/resource_probe.json"), {"schema_version": 1, "iteration_id": "a6_o010_mlp_fixed64_v1", "steps": 10, "batch_size": 64, "wall_seconds": probe_wall, "steps_per_second": 10.0 / probe_wall, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "passed": probe_passed})
    del probe, probe_optimizer
    torch.cuda.empty_cache()
    if not probe_passed:
        summary = {"schema_version": 1, "run_id": "a6_o010_mlp_fixed64_v1", "complete": True, "terminal": True, "status": "failed", "failure_class": "implementation_failure", "claim_supported": "no", "decision": "O010 bounded GPU probe failed; full fit not started.", "next_run_ids": [], "event_id": "a6_o010_mlp_fixed64_v1_terminal"}
        atomic_json(out_dir / "summary.json", summary)
        return 2
    torch.manual_seed(SEED)
    model = OperationMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    history: list[dict] = []
    with torch.no_grad():
        initial_prediction = model(batch["points"], batch["target_mask"], batch["state"])
        initial_loss, initial_mae = masked_metrics(initial_prediction, batch["target"], batch["valid"])
    gradients_finite = True
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["points"], batch["target_mask"], batch["state"])
        loss, mae = masked_metrics(prediction, batch["target"], batch["valid"])
        loss.backward()
        gradients_finite = gradients_finite and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        optimizer.step()
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "loss": float(loss.detach()), "mae": float(mae.detach())})
    wall = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_prediction = model(batch["points"], batch["target_mask"], batch["state"])
        final_loss, final_mae = masked_metrics(final_prediction, batch["target"], batch["valid"])
        error = torch.abs(final_prediction - batch["target"])
        mask = batch["valid"].unsqueeze(-1).expand_as(error)
        per_dim_mae = [(error[..., dim][mask[..., dim]]).mean().item() for dim in range(9)]
    checkpoint = out_dir / "last.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": STEPS, "seed": SEED, "lineage": lineage}, checkpoint)
    reloaded = OperationMLP().to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device)["model"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        reload_error = float(torch.max(torch.abs(reloaded(batch["points"], batch["target_mask"], batch["state"]) - final_prediction)))
    ratio = float(initial_loss / final_loss.clamp_min(1e-12))
    checks = {"fixed_batch_64": batch["target"].shape == (64, 32, 9), "steps_exact_2000": history[-1]["step"] == 2000, "normalized_mae_le_1e_3": float(final_mae) <= 1e-3, "loss_decrease_ge_100x": ratio >= 100.0, "all_dimensions_finite": bool(np.isfinite(per_dim_mae).all()), "gradients_finite": gradients_finite, "strict_reload_max_error_le_1e_6": reload_error <= 1e-6, "zero_affordance": True, "zero_heldout_or_outcome_reads": True}
    passed = all(checks.values())
    atomic_json(out_dir / "history.json", {"schema_version": 1, "history": history})
    summary = {"schema_version": 1, "run_id": "a6_o010_mlp_fixed64_v1", "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "training_fit_failure", "claim_supported": "yes" if passed else "no", "evidence": {"manifest": "run_manifest.json", "checkpoint": "last.pt", "history": "history.json", "checkpoint_sha256": sha256_file(checkpoint)}, "metrics": {"initial_loss": float(initial_loss), "final_loss": float(final_loss), "initial_mae": float(initial_mae), "final_mae": float(final_mae), "loss_decrease_ratio": ratio, "per_dim_mae": per_dim_mae, "reload_max_error": reload_error, "wall_seconds": wall, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))}, "checks": checks, "decision": "O010 fixed-batch memorization passes." if passed else "O010 fixed-batch memorization fails; inspect fit before DYN64.", "remaining_work": ["A6-O020 PAR fixed64", "A6-O030 CAUSAL fixed64"], "next_run_ids": ["a6_o020_par_fixed64_v1", "a6_o030_causal_fixed64_v1"], "event_id": "a6_o010_mlp_fixed64_v1_terminal"}
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-O010", "status": summary["status"]}]})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

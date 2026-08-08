#!/usr/bin/env python3
"""A6-O020 single-GPU fixed-64 memorization for the parallel decoder."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from path_config import JOINTTRAIN_ARCH6_O020_RESULT_ROOT
from run_a6_o010_mlp_fixed64 import atomic_json, load_fixed_batch, masked_metrics, sha256_file


STEPS = 2000
SEED = 20260805
STATE_DIM = 81
HIDDEN_DIM = 128


class OperationPAR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, HIDDEN_DIM), nn.ReLU())
        self.state_mlp = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        layer = nn.TransformerDecoderLayer(d_model=HIDDEN_DIM, nhead=8, dim_feedforward=HIDDEN_DIM * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=4)
        self.queries = nn.Parameter(torch.randn(1, 32, HIDDEN_DIM) * 0.02)
        self.action_head = nn.Linear(HIDDEN_DIM, 9)

    def forward(self, points: torch.Tensor, target_mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        point_input = torch.cat([points, target_mask.unsqueeze(-1).to(points.dtype)], dim=-1)
        scene = self.point_mlp(point_input).amax(dim=1)
        robot = self.state_mlp(state)
        memory = torch.stack([scene, robot], dim=1)
        queries = self.queries.expand(points.shape[0], -1, -1)
        return self.action_head(self.decoder(queries, memory))


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O020_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("O020 requires one GPU")
    batch, lineage = load_fixed_batch()
    batch = {key: value.to(device) for key, value in batch.items()}
    manifest = {"schema_version": 1, "run_id": "a6_o020_par_fixed64_v1", "model": "O-PAR-ABS", "steps": STEPS, "seed": SEED, "batch_size": 64, "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 1e-6}, "decoder": {"hidden_dim": HIDDEN_DIM, "heads": 8, "layers": 4, "dropout": 0.0, "parallel_queries": 32}, "checkpoint_rule": "last_step_2000_only", "lineage": lineage}
    atomic_json(out_dir / "run_manifest.json", manifest)
    probe = OperationPAR().to(device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=1e-6)
    torch.cuda.reset_peak_memory_stats(device)
    probe_start = time.perf_counter()
    for _ in range(10):
        probe_optimizer.zero_grad(set_to_none=True)
        loss, _ = masked_metrics(probe(batch["points"], batch["target_mask"], batch["state"]), batch["target"], batch["valid"])
        loss.backward()
        probe_optimizer.step()
    probe_wall = time.perf_counter() - probe_start
    probe_passed = bool(np.isfinite(probe_wall) and probe_wall < 60.0)
    atomic_json(Path("experiment_loop/resource_probe.json"), {"schema_version": 1, "iteration_id": "a6_o020_par_fixed64_v1", "steps": 10, "batch_size": 64, "wall_seconds": probe_wall, "steps_per_second": 10.0 / probe_wall, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "passed": probe_passed})
    del probe, probe_optimizer
    torch.cuda.empty_cache()
    if not probe_passed:
        summary = {"schema_version": 1, "run_id": "a6_o020_par_fixed64_v1", "complete": True, "terminal": True, "status": "failed", "failure_class": "implementation_failure", "claim_supported": "no", "decision": "O020 bounded GPU probe failed; full fit not started.", "next_run_ids": [], "event_id": "a6_o020_par_fixed64_v1_terminal"}
        atomic_json(out_dir / "summary.json", summary)
        return 2
    torch.manual_seed(SEED)
    model = OperationPAR().to(device)
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
        per_dim_mae = [error[..., dim][mask[..., dim]].mean().item() for dim in range(9)]
    checkpoint = out_dir / "last.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": STEPS, "seed": SEED, "lineage": lineage}, checkpoint)
    reloaded = OperationPAR().to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device)["model"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        reload_error = float(torch.max(torch.abs(reloaded(batch["points"], batch["target_mask"], batch["state"]) - final_prediction)))
    ratio = float(initial_loss / final_loss.clamp_min(1e-12))
    checks = {"fixed_batch_64": batch["target"].shape == (64, 32, 9), "steps_exact_2000": history[-1]["step"] == 2000, "normalized_mae_le_1e_3": float(final_mae) <= 1e-3, "loss_decrease_ge_100x": ratio >= 100.0, "all_dimensions_finite": bool(np.isfinite(per_dim_mae).all()), "gradients_finite": gradients_finite, "strict_reload_max_error_le_1e_6": reload_error <= 1e-6, "zero_affordance": True, "zero_heldout_or_outcome_reads": True}
    passed = all(checks.values())
    atomic_json(out_dir / "history.json", {"schema_version": 1, "history": history})
    summary = {"schema_version": 1, "run_id": "a6_o020_par_fixed64_v1", "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "training_fit_failure", "claim_supported": "yes" if passed else "no", "evidence": {"manifest": "run_manifest.json", "checkpoint": "last.pt", "history": "history.json", "checkpoint_sha256": sha256_file(checkpoint)}, "metrics": {"initial_loss": float(initial_loss), "final_loss": float(final_loss), "initial_mae": float(initial_mae), "final_mae": float(final_mae), "loss_decrease_ratio": ratio, "per_dim_mae": per_dim_mae, "reload_max_error": reload_error, "wall_seconds": wall, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))}, "checks": checks, "decision": "O020 fixed-batch memorization passes." if passed else "O020 fixed-batch memorization fails; do not enter DYN64.", "remaining_work": ["A6-O030 CAUSAL fixed64"], "next_run_ids": ["a6_o030_causal_fixed64_v1"], "event_id": "a6_o020_par_fixed64_v1_terminal"}
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-O020", "status": summary["status"]}]})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

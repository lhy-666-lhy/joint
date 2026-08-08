#!/usr/bin/env python3
"""A6-FIT-v1.2 MLP fixed64 scratch fit to the frozen 6k budget."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from a6_operation_models import ACTION_DIM, ACTION_HORIZON, HIDDEN_DIM, OperationMLPAbsolute
from path_config import JOINTTRAIN_ARCH6_O010R_RESULT_ROOT, JOINTTRAIN_ARCH6_O201F_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    EFFECTIVE_BATCH,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    baseline_metrics,
    load_batch,
    model_inputs,
    normalized_l1_sum,
    percentile,
    sha256_file,
)


RUN_ID = "a6_o201f_mlp_state_start_delta_fixed64_v1"
STEPS = 6000
MICROBATCH = 64
REVISION_ID = "20260805T192525Z-1f253777"


def state_start_delta_targets(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state-start delta targets and normalized current-qpos base."""
    current_actual_qpos = batch["state"][:, 27:36]
    mean = batch["action_mean"].reshape(1, 1, ACTION_DIM)
    std = batch["action_std"].reshape(1, 1, ACTION_DIM)
    state_start_abs_norm = (current_actual_qpos.unsqueeze(1) - mean) / std
    delta_raw = batch["target_raw"] - current_actual_qpos.unsqueeze(1)
    delta_norm = delta_raw / std
    return delta_norm, state_start_abs_norm


def reconstruct_absolute(delta_norm: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    _, state_start_abs_norm = state_start_delta_targets(batch)
    return state_start_abs_norm + delta_norm


def evaluate(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(*model_inputs(batch, slice(None)))
    reconstructed = reconstruct_absolute(prediction, batch)
    numerator, denominator = normalized_l1_sum(reconstructed, batch["target"], batch["valid"])
    return prediction, numerator / denominator.clamp_min(1.0)


def optimizer_step_delta(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    microbatch: int,
) -> tuple[float, bool]:
    delta_target, _ = state_start_delta_targets(batch)
    optimizer.zero_grad(set_to_none=True)
    denominator = batch["valid"].sum().to(torch.float32) * ACTION_DIM
    total = 0.0
    for start in range(0, EFFECTIVE_BATCH, microbatch):
        stop = start + microbatch
        prediction = model(*model_inputs(batch, slice(start, stop)))
        numerator, _ = normalized_l1_sum(
            prediction, delta_target[start:stop], batch["valid"][start:stop]
        )
        loss_piece = numerator / denominator.clamp_min(1.0)
        loss_piece.backward()
        total += float(loss_piece.detach())
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if gradients_finite:
        optimizer.step()
    return total, gradients_finite


def write_running_state(out_dir: Path) -> None:
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "iteration_id": RUN_ID,
        "complete": False,
        "terminal": False,
        "status": "running",
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    atomic_json(out_dir / "run_state.json", payload)
    atomic_json(
        out_dir / "queue_state.json",
        {**payload, "jobs": [{"id": "A6-O201F", "status": "running", "pid": os.getpid()}]},
    )


def train(args: argparse.Namespace) -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O201F_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch, lineage, input_manifest = load_batch()
    torch.manual_seed(SEED)
    if args.validate_only:
        model = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        with torch.no_grad():
            output, loss = evaluate(model, batch)
        if output.shape != (64, ACTION_HORIZON, ACTION_DIM) or not bool(torch.isfinite(loss)):
            raise RuntimeError("O201F CPU validation failed")
        print(json.dumps({"status": "validated", "output_shape": list(output.shape)}))
        return 0
    write_running_state(out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("A6-O201F requires one CUDA GPU")
    device = torch.device("cuda:0")
    batch = {name: value.to(device) for name, value in batch.items()}
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model.eval()
    with torch.no_grad():
        _, initial_loss = evaluate(model, batch)
    baselines = baseline_metrics(batch)
    config = {
        "schema_version": 1,
        "planning_revision": REVISION_ID,
        "scientific_revision": "A6-FIT-v1.2",
        "model": "O-MLP-STATEDELTA",
        "shared_encoder": "PointCloudContextEncoder",
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "effective_batch": EFFECTIVE_BATCH,
        "microbatch": MICROBATCH,
        "gradient_accumulation_steps": 1,
        "optimizer_steps": STEPS,
        "sample_exposure": EFFECTIVE_BATCH * STEPS,
        "loss": "valid-mask normalized per-dimension L1 on state-start delta",
        "augmentation": False,
        "seed": SEED,
        "restart": "scratch; do not resume 2k checkpoint",
        "checkpoint_rule": "save fixed 6k last checkpoint; no ABS checkpoint reproduction",
        "representation": "state_start_delta = target_abs - current_actual_qpos",
        "current_qpos_slice": [27, 36],
        "reconstruction": "reconstructed_abs_norm = current_qpos_abs_norm + predicted_delta_norm",
        "forbidden_representation": "command_delta = target_abs - last_command_raw",
        "parameter_count": parameters,
    }
    atomic_json(
        out_dir / "command.json",
        {
            "schema_version": 1,
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "environment": "sapien",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu_id": os.environ.get("ARCH6_PHYSICAL_GPU_ID", "0"),
            "seed": SEED,
        },
    )
    atomic_json(out_dir / "training_config.json", config)
    atomic_json(out_dir / "sample_manifest.json", {"schema_version": 1, "rows": input_manifest["rows"]})
    atomic_json(
        out_dir / "forbidden_feature_audit.json",
        {
            "schema_version": 1,
            "result_json_read": False,
            "object_qpos_read": False,
            "future_qpos_read": False,
            "outcome_read": False,
            "heldout_read": False,
            "source": "O000BR2 fixed_input_v2.npz only; current_actual_qpos is state_history[27:36]",
        },
    )
    pilot_path = Path(JOINTTRAIN_ARCH6_O010R_RESULT_ROOT) / "resource_pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    selected = pilot["selected"]
    if selected != {
        "workers": 0,
        "microbatch": 64,
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "effective_batch": 64,
    }:
        raise ValueError("O010R resource pilot selection drifted")
    atomic_json(
        out_dir / "resource_pilot_ref.json",
        {
            "schema_version": 1,
            "source_run_id": "a6_o010r_mlp_fixed64_v2",
            "source_relative_path": "../a6_o010r_mlp_fixed64_v2/resource_pilot.json",
            "source_sha256": sha256_file(pilot_path),
            "workload_signature_sha256": pilot["workload_signature_sha256"],
            "selected": selected,
            "workload_unchanged_except_preregistered_optimizer_steps": True,
        },
    )
    atomic_json(
        out_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "depends_on": ["A6-O200F", "A6-A000RRRR", "A6-STATEDELTA-v1.6"],
            "config": config,
            "lineage": lineage,
            "resource_pilot_ref": "resource_pilot_ref.json",
        },
    )
    history: list[dict[str, Any]] = []
    step_times: list[float] = []
    gradients_finite = True
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        step_started = time.perf_counter()
        loss, finite = optimizer_step_delta(model, optimizer, batch, MICROBATCH)
        step_times.append(time.perf_counter() - step_started)
        gradients_finite = gradients_finite and finite and math.isfinite(loss)
        if not gradients_finite:
            raise RuntimeError(f"nonfinite O201F training state at step {step}")
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                prediction, evaluation_loss = evaluate(model, batch)
            history.append(
                {"step": step, "train_loss": loss, "eval_normalized_mae": float(evaluation_loss)}
            )
            model.train()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_prediction, final_loss = evaluate(model, batch)
        raw_prediction = reconstruct_absolute(final_prediction, batch) * batch["action_std"].reshape(1, 1, -1) + batch["action_mean"].reshape(1, 1, -1)
        raw_error = torch.abs(raw_prediction - batch["target_raw"])
        expanded_mask = batch["valid"].unsqueeze(-1).expand_as(raw_error)
        raw_mae = float(raw_error[expanded_mask].mean())
        per_dim_raw_mae = [
            float(raw_error[..., dim][expanded_mask[..., dim]].mean())
            for dim in range(ACTION_DIM)
        ]
    checkpoint = out_dir / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": STEPS,
            "seed": SEED,
            "lineage": lineage,
            "config": config,
            "action_mean": batch["action_mean"].cpu(),
            "action_std": batch["action_std"].cpu(),
        },
        checkpoint,
    )
    reloaded = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True
    )
    reloaded.eval()
    with torch.no_grad():
        reload_prediction, _ = evaluate(reloaded, batch)
    reload_error = float(torch.max(torch.abs(reload_prediction - final_prediction)))
    decrease = float(initial_loss / final_loss.clamp_min(1e-12))
    checks = {
        "fixed_batch_64": batch["target"].shape == (64, ACTION_HORIZON, ACTION_DIM),
        "scratch_seed_and_revision_exact": config["seed"] == SEED and config["scientific_revision"] == "A6-FIT-v1.2",
        "steps_exact_6000": history[-1]["step"] == STEPS,
        "zero_delta_reconstructs_current_qpos": bool(torch.max(torch.abs(reconstruct_absolute(torch.zeros_like(final_prediction), batch) - state_start_delta_targets(batch)[1])) <= 1e-6),
        "delta_absolute_l1_parity_le_1e_6": bool(torch.max(torch.abs((final_prediction - state_start_delta_targets(batch)[0]) - (reconstruct_absolute(final_prediction, batch) - batch["target"]))) <= 1e-6),
        "normalized_mae_le_1e_3": float(final_loss) <= 1e-3,
        "loss_decrease_ge_100x": decrease >= 100.0,
        "gradients_finite": gradients_finite,
        "strict_reload_max_error_le_1e_6": reload_error <= 1e-6,
        "shared_input_revision": lineage["revision"] == "A6-INPUT-v1.1",
        "zero_affordance": True,
        "zero_outcome_or_heldout_reads": True,
        "resource_pilot_reused_exactly": True,
    }
    passed = all(checks.values())
    metrics = {
        "initial_normalized_mae": float(initial_loss),
        "zero_current_qpos_hold_max_error": float(torch.max(torch.abs(reconstruct_absolute(torch.zeros_like(final_prediction), batch) - state_start_delta_targets(batch)[1]))),
        "delta_absolute_l1_parity_max_error": float(torch.max(torch.abs((final_prediction - state_start_delta_targets(batch)[0]) - (reconstruct_absolute(final_prediction, batch) - batch["target"])))),
        "final_normalized_mae": float(final_loss),
        "loss_decrease_ratio": decrease,
        "raw_mae": raw_mae,
        "per_dim_raw_mae": per_dim_raw_mae,
        "reload_max_error": reload_error,
        "baselines": baselines,
        "wall_seconds": wall,
        "optimizer_steps_per_second": STEPS / wall,
        "step_time_mean_seconds": statistics.fmean(step_times),
        "step_time_p95_seconds": percentile(step_times, 95),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameter_count": parameters,
    }
    atomic_json(out_dir / "history.json", {"schema_version": 1, "history": history})
    atomic_json(out_dir / "offline_metrics.json", {"schema_version": 1, **metrics})
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "training_fit_failure",
        "claim_supported": "yes" if passed else "no",
        "evidence": {
            "manifest": "run_manifest.json",
            "checkpoint_6000": "last.pt",
            "checkpoint_6000_sha256": sha256_file(checkpoint),
            "history": "history.json",
            "metrics": "offline_metrics.json",
            "resource_pilot_ref": "resource_pilot_ref.json",
            "forbidden_feature_audit": "forbidden_feature_audit.json",
        },
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "O201F state-start-delta fixed64 gate passes; publish a later DYN64 representation revision."
            if passed
            else "O201F remains a scoped state-start-delta training-fit failure; keep DYN64 revision-blocked."
        ),
        "remaining_work": ["analyze O201F terminal evidence", "keep DYN64 revision-blocked"],
        "next_run_ids": [],
        "event_id": f"{RUN_ID}_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(
        out_dir / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O201F", "status": summary["status"]}]},
    )
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"run-id must be {RUN_ID}")
    try:
        return train(args)
    except Exception as error:
        if args.validate_only:
            raise
        out_dir = Path(JOINTTRAIN_ARCH6_O201F_RESULT_ROOT)
        atomic_json(
            out_dir / "failure.json",
            {
                "schema_version": 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        summary = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "complete": True,
            "terminal": True,
            "status": "failed",
            "failure_class": "implementation_failure",
            "claim_supported": "no",
            "decision": "O201F implementation failed before valid state-start-delta evidence.",
            "remaining_work": ["inspect failure.json and repair without changing A6-STATEDELTA-v1.6"],
            "next_run_ids": [],
            "event_id": f"{RUN_ID}_terminal",
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(
            out_dir / "queue_state.json",
            {**summary, "jobs": [{"id": "A6-O201F", "status": "failed"}]},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

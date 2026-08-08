#!/usr/bin/env python3
"""A6-FIT-v1.2 parallel Transformer fixed64 scratch fit to 6k."""

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

from a6_operation_models import ACTION_DIM, ACTION_HORIZON, HIDDEN_DIM, OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_O020R_RESULT_ROOT, JOINTTRAIN_ARCH6_O020S_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    EFFECTIVE_BATCH,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    baseline_metrics,
    evaluate,
    load_batch,
    optimizer_step,
    percentile,
    sha256_file,
)


RUN_ID = "a6_o020s_par_fixed64_6k_v1"
STEPS = 6000
MICROBATCH = 64
REVISION_ID = "20260805T180136Z-e5545cf0"


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
        {**payload, "jobs": [{"id": "A6-O020S", "status": "running", "pid": os.getpid()}]},
    )


def train(args: argparse.Namespace) -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O020S_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch, lineage, input_manifest = load_batch()
    torch.manual_seed(SEED)
    if args.validate_only:
        model = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        with torch.no_grad():
            output, loss = evaluate(model, batch)
        if output.shape != (64, ACTION_HORIZON, ACTION_DIM) or not bool(torch.isfinite(loss)):
            raise RuntimeError("O020S CPU validation failed")
        print(json.dumps({"status": "validated", "output_shape": list(output.shape)}))
        return 0
    write_running_state(out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("A6-O020S requires one CUDA GPU")
    device = torch.device("cuda:0")
    batch = {name: value.to(device) for name, value in batch.items()}
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
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
        "model": "O-PAR-ABS",
        "decoder": {
            "hidden_dim": HIDDEN_DIM,
            "heads": 8,
            "layers": 1,
            "feedforward_dim": 64,
            "dropout": DROPOUT,
            "parallel_queries": ACTION_HORIZON,
        },
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
        "loss": "valid-mask normalized per-dimension L1",
        "augmentation": False,
        "seed": SEED,
        "restart": "scratch; do not resume 2k checkpoint",
        "checkpoint_rule": "save 2k reproduction; fixed 6k gate only",
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
            "source": "O000BR2 fixed_input_v2.npz only",
        },
    )
    pilot_path = Path(JOINTTRAIN_ARCH6_O020R_RESULT_ROOT) / "resource_pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    selected = pilot["selected"]
    if selected != {
        "workers": 0,
        "microbatch": 64,
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "effective_batch": 64,
    }:
        raise ValueError("O020R resource pilot selection drifted")
    atomic_json(
        out_dir / "resource_pilot_ref.json",
        {
            "schema_version": 1,
            "source_run_id": "a6_o020r_par_fixed64_v2",
            "source_relative_path": "../a6_o020r_par_fixed64_v2/resource_pilot.json",
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
            "depends_on": ["A6-O000C", "A6-FIT-v1.2"],
            "config": config,
            "lineage": lineage,
            "resource_pilot_ref": "resource_pilot_ref.json",
        },
    )
    old_checkpoint = Path(JOINTTRAIN_ARCH6_O020R_RESULT_ROOT) / "last.pt"
    old_saved = torch.load(old_checkpoint, map_location=device, weights_only=False)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    old_model = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    old_model.load_state_dict(old_saved["model"], strict=True)
    old_model.eval()
    with torch.no_grad():
        old_prediction, _ = evaluate(old_model, batch)
    del old_model, old_saved
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)
    history: list[dict[str, Any]] = []
    step_times: list[float] = []
    gradients_finite = True
    reproduction_error: float | None = None
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        step_started = time.perf_counter()
        loss, finite = optimizer_step(model, optimizer, batch, MICROBATCH)
        step_times.append(time.perf_counter() - step_started)
        gradients_finite = gradients_finite and finite and math.isfinite(loss)
        if not gradients_finite:
            raise RuntimeError(f"nonfinite O020S training state at step {step}")
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                prediction, evaluation_loss = evaluate(model, batch)
            history.append(
                {"step": step, "train_loss": loss, "eval_normalized_mae": float(evaluation_loss)}
            )
            if step == 2000:
                reproduction_error = float(torch.max(torch.abs(prediction - old_prediction)))
                torch.save(
                    {"model": model.state_dict(), "step": step, "seed": SEED, "lineage": lineage, "config": config},
                    out_dir / "checkpoint_2000.pt",
                )
            model.train()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_prediction, final_loss = evaluate(model, batch)
        raw_prediction = final_prediction * batch["action_std"].reshape(1, 1, -1) + batch[
            "action_mean"
        ].reshape(1, 1, -1)
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
    reloaded = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
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
        "step_2000_reproduction_le_1e_6": reproduction_error is not None and reproduction_error <= 1e-6,
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
        "step_2000_reproduction_max_error": reproduction_error,
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
            "checkpoint_2000": "checkpoint_2000.pt",
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
            "O020S passes the revised 6k fixed64 gate; continue independent O030S before DYN64 routing."
            if passed
            else "O020S remains a scoped training-fit failure at the revised 6k budget; continue independent O030S."
        ),
        "remaining_work": ["analyze O020S terminal evidence", "route independent O030S under A6-FIT-v1.2"],
        "next_run_ids": ["a6_o030s_causal_fixed64_6k_v1"],
        "event_id": f"{RUN_ID}_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(
        out_dir / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O020S", "status": summary["status"]}]},
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
        out_dir = Path(JOINTTRAIN_ARCH6_O020S_RESULT_ROOT)
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
            "decision": "O020S implementation failed before valid revised-budget evidence.",
            "remaining_work": ["inspect failure.json and repair without changing A6-FIT-v1.2"],
            "next_run_ids": [],
            "event_id": f"{RUN_ID}_terminal",
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(
            out_dir / "queue_state.json",
            {**summary, "jobs": [{"id": "A6-O020S", "status": "failed"}]},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

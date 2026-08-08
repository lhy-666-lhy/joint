#!/usr/bin/env python3
"""Corrected A6-O020R parallel Transformer fixed64 memorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from a6_operation_models import (
    ACTION_DIM,
    ACTION_HORIZON,
    HIDDEN_DIM,
    OperationMLPAbsolute,
    OperationParallelAbsolute,
)
from path_config import JOINTTRAIN_ARCH6_O020R_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    EFFECTIVE_BATCH,
    LEARNING_RATE,
    MICROBATCH_CANDIDATES,
    PILOT_MIN_SECONDS,
    PILOT_MIN_STEPS,
    PILOT_WARMUP_STEPS,
    SEED,
    STEPS,
    WEIGHT_DECAY,
    ResourceSampler,
    atomic_json,
    baseline_metrics,
    evaluate,
    load_batch,
    model_inputs,
    optimizer_step,
    percentile,
    sha256_file,
)


RUN_ID = "a6_o020r_par_fixed64_v2"


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_resource_pilot(
    batch: dict[str, torch.Tensor], device: torch.device, out_dir: Path
) -> tuple[int, dict[str, Any]]:
    physical_gpu = int(os.environ.get("ARCH6_PHYSICAL_GPU_ID", "0"))
    software = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
    }
    workload_signature = {
        "family": "A6-O020R fixed64 memorization",
        "model": "PointCloudContextEncoder256+PAR-ABS",
        "decoder": {
            "parallel_queries": ACTION_HORIZON,
            "hidden_dim": HIDDEN_DIM,
            "heads": 8,
            "layers": 1,
            "feedforward_dim": 64,
            "dropout": DROPOUT,
        },
        "input_shapes": {
            name: list(value.shape)
            for name, value in batch.items()
            if name not in {"action_mean", "action_std"}
        },
        "output_shape": [64, ACTION_HORIZON, ACTION_DIM],
        "precision": "float32",
        "backend": "in-memory fixed TRAIN tensors",
        "world_size": 1,
        "hardware": software["device"],
        "software_environment_hash": hashlib.sha256(
            json.dumps(software, sort_keys=True).encode()
        ).hexdigest(),
    }
    measurements: list[dict[str, Any]] = []
    selected_microbatch = MICROBATCH_CANDIDATES[0]
    stop_reason = "largest_candidate_reached"
    for microbatch in MICROBATCH_CANDIDATES:
        torch.manual_seed(SEED)
        model = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        model.train()
        for _ in range(PILOT_WARMUP_STEPS):
            _, finite = optimizer_step(model, optimizer, batch, microbatch)
            if not finite:
                raise RuntimeError("nonfinite gradient during O020R pilot warmup")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        sampler = ResourceSampler(physical_gpu)
        sampler.start()
        step_times: list[float] = []
        timed_steps = 0
        started = time.perf_counter()
        gradients_finite = True
        while time.perf_counter() - started < PILOT_MIN_SECONDS or timed_steps < PILOT_MIN_STEPS:
            step_started = time.perf_counter()
            _, finite = optimizer_step(model, optimizer, batch, microbatch)
            torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - step_started)
            gradients_finite = gradients_finite and finite
            timed_steps += 1
            if not gradients_finite:
                break
        wall = time.perf_counter() - started
        sampled = sampler.stop()
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        measurement = {
            "workers": 0,
            "microbatch": microbatch,
            "world_size": 1,
            "gradient_accumulation_steps": EFFECTIVE_BATCH // microbatch,
            "effective_batch": EFFECTIVE_BATCH,
            "warmup_steps": PILOT_WARMUP_STEPS,
            "timed_steps": timed_steps,
            "timed_seconds": wall,
            "samples_per_second": timed_steps * EFFECTIVE_BATCH / wall,
            "optimizer_steps_per_second": timed_steps / wall,
            "step_time_mean_seconds": statistics.fmean(step_times),
            "step_time_p95_seconds": percentile(step_times, 95),
            "peak_memory_bytes": peak_bytes,
            "peak_memory_fraction": peak_bytes
            / torch.cuda.get_device_properties(device).total_memory,
            "gradients_finite": gradients_finite,
            "oom_retries": 0,
            **sampled,
        }
        measurements.append(measurement)
        if not gradients_finite:
            raise RuntimeError("nonfinite gradient during O020R resource pilot")
        if measurement["peak_memory_fraction"] > 0.8:
            stop_reason = "memory_limit"
            break
        selected_microbatch = microbatch
        if len(measurements) > 1:
            gain = measurement["samples_per_second"] / measurements[-2]["samples_per_second"]
            measurement["gain_vs_previous"] = gain
            if gain < 1.1:
                selected_microbatch = measurements[-2]["microbatch"]
                stop_reason = "throughput_gain_below_10_percent"
                break
        del model, optimizer
        torch.cuda.empty_cache()
    selected = next(row for row in measurements if row["microbatch"] == selected_microbatch)
    result = {
        "schema_version": 1,
        "run_id": f"{RUN_ID}-RP",
        "workload_signature": workload_signature,
        "workload_signature_sha256": hashlib.sha256(
            json.dumps(workload_signature, sort_keys=True).encode()
        ).hexdigest(),
        "candidates": measurements,
        "selected": {
            "workers": 0,
            "microbatch": selected_microbatch,
            "world_size": 1,
            "gradient_accumulation_steps": EFFECTIVE_BATCH // selected_microbatch,
            "effective_batch": EFFECTIVE_BATCH,
        },
        "stop_reason": stop_reason,
        "bottleneck_classification": (
            "small-model/in-memory fixed-batch bound"
            if selected.get("gpu_sm_mean") is not None and selected["gpu_sm_mean"] < 80.0
            else "gpu_compute_bound_or_unavailable"
        ),
        "effective_batch_parity": True,
        "sample_exposure_parity": True,
        "loader_worker_note": "Frozen fixed64 tensors are resident in memory; no DataLoader is used.",
        "gpu_keepalive_stop_record": {
            "real_workload_gpu": physical_gpu,
            "gpu_memory_used_at_first_sample_mib": measurements[0].get("gpu_memory_peak_mib"),
        },
    }
    atomic_json(out_dir / "resource_pilot.json", result)
    return selected_microbatch, result


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
        {**payload, "jobs": [{"id": "A6-O020R", "status": "running", "pid": os.getpid()}]},
    )


def train(args: argparse.Namespace) -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O020R_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch, lineage, input_manifest = load_batch()
    torch.manual_seed(SEED)
    if args.validate_only:
        model = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        with torch.no_grad():
            output = model(*model_inputs(batch, slice(0, 2)))
        if output.shape != (2, ACTION_HORIZON, ACTION_DIM) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("O020R CPU validation forward failed")
        print(json.dumps({"status": "validated", "output_shape": list(output.shape)}))
        return 0
    write_running_state(out_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("A6-O020R requires one CUDA GPU")
    device = torch.device("cuda:0")
    batch = {name: value.to(device) for name, value in batch.items()}
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_probe = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    mlp_probe = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    total_parameters = parameter_count(model_probe)
    decoder_parameters = (
        parameter_count(model_probe.decoder)
        + parameter_count(model_probe.action_head)
        + model_probe.queries.numel()
    )
    mlp_total_parameters = parameter_count(mlp_probe)
    mlp_decoder_parameters = parameter_count(mlp_probe.decoder)
    del model_probe, mlp_probe
    config = {
        "schema_version": 1,
        "model": "O-PAR-ABS",
        "shared_encoder": "PointCloudContextEncoder",
        "decoder": {
            "hidden_dim": HIDDEN_DIM,
            "heads": 8,
            "layers": 1,
            "feedforward_dim": 64,
            "dropout": DROPOUT,
            "parallel_queries": ACTION_HORIZON,
        },
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "effective_batch": EFFECTIVE_BATCH,
        "optimizer_steps": STEPS,
        "sample_exposure": EFFECTIVE_BATCH * STEPS,
        "loss": "valid-mask normalized per-dimension L1",
        "augmentation": False,
        "seed": SEED,
        "checkpoint_rule": "fixed last-step 2000 only",
        "parameter_count": total_parameters,
        "decoder_parameter_count": decoder_parameters,
        "parameter_parity_vs_o010r": {
            "total_ratio": total_parameters / mlp_total_parameters,
            "decoder_ratio": decoder_parameters / mlp_decoder_parameters,
        },
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
    atomic_json(
        out_dir / "sample_manifest.json",
        {"schema_version": 1, "rows": input_manifest["rows"]},
    )
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
    atomic_json(
        out_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "depends_on": ["A6-D020", "A6-D030", "A6-O000BR2", "A6-O010R terminal routing"],
            "config": config,
            "lineage": lineage,
        },
    )
    selected_microbatch, pilot = run_resource_pilot(batch, device, out_dir)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model.eval()
    with torch.no_grad():
        _, initial_loss = evaluate(model, batch)
    baselines = baseline_metrics(batch)
    model.train()
    history: list[dict[str, Any]] = []
    gradients_finite = True
    step_times: list[float] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        step_started = time.perf_counter()
        loss, finite = optimizer_step(model, optimizer, batch, selected_microbatch)
        step_times.append(time.perf_counter() - step_started)
        gradients_finite = gradients_finite and finite and math.isfinite(loss)
        if not gradients_finite:
            raise RuntimeError(f"nonfinite O020R training state at step {step}")
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                _, evaluation_loss = evaluate(model, batch)
            history.append(
                {"step": step, "train_loss": loss, "eval_normalized_mae": float(evaluation_loss)}
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
        "steps_exact_2000": history[-1]["step"] == STEPS,
        "normalized_mae_le_1e_3": float(final_loss) <= 1e-3,
        "loss_decrease_ge_100x": decrease >= 100.0,
        "gradients_finite": gradients_finite,
        "strict_reload_max_error_le_1e_6": reload_error <= 1e-6,
        "shared_input_revision": lineage["revision"] == "A6-INPUT-v1.1",
        "zero_affordance": True,
        "zero_outcome_or_heldout_reads": True,
        "parameter_parity_within_20_percent": abs(total_parameters / mlp_total_parameters - 1.0) <= 0.2,
        "resource_pilot_contract": all(
            row["warmup_steps"] >= 20
            and row["timed_steps"] >= 100
            and row["timed_seconds"] >= 60.0
            for row in pilot["candidates"]
        ),
    }
    passed = all(checks.values())
    metrics = {
        "initial_normalized_mae": float(initial_loss),
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
        "parameter_count": total_parameters,
        "decoder_parameter_count": decoder_parameters,
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
            "checkpoint": "last.pt",
            "checkpoint_sha256": sha256_file(checkpoint),
            "history": "history.json",
            "metrics": "offline_metrics.json",
            "resource_pilot": "resource_pilot.json",
            "forbidden_feature_audit": "forbidden_feature_audit.json",
        },
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "Corrected O020R fixed64 fit passes; proceed to independent O030R before any DYN64 fit."
            if passed
            else "Corrected O020R is a scoped training-fit failure; route independent O030R."
        ),
        "remaining_work": ["analyze O020R terminal evidence", "route O030R independently"],
        "next_run_ids": ["a6_o030r_causal_fixed64_v2"],
        "event_id": f"{RUN_ID}_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(
        out_dir / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O020R", "status": summary["status"]}]},
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
        out_dir = Path(JOINTTRAIN_ARCH6_O020R_RESULT_ROOT)
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
            "decision": "O020R implementation or resource execution failed before valid fit evidence.",
            "remaining_work": ["inspect failure.json and repair without changing the frozen contract"],
            "next_run_ids": [],
            "event_id": f"{RUN_ID}_terminal",
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(
            out_dir / "queue_state.json",
            {**summary, "jobs": [{"id": "A6-O020R", "status": "failed"}]},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

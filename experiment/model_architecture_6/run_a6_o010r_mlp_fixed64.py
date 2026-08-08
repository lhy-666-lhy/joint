#!/usr/bin/env python3
"""Corrected A6-O010R fixed64 memorization and bounded GPU resource pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from a6_operation_models import ACTION_DIM, ACTION_HORIZON, HIDDEN_DIM, OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010R_RESULT_ROOT,
)


RUN_ID = "a6_o010r_mlp_fixed64_v2"
SEED = 20260805
STEPS = 2000
EFFECTIVE_BATCH = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
DROPOUT = 0.1
PILOT_WARMUP_STEPS = 20
PILOT_MIN_SECONDS = 60.0
PILOT_MIN_STEPS = 100
MICROBATCH_CANDIDATES = (16, 32, 64)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def load_batch() -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    input_root = Path(JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT)
    normalizer_path = Path(JOINTTRAIN_ARCH6_D020_RESULT_ROOT) / "normalizer.json"
    manifest_path = input_root / "input_manifest.json"
    audit_path = input_root / "forbidden_feature_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != "A6-INPUT-v1.1":
        raise ValueError("fixed input revision is not A6-INPUT-v1.1")
    if any(
        bool(audit.get(field))
        for field in ("future_qpos_read", "result_json_read", "object_qpos_read", "outcome_read", "heldout_read")
    ):
        raise ValueError("forbidden feature audit is not clean")
    with np.load(input_root / "fixed_input_v2.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    expected_shapes = {
        "point_cloud": (64, 1024, 3),
        "target_mask": (64, 1024),
        "zero_affordance": (64, 1024),
        "state_history": (64, 81),
        "context": (64, 43),
        "action_target": (64, ACTION_HORIZON, ACTION_DIM),
        "action_valid": (64, ACTION_HORIZON),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"invalid fixed input {name}: {arrays[name].shape}")
    if np.count_nonzero(arrays["zero_affordance"]):
        raise ValueError("ZERO affordance channel is not zero")
    mean = np.asarray(normalizer["mean"], dtype=np.float32).reshape(1, 1, ACTION_DIM)
    std = np.asarray(normalizer["std"], dtype=np.float32).reshape(1, 1, ACTION_DIM)
    target_raw = arrays["action_target"].astype(np.float32)
    target_normalized = (target_raw - mean) / std
    batch = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"].astype(np.float32)),
        "target_mask": torch.from_numpy(arrays["target_mask"].astype(bool)),
        "affordance": torch.from_numpy(arrays["zero_affordance"].astype(np.float32)),
        "state": torch.from_numpy(arrays["state_history"].astype(np.float32)),
        "context": torch.from_numpy(arrays["context"].astype(np.float32)),
        "target": torch.from_numpy(target_normalized.astype(np.float32)),
        "target_raw": torch.from_numpy(target_raw),
        "valid": torch.from_numpy(arrays["action_valid"].astype(bool)),
        "action_mean": torch.from_numpy(mean.reshape(ACTION_DIM)),
        "action_std": torch.from_numpy(std.reshape(ACTION_DIM)),
    }
    lineage = {
        "revision": manifest["revision"],
        "fixed_input_sha256": sha256_file(input_root / "fixed_input_v2.npz"),
        "input_manifest_sha256": sha256_file(manifest_path),
        "forbidden_feature_audit_sha256": sha256_file(audit_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "qvel_source": "240Hz causal backward finite difference",
        "live_qvel_source": "SAPIEN actual qvel",
        "affordance_channel": "ZERO",
        "result_json_read": False,
        "outcome_read": False,
        "heldout_read": False,
    }
    return batch, lineage, manifest


def model_inputs(batch: dict[str, torch.Tensor], index: slice) -> tuple[torch.Tensor, ...]:
    return (
        batch["point_cloud"][index],
        batch["target_mask"][index],
        batch["affordance"][index],
        batch["state"][index],
        batch["context"][index],
    )


def normalized_l1_sum(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = valid.unsqueeze(-1).expand_as(prediction)
    absolute = torch.abs(prediction - target)
    return absolute[mask].sum(), mask.sum().to(prediction.dtype)


def evaluate(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(*model_inputs(batch, slice(None)))
    numerator, denominator = normalized_l1_sum(prediction, batch["target"], batch["valid"])
    return prediction, numerator / denominator.clamp_min(1.0)


def optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    microbatch: int,
) -> tuple[float, bool]:
    optimizer.zero_grad(set_to_none=True)
    denominator = batch["valid"].sum().to(torch.float32) * ACTION_DIM
    total = 0.0
    for start in range(0, EFFECTIVE_BATCH, microbatch):
        stop = start + microbatch
        prediction = model(*model_inputs(batch, slice(start, stop)))
        numerator, _ = normalized_l1_sum(
            prediction, batch["target"][start:stop], batch["valid"][start:stop]
        )
        loss_piece = numerator / denominator.clamp_min(1.0)
        loss_piece.backward()
        total += float(loss_piece.detach())
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    optimizer.step()
    return total, gradients_finite


class ResourceSampler:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.stop_event = threading.Event()
        self.gpu_util: list[float] = []
        self.gpu_memory_mib: list[float] = []
        self.cpu_util: list[float] = []
        self.io_wait: list[float] = []
        self.rss_mib: list[float] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _cpu_row() -> list[int]:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        return [int(value) for value in fields]

    @staticmethod
    def _rss_mib() -> float:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        return {
            "gpu_sm_mean": statistics.fmean(self.gpu_util) if self.gpu_util else None,
            "gpu_sm_p95": percentile(self.gpu_util, 95),
            "gpu_memory_peak_mib": max(self.gpu_memory_mib, default=None),
            "cpu_mean": statistics.fmean(self.cpu_util) if self.cpu_util else None,
            "cpu_p95": percentile(self.cpu_util, 95),
            "io_wait_mean": statistics.fmean(self.io_wait) if self.io_wait else None,
            "rss_peak_mib": max(self.rss_mib, default=None),
            "samples": len(self.gpu_util),
        }

    def _run(self) -> None:
        previous = self._cpu_row()
        while not self.stop_event.wait(1.0):
            current = self._cpu_row()
            delta = [max(0, right - left) for left, right in zip(previous, current)]
            total = sum(delta)
            idle = sum(delta[index] for index in (3, 4) if index < len(delta))
            self.cpu_util.append(0.0 if total == 0 else 100.0 * (total - idle) / total)
            self.io_wait.append(0.0 if total == 0 or len(delta) <= 4 else 100.0 * delta[4] / total)
            self.rss_mib.append(self._rss_mib())
            previous = current
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2.0,
                )
                for line in output.splitlines():
                    index, utilization, memory = [part.strip() for part in line.split(",")]
                    if int(index) == self.physical_gpu:
                        self.gpu_util.append(float(utilization))
                        self.gpu_memory_mib.append(float(memory))
                        break
            except (OSError, subprocess.SubprocessError, ValueError):
                continue


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
        "family": "A6-O010R fixed64 memorization",
        "model": "PointCloudContextEncoder256+MLP-ABS",
        "input_shapes": {name: list(value.shape) for name, value in batch.items() if name not in {"action_mean", "action_std"}},
        "output_shape": [64, ACTION_HORIZON, ACTION_DIM],
        "precision": "float32",
        "backend": "in-memory fixed TRAIN tensors",
        "renderer": None,
        "world_size": 1,
        "hardware": software["device"],
        "software_environment_hash": hashlib.sha256(json.dumps(software, sort_keys=True).encode()).hexdigest(),
    }
    measurements: list[dict[str, Any]] = []
    selected_microbatch = MICROBATCH_CANDIDATES[0]
    stop_reason = "largest_candidate_reached"
    for microbatch in MICROBATCH_CANDIDATES:
        torch.manual_seed(SEED)
        model = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model.train()
        for _ in range(PILOT_WARMUP_STEPS):
            _, finite = optimizer_step(model, optimizer, batch, microbatch)
            if not finite:
                raise RuntimeError("nonfinite gradient during resource pilot warmup")
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
            "peak_memory_fraction": peak_bytes / torch.cuda.get_device_properties(device).total_memory,
            "gradients_finite": gradients_finite,
            "oom_retries": 0,
            **sampled,
        }
        measurements.append(measurement)
        if not gradients_finite or measurement["peak_memory_fraction"] > 0.8:
            stop_reason = "nonfinite_or_memory_limit"
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
    keepalive_state = Path(JOINTTRAIN_ARCH6_O010R_RESULT_ROOT).parents[1] / "experiment_loop" / "gpu_keepalive_state.json"
    keepalive = json.loads(keepalive_state.read_text(encoding="utf-8")) if keepalive_state.exists() else {}
    result = {
        "schema_version": 1,
        "run_id": f"{RUN_ID}-RP",
        "workload_signature": workload_signature,
        "workload_signature_sha256": hashlib.sha256(json.dumps(workload_signature, sort_keys=True).encode()).hexdigest(),
        "candidates": measurements,
        "selected": {
            "workers": 0,
            "microbatch": selected_microbatch,
            "world_size": 1,
            "gradient_accumulation_steps": EFFECTIVE_BATCH // selected_microbatch,
            "effective_batch": EFFECTIVE_BATCH,
        },
        "stop_reason": stop_reason,
        "bottleneck_classification": "small-model/in-memory fixed-batch bound" if selected.get("gpu_sm_mean", 100.0) < 80.0 else "gpu_compute_bound",
        "effective_batch_parity": True,
        "sample_exposure_parity": True,
        "loader_worker_note": "No DataLoader is used by the frozen in-memory fixed64 workload; workers=0 is the only applicable setting.",
        "gpu_keepalive_stop_record": {
            "state_resource_mode": keepalive.get("resource_mode"),
            "session_running_at_pilot_start": keepalive.get("session_running"),
            "active_gpu_ids_at_pilot_start": keepalive.get("active_gpu_ids", []),
            "real_workload_gpu": physical_gpu,
        },
    }
    atomic_json(out_dir / "resource_pilot.json", result)
    return selected_microbatch, result


def baseline_metrics(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    valid = batch["valid"]
    mean = batch["action_mean"].reshape(1, 1, ACTION_DIM)
    std = batch["action_std"].reshape(1, 1, ACTION_DIM)
    repeat_raw = batch["state"][:, -ACTION_DIM:].unsqueeze(1).expand(-1, ACTION_HORIZON, -1)
    repeat = (repeat_raw - mean) / std
    train_mean = torch.zeros_like(batch["target"])
    for horizon in range(ACTION_HORIZON):
        horizon_valid = valid[:, horizon]
        if bool(horizon_valid.any()):
            train_mean[:, horizon] = batch["target"][horizon_valid, horizon].mean(dim=0)
    metrics: dict[str, float] = {}
    for name, prediction in (("repeat_last_command", repeat), ("train_mean_chunk", train_mean)):
        numerator, denominator = normalized_l1_sum(prediction, batch["target"], valid)
        metrics[f"{name}_normalized_mae"] = float(numerator / denominator.clamp_min(1.0))
    return metrics


def write_running_state(out_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "iteration_id": args.run_id,
        "complete": False,
        "terminal": False,
        "status": "running",
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    atomic_json(out_dir / "run_state.json", payload)
    atomic_json(out_dir / "queue_state.json", {**payload, "jobs": [{"id": "A6-O010R", "status": "running", "pid": os.getpid()}]})


def train(args: argparse.Namespace) -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O010R_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch, lineage, input_manifest = load_batch()
    if args.validate_only:
        torch.manual_seed(SEED)
        model = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        with torch.no_grad():
            output = model(*model_inputs(batch, slice(0, 2)))
        if output.shape != (2, ACTION_HORIZON, ACTION_DIM) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("CPU validation forward failed")
        print(json.dumps({"status": "validated", "output_shape": list(output.shape)}))
        return 0
    write_running_state(out_dir, args)
    if not torch.cuda.is_available():
        raise RuntimeError("A6-O010R requires one CUDA GPU")
    device = torch.device("cuda:0")
    batch = {name: value.to(device) for name, value in batch.items()}
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parameter_probe = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    parameter_count = sum(parameter.numel() for parameter in parameter_probe.parameters())
    del parameter_probe
    command = {
        "schema_version": 1,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "environment": "sapien",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_id": os.environ.get("ARCH6_PHYSICAL_GPU_ID", "0"),
        "seed": SEED,
    }
    config = {
        "schema_version": 1,
        "model": "O-MLP-ABS",
        "shared_encoder": "PointCloudContextEncoder",
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
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
        "parameter_count": parameter_count,
    }
    atomic_json(out_dir / "command.json", command)
    atomic_json(out_dir / "training_config.json", config)
    atomic_json(out_dir / "sample_manifest.json", {"schema_version": 1, "rows": input_manifest["rows"]})
    atomic_json(out_dir / "forbidden_feature_audit.json", {
        "schema_version": 1,
        "result_json_read": False,
        "object_qpos_read": False,
        "future_qpos_read": False,
        "outcome_read": False,
        "heldout_read": False,
        "source": "O000BR2 forbidden audit and fixed_input_v2.npz only",
    })
    atomic_json(out_dir / "run_manifest.json", {
        "schema_version": 1,
        "run_id": RUN_ID,
        "depends_on": ["A6-D020", "A6-D030", "A6-O000BR2"],
        "config": config,
        "lineage": lineage,
    })
    selected_microbatch, pilot = run_resource_pilot(batch, device, out_dir)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.eval()
    with torch.no_grad():
        initial_prediction, initial_loss = evaluate(model, batch)
    baseline = baseline_metrics(batch)
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
            raise RuntimeError(f"nonfinite training state at step {step}")
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                _, evaluation_loss = evaluate(model, batch)
            history.append({"step": step, "train_loss": loss, "eval_normalized_mae": float(evaluation_loss)})
            model.train()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_prediction, final_loss = evaluate(model, batch)
        raw_prediction = final_prediction * batch["action_std"].reshape(1, 1, -1) + batch["action_mean"].reshape(1, 1, -1)
        raw_error = torch.abs(raw_prediction - batch["target_raw"])
        expanded_mask = batch["valid"].unsqueeze(-1).expand_as(raw_error)
        per_dim_raw_mae = [float(raw_error[..., dim][expanded_mask[..., dim]].mean()) for dim in range(ACTION_DIM)]
        raw_mae = float(raw_error[expanded_mask].mean())
    checkpoint = out_dir / "last.pt"
    torch.save({
        "model": model.state_dict(),
        "step": STEPS,
        "seed": SEED,
        "lineage": lineage,
        "config": config,
        "action_mean": batch["action_mean"].cpu(),
        "action_std": batch["action_std"].cpu(),
    }, checkpoint)
    reloaded = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(saved["model"], strict=True)
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
        "resource_pilot_contract": all(
            row["warmup_steps"] >= 20 and row["timed_steps"] >= 100 and row["timed_seconds"] >= 60.0
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
        "baselines": baseline,
        "wall_seconds": wall,
        "optimizer_steps_per_second": STEPS / wall,
        "step_time_mean_seconds": statistics.fmean(step_times),
        "step_time_p95_seconds": percentile(step_times, 95),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameter_count": parameter_count,
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
        "decision": "Corrected O010R fixed64 fit passes; shared runner may be extended to O020R." if passed else "Corrected O010R is a scoped training-fit failure; do not block independent corrected architectures.",
        "remaining_work": ["analyze O010R terminal evidence", "route O020R independently"],
        "next_run_ids": ["a6_o020r_par_fixed64_v2"],
        "event_id": f"{RUN_ID}_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-O010R", "status": summary["status"]}]})
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
        out_dir = Path(JOINTTRAIN_ARCH6_O010R_RESULT_ROOT)
        atomic_json(out_dir / "failure.json", {
            "schema_version": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        summary = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "complete": True,
            "terminal": True,
            "status": "failed",
            "failure_class": "implementation_failure",
            "claim_supported": "no",
            "decision": "O010R implementation or resource execution failed before valid fit evidence.",
            "remaining_work": ["inspect failure.json and repair without changing frozen scientific contract"],
            "next_run_ids": [],
            "event_id": f"{RUN_ID}_terminal",
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-O010R", "status": "failed"}]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a zero-init residual over frozen O127C for D180C recovery states."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute, OperationMLPRecoveryResidual
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D180C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O185C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    normalized_l1_sum,
)


RUN_ID = "a6_o185c_recovery_residual_train_v1"
QUEUE_RUN_ID = "A6-O185C"
NEXT_RUN_IDS = ["A6-O186C", "A6-O187C"]
OUTPUT_ROOT = JOINTTRAIN_ARCH6_O185C_RESULT_ROOT
SAMPLING_MODE = "random_with_replacement"
SCIENTIFIC_SCOPE = "frozen O127C plus isolated time-recovery residual"
STEPS = 6000
HALF_BATCH = 32
PREFIX_ROWS = 1024
TOTAL_ROWS = 1152
ACTION_DIM = 9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_inputs(data: dict[str, torch.Tensor], index: torch.Tensor) -> tuple:
    return (
        data["point_cloud"][index],
        data["target_mask"][index],
        data["affordance"][index],
        data["state"][index],
        data["context"][index],
    )


def build_sampling_indices(
    *,
    start: int,
    stop: int,
    steps: int,
    batch_size: int,
    generator: torch.Generator,
    mode: str,
) -> torch.Tensor:
    row_count = stop - start
    if mode == "random_with_replacement":
        return torch.randint(start, stop, (steps, batch_size), generator=generator)
    if mode != "epoch_balanced_without_replacement":
        raise ValueError(f"unsupported sampling mode: {mode}")
    if row_count <= 0 or row_count % batch_size:
        raise ValueError("epoch-balanced sampling requires rows divisible by batch size")
    batches_per_epoch = row_count // batch_size
    epoch_count = (steps + batches_per_epoch - 1) // batches_per_epoch
    epochs = [
        torch.randperm(row_count, generator=generator).reshape(-1, batch_size) + start
        for _ in range(epoch_count)
    ]
    return torch.cat(epochs, dim=0)[:steps]


def main() -> int:
    out = Path(OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    d180 = Path(JOINTTRAIN_ARCH6_D180C_RESULT_ROOT)
    source_path = d180 / "train_recovery_time1152.npz"
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    action_std = torch.tensor(
        json.loads(
            (Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text(
                encoding="utf-8"
            )
        )["std"],
        dtype=torch.float32,
    ).reshape(1, 1, ACTION_DIM)
    data = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"].astype(np.float32)),
        "target_mask": torch.from_numpy(arrays["target_mask"].astype(bool)),
        "affordance": torch.from_numpy(arrays["zero_affordance"].astype(np.float32)),
        "state": torch.from_numpy(arrays["state_history"].astype(np.float32)),
        "context": torch.from_numpy(arrays["context"].astype(np.float32)),
        "target": torch.from_numpy(
            arrays["command_delta_target"].astype(np.float32)
        )
        / action_std,
        "valid": torch.from_numpy(arrays["action_valid"].astype(bool)),
    }
    if data["state"].shape != (TOTAL_ROWS, 81):
        raise ValueError("unexpected recovery residual input shape")
    if not torch.cuda.is_available():
        raise RuntimeError("O185C requires CUDA")
    device = torch.device("cuda:0")
    baseline_checkpoint = Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt"
    baseline_state = torch.load(
        baseline_checkpoint, map_location=device, weights_only=False
    )["model"]
    baseline = OperationMLPAbsolute(dropout=DROPOUT).to(device)
    baseline.load_state_dict(baseline_state, strict=True)
    baseline.eval()
    model = OperationMLPRecoveryResidual(dropout=DROPOUT).to(device)
    model.baseline.load_state_dict(baseline_state, strict=True)
    for parameter in model.baseline.parameters():
        parameter.requires_grad_(False)
    model.baseline.eval()

    with torch.no_grad():
        parity_index = torch.arange(64)
        parity_inputs = tuple(value.to(device) for value in model_inputs(data, parity_index))
        step0_parity = float(
            torch.max(torch.abs(model(*parity_inputs) - baseline(*parity_inputs)))
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    prefix_indices = build_sampling_indices(
        start=0,
        stop=PREFIX_ROWS,
        steps=STEPS,
        batch_size=HALF_BATCH,
        generator=generator,
        mode=SAMPLING_MODE,
    )
    recovery_indices = build_sampling_indices(
        start=PREFIX_ROWS,
        stop=TOTAL_ROWS,
        steps=STEPS,
        batch_size=HALF_BATCH,
        generator=generator,
        mode=SAMPLING_MODE,
    )
    prefix_exposure = torch.bincount(
        prefix_indices.reshape(-1), minlength=PREFIX_ROWS
    )[:PREFIX_ROWS]
    recovery_exposure = torch.bincount(
        recovery_indices.reshape(-1) - PREFIX_ROWS,
        minlength=TOTAL_ROWS - PREFIX_ROWS,
    )
    optimizer = torch.optim.AdamW(
        model.recovery_head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict] = []
    started = time.perf_counter()
    for step in range(STEPS):
        prefix_index = prefix_indices[step]
        recovery_index = recovery_indices[step]
        prefix_input = tuple(
            value.to(device) for value in model_inputs(data, prefix_index)
        )
        recovery_input = tuple(
            value.to(device) for value in model_inputs(data, recovery_index)
        )
        prefix_valid = data["valid"][prefix_index].to(device)
        recovery_valid = data["valid"][recovery_index].to(device)
        recovery_target = data["target"][recovery_index].to(device)
        optimizer.zero_grad(set_to_none=True)
        prefix_prediction = model(*prefix_input)
        with torch.no_grad():
            prefix_baseline = baseline(*prefix_input)
        recovery_prediction = model(*recovery_input)
        prefix_numerator, prefix_denominator = normalized_l1_sum(
            prefix_prediction, prefix_baseline, prefix_valid
        )
        recovery_numerator, recovery_denominator = normalized_l1_sum(
            recovery_prediction, recovery_target, recovery_valid
        )
        prefix_loss = prefix_numerator / prefix_denominator.clamp_min(1.0)
        recovery_loss = recovery_numerator / recovery_denominator.clamp_min(1.0)
        loss = prefix_loss + recovery_loss
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.recovery_head.parameters()
        ):
            raise RuntimeError(f"nonfinite residual gradient at step {step + 1}")
        optimizer.step()
        if step == 0 or (step + 1) % 500 == 0:
            history.append(
                {
                    "step": step + 1,
                    "prefix_distill_normalized_mae": float(prefix_loss.detach()),
                    "recovery_normalized_mae": float(recovery_loss.detach()),
                    "total_loss": float(loss.detach()),
                }
            )

    model.eval()
    scale = action_std.to(device)

    def evaluate(begin: int, end: int) -> dict[str, float]:
        teacher_sum = 0.0
        baseline_drift_sum = 0.0
        count = 0.0
        with torch.no_grad():
            for offset in range(begin, end, 64):
                index = torch.arange(offset, min(end, offset + 64))
                inputs = tuple(value.to(device) for value in model_inputs(data, index))
                prediction = model(*inputs)
                baseline_prediction = baseline(*inputs)
                target = data["target"][index].to(device)
                valid = data["valid"][index].to(device).unsqueeze(-1).expand_as(
                    prediction
                )
                teacher_sum += float(
                    torch.abs((prediction - target) * scale)[valid].sum()
                )
                baseline_drift_sum += float(
                    torch.abs((prediction - baseline_prediction) * scale)[valid].sum()
                )
                count += float(valid.sum())
        return {
            "teacher_raw_mae": teacher_sum / count,
            "baseline_drift_raw_mae": baseline_drift_sum / count,
        }

    prefix_metrics = evaluate(0, PREFIX_ROWS)
    recovery_metrics = evaluate(PREFIX_ROWS, TOTAL_ROWS)
    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "seed": SEED,
            "input_schema": "O127C-frozen-plus-D180C-time-recovery-residual",
            "training_rows": TOTAL_ROWS,
            "baseline_frozen": True,
        },
        checkpoint,
    )
    reloaded = OperationMLPRecoveryResidual(dropout=DROPOUT).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    reloaded.eval()
    with torch.no_grad():
        reload_error = float(
            torch.max(torch.abs(model(*parity_inputs) - reloaded(*parity_inputs)))
        )
    metrics = {
        "step0_output_parity_max_abs": step0_parity,
        "prefix_teacher_raw_mae": prefix_metrics["teacher_raw_mae"],
        "prefix_baseline_drift_raw_mae": prefix_metrics[
            "baseline_drift_raw_mae"
        ],
        "recovery_teacher_raw_mae": recovery_metrics["teacher_raw_mae"],
        "recovery_baseline_drift_raw_mae": recovery_metrics[
            "baseline_drift_raw_mae"
        ],
        "residual_weight_norm": float(model.recovery_head.weight.norm().detach()),
        "prefix_exposure_min": int(prefix_exposure.min()),
        "prefix_exposure_max": int(prefix_exposure.max()),
        "recovery_exposure_min": int(recovery_exposure.min()),
        "recovery_exposure_max": int(recovery_exposure.max()),
        "reload_max_abs": reload_error,
        "wall_seconds": time.perf_counter() - started,
    }
    checks = {
        "d180c_passed": json.loads(
            (d180 / "summary.json").read_text(encoding="utf-8")
        ).get("status")
        == "passed",
        "rows_1152": data["state"].shape == (TOTAL_ROWS, 81),
        "step0_output_parity": step0_parity == 0.0,
        "baseline_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.baseline.parameters()
        ),
        "only_residual_optimized": set(
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        == {id(parameter) for parameter in model.recovery_head.parameters()},
        "balanced_batch_32_32": prefix_indices.shape == recovery_indices.shape
        == (STEPS, HALF_BATCH),
        "sampling_indices_in_range": bool(
            (prefix_indices >= 0).all()
            and (prefix_indices < PREFIX_ROWS).all()
            and (recovery_indices >= PREFIX_ROWS).all()
            and (recovery_indices < TOTAL_ROWS).all()
        ),
        "balanced_exposure_contract": SAMPLING_MODE
        != "epoch_balanced_without_replacement"
        or (
            int(prefix_exposure.max() - prefix_exposure.min()) <= 1
            and int(recovery_exposure.max() - recovery_exposure.min()) == 0
        ),
        "finite": bool(np.isfinite(list(metrics.values())).all()),
        "reload_exact": reload_error == 0.0,
    }
    passed = all(checks.values())
    atomic_json(out / "history.json", {"history": history})
    atomic_json(out / "offline_metrics.json", metrics)
    atomic_json(
        out / "training_config.json",
        {
            "seed": SEED,
            "steps": STEPS,
            "prefix_batch": HALF_BATCH,
            "recovery_batch": HALF_BATCH,
            "prefix_loss": "distill frozen O127C output",
            "recovery_loss": "time-aligned D180C teacher",
            "optimized_parameters": "zero-init linear recovery head only",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "sampling_mode": SAMPLING_MODE,
            "prefix_exposure_min": metrics["prefix_exposure_min"],
            "prefix_exposure_max": metrics["prefix_exposure_max"],
            "recovery_exposure_min": metrics["recovery_exposure_min"],
            "recovery_exposure_max": metrics["recovery_exposure_max"],
        },
    )
    atomic_json(
        out / "command.json",
        {"argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    )
    atomic_json(
        out / "run_manifest.json",
        {
            "run_id": RUN_ID,
            "input_sha256": sha256_file(source_path),
            "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    atomic_json(
        out / "forbidden_feature_audit.json",
        json.loads(
            (d180 / "forbidden_feature_audit.json").read_text(encoding="utf-8")
        ),
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": SCIENTIFIC_SCOPE,
        "metrics": metrics,
        "checks": checks,
        "decision": "residual isolation valid; run CAL/live comparison"
        if passed
        else "repair residual isolation",
        "next_run_ids": NEXT_RUN_IDS if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": QUEUE_RUN_ID, "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

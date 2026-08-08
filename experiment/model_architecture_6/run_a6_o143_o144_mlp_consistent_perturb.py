#!/usr/bin/env python3
"""Train MLP policies with label-consistent arm state/command offsets."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O143C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O144C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    normalized_l1_sum,
)

CONFIGS = {
    "1x": {
        "scale": 1.0,
        "root": JOINTTRAIN_ARCH6_O143C_RESULT_ROOT,
        "run_id": "a6_o143c_mlp_state_command_perturb_1x_v1",
    },
    "3x": {
        "scale": 3.0,
        "root": JOINTTRAIN_ARCH6_O144C_RESULT_ROOT,
        "run_id": "a6_o144c_mlp_state_command_perturb_3x_v1",
    },
}
STEPS = 6000
BATCH = 64
AUGMENT_PROBABILITY = 0.5
ARM_DIM = 7


def apply_offset(
    state: torch.Tensor, target_raw: torch.Tensor, offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented_state = state.clone()
    for history_start in (0, 9, 18, 27):
        augmented_state[:, history_start : history_start + ARM_DIM] += offset
    augmented_state[:, 72 : 72 + ARM_DIM] += offset
    augmented_target = target_raw.clone()
    augmented_target[:, :, :ARM_DIM] -= offset[:, None, :]
    return augmented_state, augmented_target


def evaluate(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    std: torch.Tensor,
    device: torch.device,
    *,
    state: torch.Tensor,
    target_raw: torch.Tensor,
) -> tuple[float, float]:
    normalized_total = 0.0
    raw_total = 0.0
    valid_total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(state), BATCH):
            stop = start + BATCH
            batch = {
                key: value[start:stop].to(device)
                for key, value in data.items()
                if key not in {"state", "target_raw"}
            }
            batch_state = state[start:stop].to(device)
            batch_target_raw = target_raw[start:stop].to(device)
            batch_target = batch_target_raw / std
            prediction = model(
                batch["point_cloud"],
                batch["target_mask"],
                batch["affordance"],
                batch_state,
                batch["context"],
            )
            numerator, denominator = normalized_l1_sum(
                prediction, batch_target, batch["valid"]
            )
            mask = batch["valid"].unsqueeze(-1).expand_as(prediction)
            normalized_total += float(numerator)
            valid_total += float(denominator)
            raw_total += float(torch.abs(prediction * std - batch_target_raw)[mask].sum())
    return normalized_total / valid_total, raw_total / valid_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()
    config = CONFIGS[args.scale]
    scale = float(config["scale"])
    out = Path(config["root"])
    out.mkdir(parents=True, exist_ok=True)

    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    std = torch.tensor(
        json.load(open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json"))["std"],
        dtype=torch.float32,
    ).reshape(1, 1, 9)
    data = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"]),
        "target_mask": torch.from_numpy(arrays["target_mask"]),
        "affordance": torch.from_numpy(arrays["zero_affordance"]),
        "state": torch.from_numpy(arrays["state_history"]),
        "context": torch.from_numpy(arrays["context"]),
        "target_raw": torch.from_numpy(arrays["command_delta_target"]),
        "valid": torch.from_numpy(arrays["action_valid"]),
    }
    tracking_residual = data["state"][:, 27:34] - data["state"][:, 72:79]
    base_sigma = tracking_residual.std(dim=0, unbiased=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("consistent perturbation training requires CUDA")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    index_generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(0, len(data["state"]), (STEPS, BATCH), generator=index_generator)
    augmentation_generator = torch.Generator().manual_seed(SEED + 143)
    standard_noise = torch.randn(
        (STEPS, BATCH, ARM_DIM), generator=augmentation_generator
    ).clamp(-3.0, 3.0)
    apply_mask = (
        torch.rand((STEPS, BATCH, 1), generator=augmentation_generator)
        < AUGMENT_PROBABILITY
    )
    offsets = standard_noise * (base_sigma * scale).reshape(1, 1, ARM_DIM)
    offsets *= apply_mask

    model = OperationMLPAbsolute(dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []
    started = time.perf_counter()
    std = std.to(device)
    model.train()
    for step in range(STEPS):
        index = indices[step]
        batch = {key: value[index].to(device) for key, value in data.items()}
        offset = offsets[step].to(device)
        augmented_state, augmented_target_raw = apply_offset(
            batch["state"], batch["target_raw"], offset
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch["point_cloud"],
            batch["target_mask"],
            batch["affordance"],
            augmented_state,
            batch["context"],
        )
        numerator, denominator = normalized_l1_sum(
            prediction, augmented_target_raw / std, batch["valid"]
        )
        loss = numerator / denominator.clamp_min(1)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 500 == 0:
            history.append(
                {
                    "step": step + 1,
                    "train_batch_normalized_delta_mae": float(loss.detach()),
                }
            )

    clean_normalized_mae, clean_raw_mae = evaluate(
        model,
        data,
        std,
        device,
        state=data["state"],
        target_raw=data["target_raw"],
    )
    stress_generator = torch.Generator().manual_seed(SEED + 144)
    stress_noise = torch.randn(
        (len(data["state"]), ARM_DIM), generator=stress_generator
    ).clamp(-3.0, 3.0)
    stress_offset = stress_noise * (base_sigma * scale).reshape(1, ARM_DIM)
    stress_state, stress_target = apply_offset(
        data["state"], data["target_raw"], stress_offset
    )
    stress_normalized_mae, stress_raw_mae = evaluate(
        model,
        data,
        std,
        device,
        state=stress_state,
        target_raw=stress_target,
    )
    original_absolute = data["state"][:, 72:81, None].transpose(1, 2) + data[
        "target_raw"
    ]
    augmented_absolute = stress_state[:, 72:81, None].transpose(1, 2) + stress_target
    invariant_error = float(torch.max(torch.abs(original_absolute - augmented_absolute)))

    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder": "mlp",
            "seed": SEED,
            "input_schema": "D042C-zero-contact-consistent-arm-perturbation",
            "training_rows": 1024,
            "perturb_scale": scale,
            "base_sigma": base_sigma.tolist(),
            "augment_probability": AUGMENT_PROBABILITY,
        },
        checkpoint,
    )
    checks = {
        "train_rows_1024": len(data["state"]) == 1024,
        "zero_contact": not bool(torch.count_nonzero(data["context"][:, :34])),
        "arm_only_offsets": not bool(torch.count_nonzero(stress_target[:, :, 7:] - data["target_raw"][:, :, 7:])),
        "absolute_target_invariant_le_1e_6": invariant_error <= 1e-6,
        "steps_6000": history[-1]["step"] == STEPS,
        "finite": bool(
            np.isfinite(
                [clean_normalized_mae, clean_raw_mae, stress_normalized_mae, stress_raw_mae]
            ).all()
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": "D042C TRAIN1024 MLP consistent state/command perturbation",
        "configuration": {
            "scale": scale,
            "base_sigma": base_sigma.tolist(),
            "augment_probability": AUGMENT_PROBABILITY,
            "offset_clip_sigma": 3.0,
            "arm_dim": ARM_DIM,
            "seed": SEED,
            "steps": STEPS,
            "batch_size": BATCH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "metrics": {
            "clean_train_normalized_delta_mae": clean_normalized_mae,
            "clean_train_raw_delta_mae": clean_raw_mae,
            "stress_train_normalized_delta_mae": stress_normalized_mae,
            "stress_train_raw_delta_mae": stress_raw_mae,
            "absolute_target_invariance_max_error": invariant_error,
            "wall_seconds": time.perf_counter() - started,
        },
        "checks": checks,
        "decision": (
            "perturbation fit valid; run frozen clean/perturbed CAL"
            if passed
            else "perturbation training invalid"
        ),
    }
    atomic_json(out / "history.json", {"history": history})
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

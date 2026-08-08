#!/usr/bin/env python3
"""Evaluate baseline/perturbed MLPs on clean and consistently shifted CAL inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O143C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O144C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O145C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json
from run_a6_o143_o144_mlp_consistent_perturb import ARM_DIM, apply_offset

MODELS = {
    "baseline": Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
    "perturb_1x": Path(JOINTTRAIN_ARCH6_O143C_RESULT_ROOT) / "last.pt",
    "perturb_3x": Path(JOINTTRAIN_ARCH6_O144C_RESULT_ROOT) / "last.pt",
}
CONTEXT_SCALES = {"clean": 0.0, "shift_1x": 1.0, "shift_3x": 3.0}


def bootstrap(values: np.ndarray) -> dict:
    rng = np.random.default_rng(20260806)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def main() -> int:
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz",
        allow_pickle=False,
    ) as train_source:
        train_state = torch.from_numpy(np.asarray(train_source["state_history"]))
    base_sigma = (train_state[:, 27:34] - train_state[:, 72:79]).std(
        dim=0, unbiased=False
    )
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "cal_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    rows = json.load(
        open(Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT) / "full" / "input_manifest.json")
    )["rows"]
    names = np.asarray([row["target"] for row in rows])
    unique_targets = sorted(set(names))
    std = torch.tensor(
        json.load(open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json"))["std"],
        dtype=torch.float32,
    ).reshape(1, 1, 9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = {
        key: torch.from_numpy(arrays[key]).to(device)
        for key in (
            "point_cloud",
            "target_mask",
            "zero_affordance",
            "state_history",
            "context",
            "command_delta_target",
            "action_valid",
        )
    }
    std = std.to(device)
    base_sigma = base_sigma.to(device)
    generator = torch.Generator().manual_seed(20260806 + 145)
    standard_noise = torch.randn((len(rows), ARM_DIM), generator=generator).clamp(-3, 3)
    standard_noise = standard_noise.to(device)
    context_inputs = {}
    invariance = {}
    for context_name, scale in CONTEXT_SCALES.items():
        offset = standard_noise * base_sigma.reshape(1, ARM_DIM) * scale
        state, target = apply_offset(
            tensors["state_history"], tensors["command_delta_target"], offset
        )
        context_inputs[context_name] = (state, target)
        original_absolute = (
            tensors["state_history"][:, None, 72:81]
            + tensors["command_delta_target"]
        )
        shifted_absolute = state[:, None, 72:81] + target
        invariance[context_name] = float(
            torch.max(torch.abs(original_absolute - shifted_absolute)).cpu()
        )

    predictions = {}
    for model_name, path in MODELS.items():
        model = OperationMLPAbsolute().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["model"], strict=True
        )
        model.eval()
        predictions[model_name] = {}
        with torch.no_grad():
            for context_name, (state, _) in context_inputs.items():
                predictions[model_name][context_name] = model(
                    tensors["point_cloud"],
                    tensors["target_mask"],
                    tensors["zero_affordance"],
                    state,
                    tensors["context"],
                ) * std

    metrics = {}
    target_errors = {}
    valid = tensors["action_valid"]
    for model_name, by_context in predictions.items():
        metrics[model_name] = {}
        target_errors[model_name] = {}
        for context_name, prediction in by_context.items():
            target = context_inputs[context_name][1]
            error = torch.abs(prediction - target)
            mask = valid.unsqueeze(-1).expand_as(error)
            prefix_mask = valid[:, :8].unsqueeze(-1).expand_as(error[:, :8])
            row = (error * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
            row_values = row.cpu().numpy()
            per_target = np.asarray(
                [row_values[names == target_name].mean() for target_name in unique_targets]
            )
            prefix_row = (error[:, :8] * prefix_mask).sum((1, 2)) / prefix_mask.sum(
                (1, 2)
            ).clamp_min(1)
            prefix_values = prefix_row.cpu().numpy()
            per_target_prefix = np.asarray(
                [prefix_values[names == target_name].mean() for target_name in unique_targets]
            )
            target_errors[model_name][context_name] = {
                "full": per_target,
                "prefix8": per_target_prefix,
            }
            metrics[model_name][context_name] = {
                "full_raw_mae": float(error[mask].mean().cpu()),
                "prefix8_raw_mae": float(error[:, :8][prefix_mask].mean().cpu()),
            }

    comparisons = {}
    for context_name in CONTEXT_SCALES:
        baseline = target_errors["baseline"][context_name]
        comparisons[context_name] = {
            f"{model_name}_minus_baseline": {
                metric_name: bootstrap(
                    target_errors[model_name][context_name][metric_name]
                    - baseline[metric_name]
                )
                for metric_name in ("full", "prefix8")
            }
            for model_name in ("perturb_1x", "perturb_3x")
        }

    checks = {
        "rows_280": len(rows) == 280,
        "targets_35": len(unique_targets) == 35,
        "train_only_sigma": True,
        "absolute_target_invariant": all(value <= 1e-6 for value in invariance.values()),
        "finite": all(
            np.isfinite(list(context_metrics.values())).all()
            for model_metrics in metrics.values()
            for context_metrics in model_metrics.values()
        ),
        "zero_contact": not bool(torch.count_nonzero(tensors["context"][:, :34])),
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_O145C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "run_id": "a6_o145c_perturb_cal_screen_v1",
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": "frozen clean/shifted A5_CAL perturbation diagnostic",
        "configuration": {
            "base_sigma": base_sigma.cpu().tolist(),
            "context_scales": CONTEXT_SCALES,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "absolute_target_invariance_max_error": invariance,
        "checks": checks,
        "decision": (
            "CAL perturbation matrix valid; run exact-paired live8"
            if passed
            else "perturbation CAL invalid"
        ),
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

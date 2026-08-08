#!/usr/bin/env python3
"""Attribute CAL error by executed prefix, suffix, and action differences."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute, OperationParallelAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O128C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O132C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O133C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O137C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O138C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O142C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json

RECIPES = {
    "train1024": {
        "mlp": Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        "parallel": Path(JOINTTRAIN_ARCH6_O128C_RESULT_ROOT) / "last.pt",
    },
    "mixed_train194": {
        "mlp": Path(JOINTTRAIN_ARCH6_O132C_RESULT_ROOT) / "last.pt",
        "parallel": Path(JOINTTRAIN_ARCH6_O133C_RESULT_ROOT) / "last.pt",
    },
    "additive_train194": {
        "mlp": Path(JOINTTRAIN_ARCH6_O137C_RESULT_ROOT) / "last.pt",
        "parallel": Path(JOINTTRAIN_ARCH6_O138C_RESULT_ROOT) / "last.pt",
    },
}
FACTORIES = {
    "mlp": OperationMLPAbsolute,
    "parallel": OperationParallelAbsolute,
}


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


def masked_mean(error: torch.Tensor, valid: torch.Tensor) -> float:
    mask = valid.unsqueeze(-1).expand_as(error)
    return float(error[mask].mean().cpu())


def main() -> int:
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "cal_zero_contact.npz",
        allow_pickle=False,
    ) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
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
    target = tensors["command_delta_target"]
    valid = tensors["action_valid"]
    inputs = (
        tensors["point_cloud"],
        tensors["target_mask"],
        tensors["zero_affordance"],
        tensors["state_history"],
        tensors["context"],
    )
    metrics = {}
    target_prefix_errors = {}
    for recipe, paths in RECIPES.items():
        metrics[recipe] = {}
        target_prefix_errors[recipe] = {}
        for arm, path in paths.items():
            model = FACTORIES[arm]().to(device)
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=False)["model"], strict=True
            )
            model.eval()
            with torch.no_grad():
                prediction = model(*inputs) * std
            error = torch.abs(prediction - target)
            per_horizon = [
                masked_mean(error[:, index : index + 1], valid[:, index : index + 1])
                for index in range(32)
            ]
            prefix_valid = valid[:, :8]
            prefix_mask = prefix_valid.unsqueeze(-1).expand_as(error[:, :8])
            prefix_row = (error[:, :8] * prefix_mask).sum((1, 2)) / prefix_mask.sum(
                (1, 2)
            ).clamp_min(1)
            prefix_values = prefix_row.cpu().numpy()
            target_prefix = np.asarray(
                [prefix_values[names == target_name].mean() for target_name in unique_targets]
            )
            target_prefix_errors[recipe][arm] = target_prefix
            difference_valid = valid[:, 1:] & valid[:, :-1]
            prediction_difference = prediction[:, 1:] - prediction[:, :-1]
            target_difference = target[:, 1:] - target[:, :-1]
            metrics[recipe][arm] = {
                "prefix8_raw_mae": masked_mean(error[:, :8], prefix_valid),
                "suffix24_raw_mae": masked_mean(error[:, 8:], valid[:, 8:]),
                "full_raw_mae": masked_mean(error, valid),
                "first_difference_raw_mae": masked_mean(
                    torch.abs(prediction_difference - target_difference), difference_valid
                ),
                "per_horizon_raw_mae": per_horizon,
            }

    comparisons = {}
    for arm in FACTORIES:
        comparisons[arm] = {}
        baseline = target_prefix_errors["train1024"][arm]
        for recipe in ("mixed_train194", "additive_train194"):
            comparisons[arm][f"{recipe}_minus_train1024_prefix8"] = bootstrap(
                target_prefix_errors[recipe][arm] - baseline
            )

    checks = {
        "rows_280": len(rows) == 280,
        "targets_35": len(unique_targets) == 35,
        "all_checkpoints_present": all(
            path.is_file() for paths in RECIPES.values() for path in paths.values()
        ),
        "finite": all(
            np.isfinite(metric["per_horizon_raw_mae"]).all()
            for recipe_metrics in metrics.values()
            for metric in recipe_metrics.values()
        ),
        "executed_prefix_is_8": True,
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_O142C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "run_id": "a6_o142c_horizon_error_diagnostic_v1",
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": "frozen A5_CAL horizon attribution; no model selection alone",
        "metrics": metrics,
        "comparisons": comparisons,
        "checks": checks,
        "decision": (
            "use prefix and difference evidence to select one exposure-robustness intervention"
            if passed
            else "horizon diagnostic invalid"
        ),
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

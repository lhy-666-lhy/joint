#!/usr/bin/env python3
"""Evaluate additive TRAIN194 checkpoints on the unchanged A5_CAL input."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o125c_zero_contact_cal_screen as evaluator
from a6_operation_models import (
    OperationCausalAbsolute,
    OperationMLPAbsolute,
    OperationParallelAbsolute,
)
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O124C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O128C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O137C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O138C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O139C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


def additive_vs_train1024() -> dict:
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
    mask = valid.unsqueeze(-1).expand_as(target)
    inputs = (
        tensors["point_cloud"],
        tensors["target_mask"],
        tensors["zero_affordance"],
        tensors["state_history"],
        tensors["context"],
    )
    checkpoints = {
        "mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
            Path(JOINTTRAIN_ARCH6_O137C_RESULT_ROOT) / "last.pt",
        ),
        "parallel": (
            OperationParallelAbsolute,
            Path(JOINTTRAIN_ARCH6_O128C_RESULT_ROOT) / "last.pt",
            Path(JOINTTRAIN_ARCH6_O138C_RESULT_ROOT) / "last.pt",
        ),
    }
    result = {}
    for arm, (factory, baseline_path, additive_path) in checkpoints.items():
        per_checkpoint = []
        for path in (baseline_path, additive_path):
            model = factory().to(device)
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=False)["model"], strict=True
            )
            model.eval()
            with torch.no_grad():
                prediction = model(*inputs)
                error = torch.abs(prediction * std - target)
                row_error = (error * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
            row_values = row_error.cpu().numpy()
            per_checkpoint.append(
                np.asarray(
                    [row_values[names == target_name].mean() for target_name in unique_targets]
                )
            )
        baseline_values, additive_values = per_checkpoint
        result[arm] = {
            "additive_minus_train1024": evaluator.boot(
                additive_values - baseline_values
            )
        }
    return result


def main() -> int:
    evaluator.ARMS = {
        "mlp": (OperationMLPAbsolute, Path(JOINTTRAIN_ARCH6_O137C_RESULT_ROOT) / "last.pt"),
        "parallel": (
            OperationParallelAbsolute,
            Path(JOINTTRAIN_ARCH6_O138C_RESULT_ROOT) / "last.pt",
        ),
        "causal": (
            OperationCausalAbsolute,
            Path(JOINTTRAIN_ARCH6_O124C_RESULT_ROOT) / "last.pt",
        ),
    }
    evaluator.JOINTTRAIN_ARCH6_O125C_RESULT_ROOT = JOINTTRAIN_ARCH6_O139C_RESULT_ROOT
    code = evaluator.main()
    out = Path(JOINTTRAIN_ARCH6_O139C_RESULT_ROOT)
    summary = json.load(open(out / "summary.json"))
    summary["run_id"] = "a6_o139c_additive_train194_cal_v1"
    summary["scientific_scope"] = (
        "unchanged A5_CAL zero-contact screen for D044C additive TRAIN194"
    )
    summary["checkpoint_scope"] = {
        "mlp": "A6-O137C additive TRAIN194",
        "parallel": "A6-O138C additive TRAIN194",
        "causal": "fixed64 diagnostic",
    }
    summary["checkpoint_pairwise"] = additive_vs_train1024()
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

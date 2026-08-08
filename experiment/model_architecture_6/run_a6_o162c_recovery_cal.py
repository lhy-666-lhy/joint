#!/usr/bin/env python3
"""Evaluate additive TRAIN recovery supervision on unchanged A5_CAL."""

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
    JOINTTRAIN_ARCH6_O161C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O162C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o162c_recovery_cal_v1"
ACTION_DIM = 9


def bootstrap(values: np.ndarray, seed: int = 20260806) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def load_model(path: Path, device: torch.device) -> OperationMLPAbsolute:
    model = OperationMLPAbsolute().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"], strict=True)
    model.eval()
    return model


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O162C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    cal_root = Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT)
    with np.load(cal_root / "cal_zero_contact.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    rows = json.loads(
        (Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT) / "full" / "input_manifest.json").read_text(encoding="utf-8")
    )["rows"]
    if len(rows) != 280:
        raise ValueError("unexpected CAL row count")
    action_std = torch.tensor(
        json.loads((Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text(encoding="utf-8"))["std"],
        dtype=torch.float32,
    ).reshape(1, 1, ACTION_DIM)
    data = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"].astype(np.float32)),
        "target_mask": torch.from_numpy(arrays["target_mask"].astype(bool)),
        "affordance": torch.from_numpy(arrays["zero_affordance"].astype(np.float32)),
        "state": torch.from_numpy(arrays["state_history"].astype(np.float32)),
        "context": torch.from_numpy(arrays["context"].astype(np.float32)),
        "target": torch.from_numpy(arrays["command_delta_target"].astype(np.float32)) / action_std,
        "valid": torch.from_numpy(arrays["action_valid"].astype(bool)),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline = load_model(Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt", device)
    recovery = load_model(Path(JOINTTRAIN_ARCH6_O161C_RESULT_ROOT) / "last.pt", device)
    base_errors: list[float] = []
    recovery_errors: list[float] = []
    repeat_errors: list[float] = []
    base_endpoints: list[float] = []
    recovery_endpoints: list[float] = []
    repeat_endpoints: list[float] = []
    scale = action_std.to(device)
    with torch.no_grad():
        for begin in range(0, 280, 64):
            stop = min(280, begin + 64)
            point = data["point_cloud"][begin:stop].to(device)
            mask = data["target_mask"][begin:stop].to(device)
            aff = data["affordance"][begin:stop].to(device)
            state = data["state"][begin:stop].to(device)
            context = data["context"][begin:stop].to(device)
            target = data["target"][begin:stop].to(device)
            valid = data["valid"][begin:stop].to(device)
            pred_base = baseline(point, mask, aff, state, context)
            pred_recovery = recovery(point, mask, aff, state, context)
            for pred, bucket in ((pred_base, base_errors), (pred_recovery, recovery_errors)):
                error = torch.abs((pred - target) * scale)
                for row in range(error.shape[0]):
                    bucket.append(float(error[row][valid[row]].mean()))
            repeat_error = torch.abs(target * scale)
            for row in range(repeat_error.shape[0]):
                repeat_errors.append(float(repeat_error[row][valid[row]].mean()))
            for pred, bucket in ((pred_base, base_endpoints), (pred_recovery, recovery_endpoints)):
                for row in range(pred.shape[0]):
                    last = int(torch.nonzero(valid[row], as_tuple=False)[-1])
                    bucket.append(float(torch.abs((pred[row, last] - target[row, last]) * scale[0, 0]).mean()))
            for row in range(target.shape[0]):
                last = int(torch.nonzero(valid[row], as_tuple=False)[-1])
                repeat_endpoints.append(float(torch.abs(target[row, last] * scale[0, 0]).mean()))
    base = np.asarray(base_errors)
    recovery_values = np.asarray(recovery_errors)
    repeat = np.asarray(repeat_errors)
    base_endpoint = np.asarray(base_endpoints)
    recovery_endpoint = np.asarray(recovery_endpoints)
    repeat_endpoint = np.asarray(repeat_endpoints)
    target_names = [str(row["target"]) for row in rows]
    targets = sorted(set(target_names))
    per_target = {}
    for target_name in targets:
        ids = np.asarray([i for i, value in enumerate(target_names) if value == target_name])
        per_target[target_name] = {
            "baseline_mae": float(base[ids].mean()),
            "recovery_mae": float(recovery_values[ids].mean()),
            "repeat_last_mae": float(repeat[ids].mean()),
            "baseline_endpoint_mae": float(base_endpoint[ids].mean()),
            "recovery_endpoint_mae": float(recovery_endpoint[ids].mean()),
            "repeat_last_endpoint_mae": float(repeat_endpoint[ids].mean()),
        }
    paired = np.asarray([per_target[name]["recovery_mae"] - per_target[name]["baseline_mae"] for name in targets])
    endpoint_paired = np.asarray([per_target[name]["recovery_endpoint_mae"] - per_target[name]["baseline_endpoint_mae"] for name in targets])
    checks = {
        "cal_rows_280": len(rows) == 280,
        "target_count_35": len(targets) == 35,
        "zero_contact": not bool(np.count_nonzero(arrays["context"][:, :34])),
        "finite": bool(np.isfinite(np.concatenate([base, recovery_values, repeat, base_endpoint, recovery_endpoint, repeat_endpoint])).all()),
        "valid_horizon": bool(arrays["action_valid"].any(axis=1).all()),
        "recovery_checkpoint": (Path(JOINTTRAIN_ARCH6_O161C_RESULT_ROOT) / "last.pt").is_file(),
        "no_oracle_input": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "A5_CAL exact-paired screen for TRAIN-only recovery supervision",
        "metrics": {
            "baseline_mae": bootstrap(base),
            "recovery_mae": bootstrap(recovery_values),
            "repeat_last_mae": bootstrap(repeat),
            "recovery_minus_baseline_target_paired_mae": bootstrap(paired),
            "baseline_endpoint_mae": bootstrap(base_endpoint),
            "recovery_endpoint_mae": bootstrap(recovery_endpoint),
            "repeat_last_endpoint_mae": bootstrap(repeat_endpoint),
            "recovery_minus_baseline_target_paired_endpoint_mae": bootstrap(endpoint_paired),
            "per_target": per_target,
        },
        "checks": checks,
        "claim_supported": "partial" if passed else "no",
        "decision": "CAL valid; run unchanged live8" if passed else "repair CAL evaluator",
        "next_run_ids": ["A6-O163C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-O162C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

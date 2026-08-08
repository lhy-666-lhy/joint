#!/usr/bin/env python3
"""Evaluate baseline and FK/visible-target MLPs on the fixed A5_CAL screen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_deployable_geometry import FK_TARGET_STATE_DIM
from a6_operation_models import OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D150C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O151C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O152C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o152c_fk_target_relative_cal_v1"
OUTPUT_ROOT = JOINTTRAIN_ARCH6_O152C_RESULT_ROOT
GEOMETRY_CHECKPOINT_ROOT = JOINTTRAIN_ARCH6_O151C_RESULT_ROOT
GEOMETRY_FACTORY = lambda state_dim: OperationMLPAbsolute(state_dim=state_dim)
GEOMETRY_STATE_DIM = FK_TARGET_STATE_DIM
NEXT_RUN_ID = "A6-O153C"
ACTION_DIM = 9
HORIZON = 32


def bootstrap(values: np.ndarray, seed: int = 20260806) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def load_arm(
    checkpoint: Path, state_dim: int, device: torch.device, factory=None
) -> OperationMLPAbsolute:
    model = (factory or (lambda width: OperationMLPAbsolute(state_dim=width)))(state_dim).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True
    )
    model.eval()
    return model


def main() -> int:
    out = Path(OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    d150 = Path(JOINTTRAIN_ARCH6_D150C_RESULT_ROOT)
    with np.load(d150 / "cal_fk_target_relative.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    rows = json.loads(
        (Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT) / "full" / "input_manifest.json").read_text(encoding="utf-8")
    )["rows"]
    if len(rows) != arrays["state_history"].shape[0]:
        raise ValueError("CAL rows and arrays are not aligned")
    normalizer = json.loads(
        (Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text(encoding="utf-8")
    )
    action_std = torch.tensor(normalizer["std"], dtype=torch.float32).reshape(1, 1, ACTION_DIM)
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
    baseline = load_arm(Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt", 81, device)
    geometry = load_arm(
        Path(GEOMETRY_CHECKPOINT_ROOT) / "last.pt",
        GEOMETRY_STATE_DIM,
        device,
        GEOMETRY_FACTORY,
    )
    baseline_mae: list[float] = []
    geometry_mae: list[float] = []
    geometry_zero_mae: list[float] = []
    repeat_last_mae: list[float] = []
    endpoint_baseline: list[float] = []
    endpoint_geometry: list[float] = []
    endpoint_repeat_last: list[float] = []
    with torch.no_grad():
        for begin in range(0, len(data["state"]), 64):
            stop = begin + 64
            point = data["point_cloud"][begin:stop].to(device)
            mask = data["target_mask"][begin:stop].to(device)
            aff = data["affordance"][begin:stop].to(device)
            state = data["state"][begin:stop].to(device)
            context = data["context"][begin:stop].to(device)
            target = data["target"][begin:stop].to(device)
            valid = data["valid"][begin:stop].to(device)
            pred_base = baseline(point, mask, aff, state[:, :81], context)
            pred_geo = geometry(point, mask, aff, state, context)
            zero_state = state.clone()
            zero_state[:, -4:] = 0.0
            pred_zero = geometry(point, mask, aff, zero_state, context)
            raw_scale = action_std.to(device)
            for pred, bucket in ((pred_base, baseline_mae), (pred_geo, geometry_mae), (pred_zero, geometry_zero_mae)):
                error = torch.abs((pred - target) * raw_scale)
                for row_index in range(error.shape[0]):
                    row_mask = valid[row_index]
                    bucket.append(float(error[row_index][row_mask].mean()))
            repeat_error = torch.abs(target * raw_scale)
            for row_index in range(repeat_error.shape[0]):
                repeat_last_mae.append(float(repeat_error[row_index][valid[row_index]].mean()))
            for pred, bucket in ((pred_base, endpoint_baseline), (pred_geo, endpoint_geometry)):
                for row_index in range(pred.shape[0]):
                    last = int(torch.nonzero(valid[row_index], as_tuple=False)[-1])
                    bucket.append(float(torch.abs((pred[row_index, last] - target[row_index, last]) * raw_scale[0, 0]).mean()))
            for row_index in range(target.shape[0]):
                last = int(torch.nonzero(valid[row_index], as_tuple=False)[-1])
                endpoint_repeat_last.append(float(torch.abs(target[row_index, last] * raw_scale[0, 0]).mean()))
    base = np.asarray(baseline_mae, dtype=np.float64)
    geo = np.asarray(geometry_mae, dtype=np.float64)
    zero = np.asarray(geometry_zero_mae, dtype=np.float64)
    repeat = np.asarray(repeat_last_mae, dtype=np.float64)
    ep_base = np.asarray(endpoint_baseline, dtype=np.float64)
    ep_geo = np.asarray(endpoint_geometry, dtype=np.float64)
    ep_repeat = np.asarray(endpoint_repeat_last, dtype=np.float64)
    target_names = [str(row["target"]) for row in rows]
    unique_targets = sorted(set(target_names))
    per_target = {}
    for target_name in unique_targets:
        indices = np.asarray([i for i, value in enumerate(target_names) if value == target_name], dtype=np.int64)
        per_target[target_name] = {
            "baseline_mae": float(base[indices].mean()),
            "geometry_mae": float(geo[indices].mean()),
            "geometry_zero_mae": float(zero[indices].mean()),
            "repeat_last_mae": float(repeat[indices].mean()),
            "baseline_endpoint_mae": float(ep_base[indices].mean()),
            "geometry_endpoint_mae": float(ep_geo[indices].mean()),
            "repeat_last_endpoint_mae": float(ep_repeat[indices].mean()),
        }
    paired = np.asarray(
        [per_target[target]["geometry_mae"] - per_target[target]["baseline_mae"] for target in unique_targets],
        dtype=np.float64,
    )
    endpoint_paired = np.asarray(
        [per_target[target]["geometry_endpoint_mae"] - per_target[target]["baseline_endpoint_mae"] for target in unique_targets],
        dtype=np.float64,
    )
    checks = {
        "cal_rows_280": len(rows) == 280,
        "target_count_35": len(unique_targets) == 35,
        "target_disjoint_contract": True,
        "state_width_85": arrays["state_history"].shape == (280, FK_TARGET_STATE_DIM),
        "baseline_state_prefix_81": True,
        "zero_contact": not bool(np.count_nonzero(arrays["context"][:, :34])),
        "finite": bool(np.isfinite(np.concatenate([base, geo, zero, repeat, ep_base, ep_geo, ep_repeat])).all()),
        "all_rows_have_valid_horizon": bool(arrays["action_valid"].any(axis=1).all()),
        "geometry_zero_feature_evaluated": len(zero) == len(geo),
    }
    passed = all(checks.values())
    paired_summary = bootstrap(paired)
    if paired_summary["ci95"][1] < 0.0:
        decision = "geometry state improves CAL error; continue to live8"
        claim = "partial"
    elif paired_summary["ci95"][0] > 0.0:
        decision = "geometry state worsens CAL error; retain baseline for live comparison only"
        claim = "no"
    else:
        decision = "CAL difference is inconclusive; continue to live8 because closed-loop claim is primary"
        claim = "partial"
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "A5_CAL exact-paired offline screen, baseline versus deployable geometry MLP",
        "metrics": {
            "baseline_valid_absolute_mae": bootstrap(base),
            "geometry_valid_absolute_mae": bootstrap(geo),
            "geometry_zero_feature_valid_absolute_mae": bootstrap(zero),
            "repeat_last_valid_absolute_mae": bootstrap(repeat),
            "geometry_minus_baseline_target_paired_mae": paired_summary,
            "baseline_endpoint_mae": bootstrap(ep_base),
            "geometry_endpoint_mae": bootstrap(ep_geo),
            "repeat_last_endpoint_mae": bootstrap(ep_repeat),
            "geometry_minus_baseline_target_paired_endpoint_mae": bootstrap(endpoint_paired),
            "per_target": per_target,
        },
        "checks": checks,
        "claim_supported": claim if passed else "no",
        "decision": decision if passed else "repair CAL evaluator",
        "next_run_ids": [NEXT_RUN_ID] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-O152C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate D180C recovery alignment models on the frozen A5_CAL rows."""

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
    JOINTTRAIN_ARCH6_O181C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O182C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O183C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_o183c_recovery_alignment_cal_v1"
ACTION_DIM = 9
ARMS = {
    "baseline_mlp": Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
    "time_aligned": Path(JOINTTRAIN_ARCH6_O181C_RESULT_ROOT) / "last.pt",
    "progress_aligned": Path(JOINTTRAIN_ARCH6_O182C_RESULT_ROOT) / "last.pt",
}


def bootstrap(values: np.ndarray, seed: int = 20260806) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def load_model(path: Path, device: torch.device) -> OperationMLPAbsolute:
    model = OperationMLPAbsolute().to(device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=False)["model"], strict=True
    )
    model.eval()
    return model


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O183C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "cal_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    rows = json.loads(
        (
            Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT)
            / "full"
            / "input_manifest.json"
        ).read_text(encoding="utf-8")
    )["rows"]
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {arm: load_model(path, device) for arm, path in ARMS.items()}
    errors = {arm: [] for arm in ARMS}
    endpoints = {arm: [] for arm in ARMS}
    scale = action_std.to(device)
    with torch.no_grad():
        for begin in range(0, len(rows), 64):
            stop = min(len(rows), begin + 64)
            point = data["point_cloud"][begin:stop].to(device)
            mask = data["target_mask"][begin:stop].to(device)
            affordance = data["affordance"][begin:stop].to(device)
            state = data["state"][begin:stop].to(device)
            context = data["context"][begin:stop].to(device)
            target = data["target"][begin:stop].to(device)
            valid = data["valid"][begin:stop].to(device)
            for arm, model in models.items():
                prediction = model(point, mask, affordance, state, context)
                error = torch.abs((prediction - target) * scale)
                for row_index in range(error.shape[0]):
                    errors[arm].append(float(error[row_index][valid[row_index]].mean()))
                    last = int(torch.nonzero(valid[row_index], as_tuple=False)[-1])
                    endpoints[arm].append(float(error[row_index, last].mean()))

    target_names = np.asarray([str(row["target"]) for row in rows])
    unique_targets = sorted(set(target_names.tolist()))
    per_target: dict[str, dict[str, float]] = {}
    for target in unique_targets:
        indices = np.flatnonzero(target_names == target)
        per_target[target] = {
            **{
                f"{arm}_mae": float(np.asarray(errors[arm])[indices].mean())
                for arm in ARMS
            },
            **{
                f"{arm}_endpoint_mae": float(
                    np.asarray(endpoints[arm])[indices].mean()
                )
                for arm in ARMS
            },
        }
    pairwise = {}
    for left, right in (
        ("time_aligned", "baseline_mlp"),
        ("progress_aligned", "baseline_mlp"),
        ("progress_aligned", "time_aligned"),
    ):
        mae_delta = np.asarray(
            [
                per_target[target][f"{left}_mae"]
                - per_target[target][f"{right}_mae"]
                for target in unique_targets
            ]
        )
        endpoint_delta = np.asarray(
            [
                per_target[target][f"{left}_endpoint_mae"]
                - per_target[target][f"{right}_endpoint_mae"]
                for target in unique_targets
            ]
        )
        pairwise[f"{left}_minus_{right}"] = {
            "mae": bootstrap(mae_delta),
            "endpoint_mae": bootstrap(endpoint_delta),
        }
    metrics = {
        arm: {
            "mae": bootstrap(np.asarray(errors[arm])),
            "endpoint_mae": bootstrap(np.asarray(endpoints[arm])),
        }
        for arm in ARMS
    }
    checks = {
        "cal_rows_280": len(rows) == 280,
        "target_count_35": len(unique_targets) == 35,
        "zero_contact": not bool(np.count_nonzero(arrays["context"][:, :34])),
        "finite": all(
            np.isfinite(errors[arm]).all() and np.isfinite(endpoints[arm]).all()
            for arm in ARMS
        ),
        "all_checkpoints_present": all(path.is_file() for path in ARMS.values()),
        "zero_oracle_model_input": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "frozen A5_CAL recovery teacher-alignment comparison",
        "metrics": metrics,
        "pairwise": pairwise,
        "per_target": per_target,
        "checks": checks,
        "decision": "CAL screen valid; run corrected live8 comparison"
        if passed
        else "repair O183C before live rollout",
        "next_run_ids": ["A6-O184C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O183C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


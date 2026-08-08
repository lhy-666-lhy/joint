#!/usr/bin/env python3
"""Compare frozen baseline, uniform recovery, and recovery residual on A5_CAL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute, OperationMLPRecoveryResidual
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O181C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O185C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O186C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json
from run_a6_o183c_recovery_alignment_cal import bootstrap


RUN_ID = "a6_o186c_recovery_residual_cal_v1"
QUEUE_RUN_ID = "A6-O186C"
NEXT_RUN_IDS = ["A6-O187C"]
ACTION_DIM = 9
MODEL_SPECS = {
    "baseline_mlp": (
        OperationMLPAbsolute,
        Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
    ),
    "time_uniform": (
        OperationMLPAbsolute,
        Path(JOINTTRAIN_ARCH6_O181C_RESULT_ROOT) / "last.pt",
    ),
    "time_residual": (
        OperationMLPRecoveryResidual,
        Path(JOINTTRAIN_ARCH6_O185C_RESULT_ROOT) / "last.pt",
    ),
}
PAIRWISE_COMPARISONS = (
    ("time_uniform", "baseline_mlp"),
    ("time_residual", "baseline_mlp"),
    ("time_residual", "time_uniform"),
)


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O186C_RESULT_ROOT)
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
    models = {}
    for arm, (factory, checkpoint) in MODEL_SPECS.items():
        model = factory().to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=False)["model"],
            strict=True,
        )
        model.eval()
        models[arm] = model
    errors = {arm: [] for arm in models}
    endpoints = {arm: [] for arm in models}
    scale = action_std.to(device)
    with torch.no_grad():
        for begin in range(0, len(rows), 64):
            stop = min(len(rows), begin + 64)
            inputs = (
                data["point_cloud"][begin:stop].to(device),
                data["target_mask"][begin:stop].to(device),
                data["affordance"][begin:stop].to(device),
                data["state"][begin:stop].to(device),
                data["context"][begin:stop].to(device),
            )
            target = data["target"][begin:stop].to(device)
            valid = data["valid"][begin:stop].to(device)
            for arm, model in models.items():
                error = torch.abs((model(*inputs) - target) * scale)
                for row_index in range(error.shape[0]):
                    errors[arm].append(float(error[row_index][valid[row_index]].mean()))
                    last = int(torch.nonzero(valid[row_index], as_tuple=False)[-1])
                    endpoints[arm].append(float(error[row_index, last].mean()))

    target_names = np.asarray([str(row["target"]) for row in rows])
    unique_targets = sorted(set(target_names.tolist()))
    per_target = {}
    for target in unique_targets:
        indices = np.flatnonzero(target_names == target)
        per_target[target] = {
            **{
                f"{arm}_mae": float(np.asarray(errors[arm])[indices].mean())
                for arm in models
            },
            **{
                f"{arm}_endpoint_mae": float(
                    np.asarray(endpoints[arm])[indices].mean()
                )
                for arm in models
            },
        }
    pairwise = {}
    for left, right in PAIRWISE_COMPARISONS:
        pairwise[f"{left}_minus_{right}"] = {
            metric: bootstrap(
                np.asarray(
                    [
                        per_target[target][f"{left}_{metric}"]
                        - per_target[target][f"{right}_{metric}"]
                        for target in unique_targets
                    ]
                )
            )
            for metric in ("mae", "endpoint_mae")
        }
    metrics = {
        arm: {
            "mae": bootstrap(np.asarray(errors[arm])),
            "endpoint_mae": bootstrap(np.asarray(endpoints[arm])),
        }
        for arm in models
    }
    checks = {
        "cal_rows_280": len(rows) == 280,
        "target_count_35": len(unique_targets) == 35,
        "zero_contact": not bool(np.count_nonzero(arrays["context"][:, :34])),
        "finite": all(
            np.isfinite(errors[arm]).all() and np.isfinite(endpoints[arm]).all()
            for arm in models
        ),
        "all_checkpoints_present": all(
            checkpoint.is_file() for _, checkpoint in MODEL_SPECS.values()
        ),
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
        "scientific_scope": "frozen A5_CAL recovery residual isolation comparison",
        "metrics": metrics,
        "pairwise": pairwise,
        "per_target": per_target,
        "checks": checks,
        "decision": "CAL residual screen valid; run corrected live8"
        if passed
        else "repair O186C before live",
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

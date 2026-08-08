#!/usr/bin/env python3
"""Average two linear recovery heads while preserving the frozen O127C baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPRecoveryResidual
from path_config import (
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O185C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O189C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O197C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json
from run_a6_o185c_recovery_residual_train import sha256_file


RUN_ID = "a6_o197c_recovery_residual_weight_average_v1"


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_O197C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    seed_paths = (
        Path(JOINTTRAIN_ARCH6_O185C_RESULT_ROOT) / "last.pt",
        Path(JOINTTRAIN_ARCH6_O189C_RESULT_ROOT) / "last.pt",
    )
    seed_states = [
        torch.load(path, map_location="cpu", weights_only=False)["model"]
        for path in seed_paths
    ]
    baseline_path = Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt"
    baseline_state = torch.load(
        baseline_path, map_location="cpu", weights_only=False
    )["model"]
    if set(seed_states[0]) != set(seed_states[1]):
        raise ValueError("residual checkpoint state keys differ")

    head_keys = {key for key in seed_states[0] if key.startswith("recovery_head.")}
    baseline_keys = set(seed_states[0]) - head_keys
    merged_state = {key: value.clone() for key, value in seed_states[0].items()}
    for key in head_keys:
        merged_state[key] = 0.5 * (seed_states[0][key] + seed_states[1][key])
    head_average_exact = all(
        torch.equal(
            merged_state[key], 0.5 * (seed_states[0][key] + seed_states[1][key])
        )
        for key in head_keys
    )

    baseline_seed_exact = all(
        torch.equal(seed_states[0][key], seed_states[1][key]) for key in baseline_keys
    )
    baseline_o127_exact = all(
        torch.equal(seed_states[0][f"baseline.{key}"], value)
        for key, value in baseline_state.items()
    )
    # CPU avoids TF32 matmul-order differences in this algebraic parity check.
    device = torch.device("cpu")
    models = []
    for state in (*seed_states, merged_state):
        model = OperationMLPRecoveryResidual().to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)

    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        arrays = {name: np.asarray(source[name][:64]) for name in source.files}
    inputs = (
        torch.from_numpy(arrays["point_cloud"].astype(np.float32)).to(device),
        torch.from_numpy(arrays["target_mask"].astype(bool)).to(device),
        torch.from_numpy(arrays["zero_affordance"].astype(np.float32)).to(device),
        torch.from_numpy(arrays["state_history"].astype(np.float32)).to(device),
        torch.from_numpy(arrays["context"].astype(np.float32)).to(device),
    )
    with torch.no_grad():
        seed_outputs = [model(*inputs) for model in models[:2]]
        merged_output = models[2](*inputs)
        expected = 0.5 * (seed_outputs[0] + seed_outputs[1])
        output_mean_error = float(torch.max(torch.abs(merged_output - expected)))

    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": merged_state,
            "input_schema": "O127C-frozen-plus-averaged-linear-recovery-residual",
            "source_checkpoints": [str(path) for path in seed_paths],
            "average_weights": [0.5, 0.5],
            "baseline_frozen": True,
        },
        checkpoint,
    )
    reloaded = OperationMLPRecoveryResidual().to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    reloaded.eval()
    with torch.no_grad():
        reload_error = float(torch.max(torch.abs(reloaded(*inputs) - merged_output)))

    metrics = {
        "output_mean_parity_max_abs": output_mean_error,
        "reload_max_abs": reload_error,
        "source_head_l2_distance": float(
            torch.linalg.vector_norm(
                torch.cat(
                    [
                        (seed_states[0][key] - seed_states[1][key]).reshape(-1)
                        for key in sorted(head_keys)
                    ]
                )
            )
        ),
    }
    checks = {
        "two_source_checkpoints_present": all(path.is_file() for path in seed_paths),
        "head_keys_exact": head_keys
        == {"recovery_head.weight", "recovery_head.bias"},
        "head_tensor_average_exact": head_average_exact,
        "all_non_head_tensors_equal_between_seeds": baseline_seed_exact,
        "frozen_baseline_equals_o127c": baseline_o127_exact,
        "output_equals_seed_mean": output_mean_error <= 1e-6,
        "reload_exact": reload_error == 0.0,
        "fixed_batch_zero_contact": not bool(np.count_nonzero(arrays["context"][:, :34])),
        "finite": bool(np.isfinite(list(metrics.values())).all()),
    }
    passed = all(checks.values())
    atomic_json(
        out / "run_manifest.json",
        {
            "run_id": RUN_ID,
            "source_checkpoint_sha256": [sha256_file(path) for path in seed_paths],
            "baseline_checkpoint_sha256": sha256_file(baseline_path),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "single-model 50/50 recovery-head weight average",
        "metrics": metrics,
        "checks": checks,
        "decision": "weight average valid; run CAL/live"
        if passed
        else "repair weight average before evaluation",
        "next_run_ids": ["A6-O198C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O197C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

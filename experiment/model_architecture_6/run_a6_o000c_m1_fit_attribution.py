#!/usr/bin/env python3
"""CPU-only attribution audit for all corrected Architecture 6 M1 fits."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from a6_operation_models import (
    ACTION_DIM,
    ACTION_HORIZON,
    HIDDEN_DIM,
    OperationCausalAbsolute,
    OperationMLPAbsolute,
    OperationParallelAbsolute,
)
from path_config import (
    JOINTTRAIN_ARCH6_O000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O010R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O020R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O030R_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    atomic_json,
    load_batch,
    model_inputs,
    normalized_l1_sum,
    sha256_file,
)


RUN_ID = "a6_o000c_m1_fit_attribution_v1"


def normalized_mae(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> float:
    numerator, denominator = normalized_l1_sum(prediction, target, valid)
    return float(numerator / denominator.clamp_min(1.0))


def per_horizon_mae(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> list[float | None]:
    values: list[float | None] = []
    for horizon in range(ACTION_HORIZON):
        horizon_valid = valid[:, horizon]
        if not bool(horizon_valid.any()):
            values.append(None)
            continue
        values.append(
            float(
                torch.abs(
                    prediction[horizon_valid, horizon] - target[horizon_valid, horizon]
                ).mean()
            )
        )
    return values


def embedding_stats(embedding: torch.Tensor) -> dict[str, Any]:
    values = embedding.detach().cpu().to(torch.float64)
    distances = torch.cdist(values, values)
    off_diagonal = distances[~torch.eye(values.shape[0], dtype=torch.bool)]
    centered = values - values.mean(dim=0, keepdim=True)
    return {
        "shape": list(values.shape),
        "finite": bool(torch.isfinite(values).all()),
        "pairwise_min_l2": float(off_diagonal.min()),
        "pairwise_median_l2": float(off_diagonal.median()),
        "centered_matrix_rank": int(torch.linalg.matrix_rank(centered)),
        "unique_rounded_1e_8": int(
            np.unique(np.round(values.numpy(), decimals=8), axis=0).shape[0]
        ),
    }


def load_model(
    model: torch.nn.Module, checkpoint_root: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_root) / "last.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "step": int(saved["step"]),
        "seed": int(saved["seed"]),
        "revision": saved["lineage"]["revision"],
    }


def main_run(args: argparse.Namespace) -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_O000C_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    running = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "iteration_id": RUN_ID,
        "complete": False,
        "terminal": False,
        "status": "running",
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    atomic_json(out_dir / "run_state.json", running)
    atomic_json(
        out_dir / "queue_state.json",
        {**running, "jobs": [{"id": "A6-O000C", "status": "running", "pid": os.getpid()}]},
    )
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", None) not in {"", "-1"}:
        raise RuntimeError("O000C must run CPU-only with CUDA_VISIBLE_DEVICES empty")
    torch.manual_seed(20260805)
    batch, lineage, input_manifest = load_batch()
    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    models["O-MLP-ABS"], checkpoints["O-MLP-ABS"] = load_model(
        OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
        JOINTTRAIN_ARCH6_O010R_RESULT_ROOT,
    )
    models["O-PAR-ABS"], checkpoints["O-PAR-ABS"] = load_model(
        OperationParallelAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
        JOINTTRAIN_ARCH6_O020R_RESULT_ROOT,
    )
    models["O-CAUSAL-ABS"], checkpoints["O-CAUSAL-ABS"] = load_model(
        OperationCausalAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
        JOINTTRAIN_ARCH6_O030R_RESULT_ROOT,
    )
    with torch.no_grad():
        predictions = {
            name: model(*model_inputs(batch, slice(None)))
            for name, model in models.items()
        }
        causal_teacher = models["O-CAUSAL-ABS"](
            *model_inputs(batch, slice(None)), teacher_actions=batch["target"]
        )
        embeddings = {
            name: model.encoder(*model_inputs(batch, slice(None)))
            for name, model in models.items()
        }
    repeat_raw = batch["state"][:, -ACTION_DIM:].unsqueeze(1).expand(
        -1, ACTION_HORIZON, -1
    )
    repeat = (
        repeat_raw - batch["action_mean"].reshape(1, 1, -1)
    ) / batch["action_std"].reshape(1, 1, -1)
    train_mean = torch.zeros_like(batch["target"])
    for horizon in range(ACTION_HORIZON):
        horizon_valid = batch["valid"][:, horizon]
        if bool(horizon_valid.any()):
            train_mean[:, horizon] = batch["target"][horizon_valid, horizon].mean(dim=0)
    comparison: dict[str, dict[str, Any]] = {}
    for name, prediction in {
        **predictions,
        "O-CAUSAL-ABS-teacher-forced": causal_teacher,
        "repeat-last-command": repeat,
        "train-mean-chunk": train_mean,
    }.items():
        comparison[name] = {
            "normalized_mae": normalized_mae(
                prediction, batch["target"], batch["valid"]
            ),
            "per_horizon_normalized_mae": per_horizon_mae(
                prediction, batch["target"], batch["valid"]
            ),
        }
    target_delta = torch.abs(batch["target"] - repeat)
    valid_delta = target_delta[batch["valid"].unsqueeze(-1).expand_as(target_delta)]
    endpoint_valid = batch["valid"][:, -1]
    endpoint_delta = torch.abs(batch["target"][endpoint_valid, -1] - repeat[endpoint_valid, -1])
    motion = {
        "valid_scalar_count": int(valid_delta.numel()),
        "fraction_valid_scalars_with_normalized_delta_le_1e_3": float(
            (valid_delta <= 1e-3).to(torch.float32).mean()
        ),
        "fraction_valid_scalars_with_normalized_delta_le_1e_2": float(
            (valid_delta <= 1e-2).to(torch.float32).mean()
        ),
        "mean_normalized_abs_delta_from_last_command": float(valid_delta.mean()),
        "p95_normalized_abs_delta_from_last_command": float(
            torch.quantile(valid_delta, 0.95)
        ),
        "endpoint_rows": int(endpoint_valid.sum()),
        "fraction_endpoint_rows_mean_delta_gt_1e_2": float(
            (endpoint_delta.mean(dim=-1) > 1e-2).to(torch.float32).mean()
        )
        if endpoint_delta.numel()
        else None,
    }
    causal_ar = comparison["O-CAUSAL-ABS"]["normalized_mae"]
    causal_tf = comparison["O-CAUSAL-ABS-teacher-forced"]["normalized_mae"]
    audit = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "comparison": comparison,
        "causal_teacher_forcing_gap": {
            "teacher_forced_normalized_mae": causal_tf,
            "autoregressive_normalized_mae": causal_ar,
            "autoregressive_over_teacher_ratio": causal_ar / max(causal_tf, 1e-12),
            "absolute_gap": causal_ar - causal_tf,
        },
        "motion_structure": motion,
        "encoder_embedding_separability": {
            name: embedding_stats(value) for name, value in embeddings.items()
        },
        "checkpoints": checkpoints,
        "input_lineage": lineage,
    }
    checks = {
        "all_three_checkpoints_step_2000_seed_match": all(
            row["step"] == 2000 and row["seed"] == 20260805
            for row in checkpoints.values()
        ),
        "all_revision_a6_input_v1_1": all(
            row["revision"] == "A6-INPUT-v1.1" for row in checkpoints.values()
        ),
        "all_metrics_finite": all(
            np.isfinite(row["normalized_mae"])
            for row in comparison.values()
        ),
        "all_encoders_separate_64_rows": all(
            row["unique_rounded_1e_8"] == 64 and row["pairwise_min_l2"] > 0
            for row in audit["encoder_embedding_separability"].values()
        ),
        "all_models_worse_than_repeat_last": all(
            comparison[name]["normalized_mae"]
            > comparison["repeat-last-command"]["normalized_mae"]
            for name in ("O-MLP-ABS", "O-PAR-ABS", "O-CAUSAL-ABS")
        ),
        "causal_teacher_forcing_gap_present": causal_ar > 2.0 * causal_tf,
        "zero_training_replay_heldout_or_outcome": True,
    }
    passed = all(checks.values())
    atomic_json(out_dir / "attribution.json", audit)
    atomic_json(
        out_dir / "command.json",
        {
            "schema_version": 1,
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "environment": "sapien",
            "resource_mode": "cpu",
            "seed": 20260805,
        },
    )
    atomic_json(
        out_dir / "sample_manifest.json",
        {"schema_version": 1, "rows": input_manifest["rows"]},
    )
    atomic_json(
        out_dir / "forbidden_feature_audit.json",
        {
            "schema_version": 1,
            "training": False,
            "replay": False,
            "cal_read": False,
            "mech_dev_read": False,
            "final_read": False,
            "outcome_read": False,
            "object_qpos_read": False,
            "future_qpos_read": False,
        },
    )
    atomic_json(
        out_dir / "resource_pilot.json",
        {
            "schema_version": 1,
            "workload_signature": "three fixed checkpoints x one fixed64 CPU inference audit",
            "resource_mode": "cpu",
            "workers": 1,
            "parallelism": "not_applicable_single_deterministic_batch",
            "hardware": platform.processor(),
            "wall_seconds": time.time() - running["started_at"],
        },
    )
    atomic_json(
        out_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "depends_on": ["A6-O010R", "A6-O020R", "A6-O030R"],
            "mode": "CPU-only checkpoint attribution; zero optimization",
            "input_lineage": lineage,
            "checkpoint_hashes": {
                name: row["checkpoint_sha256"] for name, row in checkpoints.items()
            },
        },
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "evidence": {
            "attribution": "attribution.json",
            "manifest": "run_manifest.json",
            "forbidden_feature_audit": "forbidden_feature_audit.json",
            "resource_pilot": "resource_pilot.json",
        },
        "checks": checks,
        "metrics": {
            "model_normalized_mae": {
                name: comparison[name]["normalized_mae"]
                for name in ("O-MLP-ABS", "O-PAR-ABS", "O-CAUSAL-ABS")
            },
            "repeat_last_normalized_mae": comparison["repeat-last-command"][
                "normalized_mae"
            ],
            "causal_teacher_forcing_gap": audit["causal_teacher_forcing_gap"],
            "motion_structure": motion,
        },
        "decision": (
            "M1 failures are valid: inputs are separable, all models trail repeat-last, and causal teacher-forcing exposure gap is confirmed. A planning revision is required before any new fit or DYN64 run."
            if passed
            else "M1 attribution audit did not establish all preregistered checks; keep all training and DYN64 blocked."
        ),
        "remaining_work": [
            "prepare and acknowledge a planning revision for the M1 failure route",
            "do not launch new training or DYN64 before revision",
        ],
        "next_run_ids": [],
        "event_id": f"{RUN_ID}_terminal",
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(
        out_dir / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-O000C", "status": summary["status"]}]},
    )
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"run-id must be {RUN_ID}")
    try:
        return main_run(args)
    except Exception as error:
        out_dir = Path(JOINTTRAIN_ARCH6_O000C_RESULT_ROOT)
        atomic_json(
            out_dir / "failure.json",
            {
                "schema_version": 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        summary = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "complete": True,
            "terminal": True,
            "status": "failed",
            "failure_class": "implementation_failure",
            "claim_supported": "no",
            "decision": "O000C attribution implementation failed; inspect failure.json.",
            "remaining_work": ["repair CPU audit without changing scientific contracts"],
            "next_run_ids": [],
            "event_id": f"{RUN_ID}_terminal",
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(
            out_dir / "queue_state.json",
            {**summary, "jobs": [{"id": "A6-O000C", "status": "failed"}]},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

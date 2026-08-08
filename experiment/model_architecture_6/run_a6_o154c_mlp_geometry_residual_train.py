#!/usr/bin/env python3
"""Train the corrected A6 geometry residual without changing the 81D state LN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute, OperationMLPGeometryResidual
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D150C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O154C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    normalized_l1_sum,
)


RUN_ID = "a6_o154c_mlp_geometry_residual_train1024_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def forward(model, batch):
    return model(
        batch["point_cloud"],
        batch["target_mask"],
        batch["affordance"],
        batch["state"],
        batch["context"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if (args.steps, args.batch_size) != (6000, 64):
        raise ValueError("A6-O154C requires the matched 6000-step/batch64 config")

    out = Path(JOINTTRAIN_ARCH6_O154C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    d150 = Path(JOINTTRAIN_ARCH6_D150C_RESULT_ROOT)
    with np.load(d150 / "train_fk_target_relative.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    normalizer_path = Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json"
    action_std = torch.tensor(
        json.loads(normalizer_path.read_text(encoding="utf-8"))["std"],
        dtype=torch.float32,
    ).reshape(1, 1, 9)
    data = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"].astype(np.float32)),
        "target_mask": torch.from_numpy(arrays["target_mask"].astype(bool)),
        "affordance": torch.from_numpy(arrays["zero_affordance"].astype(np.float32)),
        "state": torch.from_numpy(arrays["state_history"].astype(np.float32)),
        "context": torch.from_numpy(arrays["context"].astype(np.float32)),
        "target": torch.from_numpy(arrays["command_delta_target"].astype(np.float32)) / action_std,
        "valid": torch.from_numpy(arrays["action_valid"].astype(bool)),
    }

    torch.manual_seed(SEED)
    baseline_initial = OperationMLPAbsolute(dropout=DROPOUT)
    torch.manual_seed(SEED)
    model = OperationMLPGeometryResidual(dropout=DROPOUT)
    baseline_state = baseline_initial.state_dict()
    model_state = model.state_dict()
    common_initial_exact = all(
        torch.equal(model_state[name], value) for name, value in baseline_state.items()
    )
    baseline_initial.eval()
    model.eval()
    with torch.no_grad():
        baseline_initial_output = baseline_initial(
            data["point_cloud"][:2],
            data["target_mask"][:2],
            data["affordance"][:2],
            data["state"][:2, :81],
            data["context"][:2],
        )
        geometry_initial_output = forward(
            model, {name: value[:2] for name, value in data.items()}
        )
    initial_output_max_abs = float(
        torch.max(torch.abs(baseline_initial_output - geometry_initial_output))
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "common_initial_exact": common_initial_exact,
                    "initial_output_max_abs": initial_output_max_abs,
                }
            )
        )
        return 0 if common_initial_exact and initial_output_max_abs == 0.0 else 2
    if not torch.cuda.is_available():
        raise RuntimeError("A6-O154C requires CUDA")

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(
        0, len(data["state"]), (args.steps, args.batch_size), generator=generator
    )
    model = OperationMLPGeometryResidual(dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict] = []
    started = time.perf_counter()
    for step in range(args.steps):
        index = indices[step]
        batch = {name: value[index].to(device) for name, value in data.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(model, batch)
        numerator, denominator = normalized_l1_sum(
            prediction, batch["target"], batch["valid"]
        )
        loss = numerator / denominator.clamp_min(1.0)
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"nonfinite gradient at step {step + 1}")
        optimizer.step()
        if step == 0 or (step + 1) % 500 == 0:
            history.append(
                {"step": step + 1, "train_batch_normalized_delta_mae": float(loss)}
            )

    model.eval()
    normalized_sum = 0.0
    raw_sum = 0.0
    count = 0.0
    reference = None
    action_std_device = action_std.to(device)
    with torch.no_grad():
        for begin in range(0, len(data["state"]), args.batch_size):
            batch = {
                name: value[begin : begin + args.batch_size].to(device)
                for name, value in data.items()
            }
            prediction = forward(model, batch)
            if reference is None:
                reference = prediction.cpu()
            numerator, denominator = normalized_l1_sum(
                prediction, batch["target"], batch["valid"]
            )
            valid = batch["valid"].unsqueeze(-1).expand_as(prediction)
            normalized_sum += float(numerator)
            raw_sum += float(
                torch.abs((prediction - batch["target"]) * action_std_device)[valid].sum()
            )
            count += float(denominator)
    normalized_mae = normalized_sum / count
    raw_mae = raw_sum / count
    geometry_weight_norm = float(torch.linalg.vector_norm(model.geometry_projection.weight))
    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder": "mlp_geometry_residual",
            "seed": SEED,
            "state_dim": 85,
            "input_schema": "D150C-FK-target-relative-residual",
            "training_rows": len(data["state"]),
        },
        checkpoint,
    )
    reloaded = OperationMLPGeometryResidual(dropout=DROPOUT).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True
    )
    reloaded.eval()
    with torch.no_grad():
        reload_prediction = forward(
            reloaded,
            {
                name: value[: args.batch_size].to(device) for name, value in data.items()
            },
        ).cpu()
    reload_max_abs = float(torch.max(torch.abs(reference - reload_prediction)))
    wall_seconds = time.perf_counter() - started
    checks = {
        "d150c_passed": json.loads((d150 / "summary.json").read_text(encoding="utf-8"))["status"] == "passed",
        "train_rows_1024": len(data["state"]) == 1024,
        "state_width_85": data["state"].shape == (1024, 85),
        "baseline_state_encoder_width_81": tuple(model.encoder.state_encoder[1].weight.shape) == (128, 81),
        "common_initial_weights_exact": common_initial_exact,
        "zero_init_output_parity": initial_output_max_abs == 0.0,
        "geometry_branch_learned": geometry_weight_norm > 0.0,
        "zero_contact": not bool(torch.count_nonzero(data["context"][:, :34])),
        "finite": bool(np.isfinite([normalized_mae, raw_mae, reload_max_abs, geometry_weight_norm]).all()),
        "reload_exact": reload_max_abs == 0.0,
    }
    passed = all(checks.values())
    metrics = {
        "train_normalized_delta_mae": normalized_mae,
        "train_raw_delta_mae": raw_mae,
        "initial_output_max_abs_vs_baseline": initial_output_max_abs,
        "geometry_projection_weight_norm": geometry_weight_norm,
        "reload_max_abs": reload_max_abs,
        "wall_seconds": wall_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(out / "history.json", {"history": history})
    atomic_json(out / "offline_metrics.json", metrics)
    atomic_json(
        out / "training_config.json",
        {
            "seed": SEED,
            "steps": args.steps,
            "effective_batch": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "base_state_dim": 81,
            "geometry_dim": 4,
            "geometry_projection_init": "zero",
        },
    )
    atomic_json(
        out / "command.json",
        {"argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    )
    atomic_json(
        out / "run_manifest.json",
        {
            "run_id": RUN_ID,
            "input_sha256": sha256_file(d150 / "train_fk_target_relative.npz"),
            "feature_normalizer_sha256": sha256_file(d150 / "feature_normalizer.json"),
            "action_normalizer_sha256": sha256_file(normalizer_path),
            "checkpoint_sha256": sha256_file(checkpoint),
            "baseline_checkpoint_sha256": sha256_file(Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt"),
        },
    )
    atomic_json(
        out / "forbidden_feature_audit.json",
        json.loads((d150 / "forbidden_feature_audit.json").read_text(encoding="utf-8")),
    )
    atomic_json(
        out / "resource_pilot_ref.json",
        {
            "reference_run": "A6-O127C",
            "reason": "same 1024-row MLP workload with one 4x256 residual projection",
            "effective_batch": args.batch_size,
            "wall_seconds": wall_seconds,
        },
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "corrected geometry residual with unchanged 81D baseline state encoder",
        "metrics": metrics,
        "checks": checks,
        "decision": "corrected residual training valid; run CAL/live8" if passed else "repair residual isolation",
        "next_run_ids": ["A6-O155C", "A6-O156C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-O154C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

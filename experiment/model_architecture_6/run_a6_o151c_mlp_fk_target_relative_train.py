#!/usr/bin/env python3
"""Train the matched A6 MLP with deployable FK/target-relative state."""

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

from a6_deployable_geometry import FK_TARGET_STATE_DIM
from a6_operation_models import OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D150C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O151C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    normalized_l1_sum,
)


RUN_ID = "a6_o151c_mlp_fk_target_relative_train1024_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.steps != 6000 or args.batch_size != 64:
        raise ValueError("A6-O151C matched comparison requires steps=6000 and batch-size=64")

    out = Path(JOINTTRAIN_ARCH6_O151C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    d150 = Path(JOINTTRAIN_ARCH6_D150C_RESULT_ROOT)
    d150_summary = json.loads((d150 / "summary.json").read_text(encoding="utf-8"))
    if d150_summary.get("status") != "passed":
        raise ValueError("A6-D150C did not pass")
    with np.load(d150 / "train_fk_target_relative.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    action_normalizer = json.loads(
        (Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text(
            encoding="utf-8"
        )
    )
    action_std = torch.tensor(action_normalizer["std"], dtype=torch.float32).reshape(1, 1, 9)
    data = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"].astype(np.float32)),
        "target_mask": torch.from_numpy(arrays["target_mask"].astype(bool)),
        "affordance": torch.from_numpy(arrays["zero_affordance"].astype(np.float32)),
        "state": torch.from_numpy(arrays["state_history"].astype(np.float32)),
        "context": torch.from_numpy(arrays["context"].astype(np.float32)),
        "target": torch.from_numpy(arrays["command_delta_target"].astype(np.float32)) / action_std,
        "valid": torch.from_numpy(arrays["action_valid"].astype(bool)),
    }
    model = OperationMLPAbsolute(state_dim=FK_TARGET_STATE_DIM, dropout=DROPOUT)
    if args.validate_only:
        with torch.no_grad():
            prediction = model(
                data["point_cloud"][:2],
                data["target_mask"][:2],
                data["affordance"][:2],
                data["state"][:2],
                data["context"][:2],
            )
        print(json.dumps({"status": "validated", "shape": list(prediction.shape)}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("A6-O151C requires CUDA")

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(0, len(data["state"]), (args.steps, args.batch_size), generator=generator)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    action_std_device = action_std.to(device)
    history: list[dict] = []
    started = time.perf_counter()
    peak_memory = 0
    for step in range(args.steps):
        index = indices[step]
        batch = {name: value[index].to(device) for name, value in data.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch["point_cloud"],
            batch["target_mask"],
            batch["affordance"],
            batch["state"],
            batch["context"],
        )
        numerator, denominator = normalized_l1_sum(
            prediction, batch["target"], batch["valid"]
        )
        loss = numerator / denominator.clamp_min(1.0)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise RuntimeError(f"nonfinite gradient at step {step + 1}")
        optimizer.step()
        peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        if step == 0 or (step + 1) % 500 == 0:
            history.append(
                {
                    "step": step + 1,
                    "train_batch_normalized_delta_mae": float(loss.detach()),
                }
            )

    model.eval()
    normalized_numerator = 0.0
    raw_numerator = 0.0
    denominator_total = 0.0
    reference_prediction = None
    with torch.no_grad():
        for begin in range(0, len(data["state"]), args.batch_size):
            batch = {
                name: value[begin : begin + args.batch_size].to(device)
                for name, value in data.items()
            }
            prediction = model(
                batch["point_cloud"],
                batch["target_mask"],
                batch["affordance"],
                batch["state"],
                batch["context"],
            )
            if reference_prediction is None:
                reference_prediction = prediction.detach().cpu()
            numerator, denominator = normalized_l1_sum(
                prediction, batch["target"], batch["valid"]
            )
            mask = batch["valid"].unsqueeze(-1).expand_as(prediction)
            normalized_numerator += float(numerator)
            raw_numerator += float(
                torch.abs((prediction - batch["target"]) * action_std_device)[mask].sum()
            )
            denominator_total += float(denominator)
    normalized_mae = normalized_numerator / denominator_total
    raw_mae = raw_numerator / denominator_total

    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder": "mlp",
            "seed": SEED,
            "state_dim": FK_TARGET_STATE_DIM,
            "input_schema": "D150C-FK-target-relative",
            "training_rows": len(data["state"]),
        },
        checkpoint,
    )
    reloaded = OperationMLPAbsolute(
        state_dim=FK_TARGET_STATE_DIM, dropout=DROPOUT
    ).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True
    )
    reloaded.eval()
    with torch.no_grad():
        reload_prediction = reloaded(
            data["point_cloud"][: args.batch_size].to(device),
            data["target_mask"][: args.batch_size].to(device),
            data["affordance"][: args.batch_size].to(device),
            data["state"][: args.batch_size].to(device),
            data["context"][: args.batch_size].to(device),
        ).cpu()
    reload_max_abs = float(torch.max(torch.abs(reference_prediction - reload_prediction)))
    wall_seconds = time.perf_counter() - started
    checks = {
        "train_rows_1024": len(data["state"]) == 1024,
        "state_dim_85": data["state"].shape == (1024, FK_TARGET_STATE_DIM),
        "zero_contact": not bool(torch.count_nonzero(data["context"][:, :34])),
        "steps_6000": history[-1]["step"] == 6000,
        "batch_64": args.batch_size == 64,
        "finite": bool(np.isfinite([normalized_mae, raw_mae, reload_max_abs]).all()),
        "reload_exact": reload_max_abs == 0.0,
        "matched_seed": SEED == 20260805,
    }
    passed = all(checks.values())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    baseline_checkpoint = Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt"
    atomic_json(out / "history.json", {"history": history})
    atomic_json(
        out / "training_config.json",
        {
            "seed": SEED,
            "steps": args.steps,
            "effective_batch": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "state_dim": FK_TARGET_STATE_DIM,
            "parameter_count": parameter_count,
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
            "action_normalizer_sha256": sha256_file(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
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
            "reason": "same in-memory 1024-row MLP workload; only state width changes from 81 to 85",
            "batch_size": args.batch_size,
            "peak_allocated_bytes": peak_memory,
            "wall_seconds": wall_seconds,
        },
    )
    metrics = {
        "train_normalized_delta_mae": normalized_mae,
        "train_raw_delta_mae": raw_mae,
        "reload_max_abs": reload_max_abs,
        "wall_seconds": wall_seconds,
        "parameter_count": parameter_count,
        "peak_allocated_bytes": peak_memory,
    }
    atomic_json(out / "offline_metrics.json", metrics)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "matched TRAIN1024 MLP with deployable FK/visible-target relative state",
        "metrics": metrics,
        "checks": checks,
        "decision": "training valid; run exact-paired CAL and live8" if passed else "repair implementation before evaluation",
        "next_run_ids": ["A6-O152C", "A6-O153C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-O151C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

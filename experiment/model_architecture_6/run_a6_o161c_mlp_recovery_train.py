#!/usr/bin/env python3
"""Train O127C MLP on the additive 1088-row TRAIN recovery dataset."""

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

from a6_operation_models import OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D160C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O161C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    normalized_l1_sum,
)


RUN_ID = "a6_o161c_mlp_recovery_train1088_v1"
STEPS = 6000
BATCH = 64
ACTION_DIM = 9
HORIZON = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predict(model, batch):
    return model(
        batch["point_cloud"],
        batch["target_mask"],
        batch["affordance"],
        batch["state"],
        batch["context"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    args = parser.parse_args()
    if (args.steps, args.batch_size) != (STEPS, BATCH):
        raise ValueError("A6-O161C requires steps=6000 and batch-size=64")
    out = Path(JOINTTRAIN_ARCH6_O161C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    d160 = Path(JOINTTRAIN_ARCH6_D160C_RESULT_ROOT)
    d160_summary = json.loads((d160 / "summary.json").read_text(encoding="utf-8"))
    if d160_summary.get("status") != "passed":
        raise ValueError("A6-D160C did not pass")
    with np.load(d160 / "train_recovery1088.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        original_arrays = {name: np.asarray(source[name]) for name in source.files}
    normalizer_path = Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json"
    action_std = torch.tensor(
        json.loads(normalizer_path.read_text(encoding="utf-8"))["std"],
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
    if data["state"].shape[0] != 1088:
        raise ValueError("recovery input row count is not 1088")
    if not torch.cuda.is_available():
        raise RuntimeError("A6-O161C requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(0, 1088, (STEPS, BATCH), generator=generator)
    model = OperationMLPAbsolute(dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict] = []
    started = time.perf_counter()
    for step in range(STEPS):
        index = indices[step]
        batch = {name: value[index].to(device) for name, value in data.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(model, batch)
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
                {"step": step + 1, "train_batch_normalized_delta_mae": float(loss.detach())}
            )

    model.eval()
    action_std_device = action_std.to(device)

    def evaluate_range(begin: int, end: int) -> tuple[float, float]:
        normalized_sum = 0.0
        raw_sum = 0.0
        count = 0.0
        with torch.no_grad():
            for start in range(begin, end, BATCH):
                batch = {
                    name: value[start : min(end, start + BATCH)].to(device)
                    for name, value in data.items()
                }
                prediction = predict(model, batch)
                numerator, denominator = normalized_l1_sum(
                    prediction, batch["target"], batch["valid"]
                )
                valid = batch["valid"].unsqueeze(-1).expand_as(prediction)
                normalized_sum += float(numerator)
                raw_sum += float(
                    torch.abs((prediction - batch["target"]) * action_std_device)[valid].sum()
                )
                count += float(denominator)
        return normalized_sum / count, raw_sum / count

    normalized_mae, raw_mae = evaluate_range(0, 1088)
    prefix_normalized_mae, prefix_raw_mae = evaluate_range(0, 1024)
    recovery_normalized_mae, recovery_raw_mae = evaluate_range(1024, 1088)
    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder": "mlp",
            "seed": SEED,
            "state_dim": 81,
            "input_schema": "D042C-zero-contact-plus-D160C-recovery",
            "training_rows": 1088,
        },
        checkpoint,
    )
    reloaded = OperationMLPAbsolute(dropout=DROPOUT).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True
    )
    reloaded.eval()
    with torch.no_grad():
        first = {name: value[:BATCH].to(device) for name, value in data.items()}
        reload_error = float(torch.max(torch.abs(predict(model, first) - predict(reloaded, first))))
    wall_seconds = time.perf_counter() - started
    checks = {
        "d160c_passed": d160_summary.get("status") == "passed",
        "rows_1088": data["state"].shape == (1088, 81),
        "prefix_contract": all(
            name in arrays
            and name in original_arrays
            and np.array_equal(arrays[name][:1024], original_arrays[name])
            for name in (
                "point_cloud",
                "target_mask",
                "zero_affordance",
                "state_history",
                "context",
                "absolute_action_target",
                "command_delta_target",
                "action_valid",
            )
        ),
        "zero_contact": not bool(torch.count_nonzero(data["context"][:, :34])),
        "steps_6000": history[-1]["step"] == STEPS,
        "batch_64": args.batch_size == BATCH,
        "finite": bool(np.isfinite([normalized_mae, raw_mae, prefix_raw_mae, recovery_raw_mae, reload_error]).all()),
        "reload_exact": reload_error == 0.0,
        "recovery_rows_present": arrays["state_history"].shape[0] == 1088
        and arrays["state_history"][1024:].shape[0] == 64,
    }
    passed = all(checks.values())
    metrics = {
        "train1088_normalized_delta_mae": normalized_mae,
        "train1088_raw_delta_mae": raw_mae,
        "prefix1024_normalized_delta_mae": prefix_normalized_mae,
        "prefix1024_raw_delta_mae": prefix_raw_mae,
        "recovery64_normalized_delta_mae": recovery_normalized_mae,
        "recovery64_raw_delta_mae": recovery_raw_mae,
        "reload_max_abs": reload_error,
        "wall_seconds": wall_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(out / "history.json", {"history": history})
    atomic_json(out / "offline_metrics.json", metrics)
    atomic_json(
        out / "training_config.json",
        {
            "seed": SEED,
            "steps": STEPS,
            "effective_batch": BATCH,
            "sampling": "uniform over additive 1088 rows",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
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
            "input_sha256": sha256_file(d160 / "train_recovery1088.npz"),
            "recovery_summary_sha256": sha256_file(d160 / "summary.json"),
            "action_normalizer_sha256": sha256_file(normalizer_path),
            "checkpoint_sha256": sha256_file(checkpoint),
            "baseline_checkpoint_sha256": sha256_file(Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt"),
        },
    )
    atomic_json(
        out / "forbidden_feature_audit.json",
        json.loads((d160 / "forbidden_feature_audit.json").read_text(encoding="utf-8")),
    )
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": "additive TRAIN-only simulator recovery supervision with O127C MLP",
        "metrics": metrics,
        "checks": checks,
        "decision": "recovery training valid; run unchanged CAL/live8" if passed else "repair recovery training",
        "next_run_ids": ["A6-O162C", "A6-O163C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-O161C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

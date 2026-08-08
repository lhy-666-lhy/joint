#!/usr/bin/env python3
"""Matched training for time- and progress-aligned D180C recovery labels."""

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
    JOINTTRAIN_ARCH6_D180C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O181C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O182C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import (
    DROPOUT,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    atomic_json,
    normalized_l1_sum,
)


ROOTS = {
    "time": Path(JOINTTRAIN_ARCH6_O181C_RESULT_ROOT),
    "progress": Path(JOINTTRAIN_ARCH6_O182C_RESULT_ROOT),
}
RUN_IDS = {
    "time": "a6_o181c_time_aligned_recovery_train_v1",
    "progress": "a6_o182c_progress_aligned_recovery_train_v1",
}
STEPS = 6000
BATCH = 64
ROWS = 1152
PREFIX_ROWS = 1024
ACTION_DIM = 9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predict(model: OperationMLPAbsolute, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["point_cloud"],
        batch["target_mask"],
        batch["affordance"],
        batch["state"],
        batch["context"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", choices=tuple(ROOTS), required=True)
    args = parser.parse_args()
    alignment = args.alignment
    out = ROOTS[alignment]
    out.mkdir(parents=True, exist_ok=True)
    d180 = Path(JOINTTRAIN_ARCH6_D180C_RESULT_ROOT)
    d180_summary = json.loads((d180 / "summary.json").read_text(encoding="utf-8"))
    if d180_summary.get("status") != "passed":
        raise ValueError("A6-D180C did not pass")
    input_path = d180 / f"train_recovery_{alignment}1152.npz"
    with np.load(input_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    with np.load(
        Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz",
        allow_pickle=False,
    ) as source:
        prefix = {name: np.asarray(source[name]) for name in source.files}
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
        "target": torch.from_numpy(
            arrays["command_delta_target"].astype(np.float32)
        )
        / action_std,
        "valid": torch.from_numpy(arrays["action_valid"].astype(bool)),
    }
    if data["state"].shape != (ROWS, 81):
        raise ValueError("unexpected D180C training shape")
    if not torch.cuda.is_available():
        raise RuntimeError("O181C/O182C requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(0, ROWS, (STEPS, BATCH), generator=generator)
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
                {
                    "step": step + 1,
                    "train_batch_normalized_delta_mae": float(loss.detach()),
                }
            )

    model.eval()
    action_std_device = action_std.to(device)

    def evaluate(begin: int, end: int) -> tuple[float, float]:
        normalized_sum = 0.0
        raw_sum = 0.0
        count = 0.0
        with torch.no_grad():
            for offset in range(begin, end, BATCH):
                batch = {
                    name: value[offset : min(end, offset + BATCH)].to(device)
                    for name, value in data.items()
                }
                prediction = predict(model, batch)
                numerator, denominator = normalized_l1_sum(
                    prediction, batch["target"], batch["valid"]
                )
                valid = batch["valid"].unsqueeze(-1).expand_as(prediction)
                normalized_sum += float(numerator)
                raw_sum += float(
                    torch.abs(
                        (prediction - batch["target"]) * action_std_device
                    )[valid].sum()
                )
                count += float(denominator)
        return normalized_sum / count, raw_sum / count

    total_normalized, total_raw = evaluate(0, ROWS)
    prefix_normalized, prefix_raw = evaluate(0, PREFIX_ROWS)
    recovery_normalized, recovery_raw = evaluate(PREFIX_ROWS, ROWS)
    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder": "mlp",
            "seed": SEED,
            "state_dim": 81,
            "input_schema": f"D042C-prefix-plus-D180C-{alignment}-aligned-recovery",
            "training_rows": ROWS,
            "recovery_alignment": alignment,
        },
        checkpoint,
    )
    reloaded = OperationMLPAbsolute(dropout=DROPOUT).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    reloaded.eval()
    with torch.no_grad():
        first = {name: value[:BATCH].to(device) for name, value in data.items()}
        reload_error = float(
            torch.max(torch.abs(predict(model, first) - predict(reloaded, first)))
        )
    metrics = {
        "total_normalized_delta_mae": total_normalized,
        "total_raw_delta_mae": total_raw,
        "prefix1024_normalized_delta_mae": prefix_normalized,
        "prefix1024_raw_delta_mae": prefix_raw,
        "recovery128_normalized_delta_mae": recovery_normalized,
        "recovery128_raw_delta_mae": recovery_raw,
        "reload_max_abs": reload_error,
        "wall_seconds": time.perf_counter() - started,
    }
    checks = {
        "d180c_passed": d180_summary.get("status") == "passed",
        "rows_1152": data["state"].shape == (ROWS, 81),
        "prefix_exact": all(
            name in arrays
            and name in prefix
            and np.array_equal(arrays[name][:PREFIX_ROWS], prefix[name])
            for name in prefix
        ),
        "zero_contact": not bool(torch.count_nonzero(data["context"][:, :34])),
        "steps_6000": history[-1]["step"] == STEPS,
        "batch_64": indices.shape == (STEPS, BATCH),
        "finite": bool(np.isfinite(list(metrics.values())).all()),
        "reload_exact": reload_error == 0.0,
    }
    passed = all(checks.values())
    run_id = RUN_IDS[alignment]
    atomic_json(out / "history.json", {"history": history})
    atomic_json(out / "offline_metrics.json", metrics)
    atomic_json(
        out / "training_config.json",
        {
            "alignment": alignment,
            "seed": SEED,
            "steps": STEPS,
            "effective_batch": BATCH,
            "sampling": "uniform over additive 1152 rows",
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
            "run_id": run_id,
            "input_sha256": sha256_file(input_path),
            "d180c_summary_sha256": sha256_file(d180 / "summary.json"),
            "normalizer_sha256": sha256_file(normalizer_path),
            "checkpoint_sha256": sha256_file(checkpoint),
            "baseline_checkpoint_sha256": sha256_file(
                Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt"
            ),
        },
    )
    atomic_json(
        out / "forbidden_feature_audit.json",
        json.loads(
            (d180 / "forbidden_feature_audit.json").read_text(encoding="utf-8")
        ),
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "implementation_failure",
        "scientific_scope": f"matched {alignment}-aligned full-horizon recovery training",
        "metrics": metrics,
        "checks": checks,
        "decision": "alignment training valid; run joint CAL screen"
        if passed
        else "repair matched recovery training",
        "next_run_ids": ["A6-O183C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {
            **summary,
            "jobs": [
                {
                    "id": "A6-O181C" if alignment == "time" else "A6-O182C",
                    "status": summary["status"],
                }
            ],
        },
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


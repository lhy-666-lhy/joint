#!/usr/bin/env python3
"""Train matched MLP/PAR operation policies on the D043C TRAIN194 input."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import OperationMLPAbsolute, OperationParallelAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D043C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O132C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O133C_RESULT_ROOT,
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
    "mlp": JOINTTRAIN_ARCH6_O132C_RESULT_ROOT,
    "parallel": JOINTTRAIN_ARCH6_O133C_RESULT_ROOT,
}
FACTORIES = {
    "mlp": OperationMLPAbsolute,
    "parallel": OperationParallelAbsolute,
}
RUN_IDS = {
    "mlp": "a6_o132c_mlp_zero_contact_train194_v1",
    "parallel": "a6_o133c_parallel_zero_contact_train194_v1",
}
STEPS = 6000
BATCH = 64
EXPECTED_ROWS = 1552
DATA_ROOT = JOINTTRAIN_ARCH6_D043C_RESULT_ROOT
INPUT_FILENAME = "train194_input.npz"
INPUT_SCHEMA = "D043C-zero-contact"
TRAINING_TARGETS = 194
SCIENTIFIC_SCOPE = "A5_TRAIN 194-target/1552-anchor zero-contact matched training"
ROW_CHECK_NAME = "train_rows_1552"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", choices=tuple(ROOTS), required=True)
    args = parser.parse_args()

    out = Path(ROOTS[args.decoder])
    out.mkdir(parents=True, exist_ok=True)
    input_path = Path(DATA_ROOT) / "full" / INPUT_FILENAME
    with np.load(input_path, allow_pickle=False) as data_file:
        arrays = {key: np.asarray(data_file[key]) for key in data_file.files}

    std = torch.tensor(
        json.load(open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json"))["std"],
        dtype=torch.float32,
    ).reshape(1, 1, 9)
    data = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"]),
        "target_mask": torch.from_numpy(arrays["target_mask"]),
        "affordance": torch.from_numpy(arrays["zero_affordance"]),
        "state": torch.from_numpy(arrays["state_history"]),
        "context": torch.from_numpy(arrays["context"]),
        "target": torch.from_numpy(arrays["command_delta_target"]) / std,
        "valid": torch.from_numpy(arrays["action_valid"]),
    }
    row_count = len(data["state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRAIN194 matched training requires CUDA")

    std = std.to(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(0, row_count, (STEPS, BATCH), generator=generator)
    model = FACTORIES[args.decoder](dropout=DROPOUT).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []
    start = time.perf_counter()

    for step in range(STEPS):
        index = indices[step]
        batch = {key: value[index].to(device) for key, value in data.items()}
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
        loss = numerator / denominator.clamp_min(1)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 500 == 0:
            history.append(
                {
                    "step": step + 1,
                    "train_batch_normalized_delta_mae": float(loss.detach()),
                }
            )

    model.eval()
    normalized_total = 0.0
    valid_total = 0.0
    raw_total = 0.0
    with torch.no_grad():
        for start_index in range(0, row_count, BATCH):
            batch = {
                key: value[start_index : start_index + BATCH].to(device)
                for key, value in data.items()
            }
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
            normalized_total += float(numerator)
            valid_total += float(denominator)
            mask = batch["valid"].unsqueeze(-1).expand_as(prediction)
            raw_total += float(
                torch.abs((prediction - batch["target"]) * std)[mask].sum()
            )

    normalized_mae = normalized_total / valid_total
    raw_mae = raw_total / valid_total
    checkpoint = out / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder": args.decoder,
            "seed": SEED,
            "input_schema": INPUT_SCHEMA,
            "training_rows": row_count,
            "training_targets": TRAINING_TARGETS,
        },
        checkpoint,
    )
    checks = {
        ROW_CHECK_NAME: row_count == EXPECTED_ROWS,
        "zero_contact": not bool(torch.count_nonzero(data["context"][:, :34])),
        "steps_6000": history[-1]["step"] == STEPS,
        "finite": bool(np.isfinite([normalized_mae, raw_mae]).all()),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": RUN_IDS[args.decoder],
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "scientific_scope": SCIENTIFIC_SCOPE,
        "configuration": {
            "seed": SEED,
            "steps": STEPS,
            "batch_size": BATCH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "parameter_count": parameter_count,
            "normalizer": "A6-D021C",
        },
        "metrics": {
            "train_normalized_delta_mae": normalized_mae,
            "train_raw_delta_mae": raw_mae,
            "wall_seconds": time.perf_counter() - start,
        },
        "checks": checks,
        "decision": (
            "TRAIN194 fit valid; evaluate frozen A5_CAL"
            if passed
            else "TRAIN194 training invalid"
        ),
    }
    atomic_json(out / "history.json", {"history": history})
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate the frozen command-delta MLP on the clean DYN64 input set."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_models import ACTION_DIM, ACTION_HORIZON, HIDDEN_DIM, OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D040C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O100C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O200F_RESULT_ROOT,
)


RUN_ID = "a6_o100c_mlp_command_delta_dyn64_v1"
DROPOUT = 0.1


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT) / "full"
    with np.load(root / "dyn64_input.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    normalizer = json.loads((Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").read_text())
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32).reshape(1, 1, ACTION_DIM)
    std = torch.tensor(normalizer["std"], dtype=torch.float32).reshape(1, 1, ACTION_DIM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {
        "point_cloud": torch.from_numpy(arrays["point_cloud"]),
        "target_mask": torch.from_numpy(arrays["target_mask"]),
        "affordance": torch.from_numpy(arrays["zero_affordance"]),
        "state": torch.from_numpy(arrays["state_history"]),
        "context": torch.from_numpy(arrays["context"]),
        "target_delta": torch.from_numpy(arrays["command_delta_target"]),
        "valid": torch.from_numpy(arrays["action_valid"]),
    }
    model = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    checkpoint_path = Path(JOINTTRAIN_ARCH6_O200F_RESULT_ROOT) / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        predicted_delta_norm = model(
            batch["point_cloud"], batch["target_mask"], batch["affordance"], batch["state"], batch["context"]
        )
        predicted_delta = predicted_delta_norm * std.to(device)
        target_delta = batch["target_delta"]
        target_delta_norm = target_delta / std.to(device)
        last_command = batch["state"][:, -ACTION_DIM:].unsqueeze(1)
        predicted_absolute = last_command + predicted_delta
        target_absolute = target_delta + last_command
        mask = batch["valid"].unsqueeze(-1).expand_as(predicted_delta)
        normalized_mae = float(torch.abs(predicted_delta_norm - target_delta_norm)[mask].mean())
        raw_mae = float(torch.abs(predicted_absolute - target_absolute)[mask].mean())
        repeat_mae = float(torch.abs(last_command - target_absolute)[mask].mean())
        horizon_mae = torch.abs(predicted_absolute - target_absolute).mean(dim=-1)
        horizon_values = [float(horizon_mae[:, index][batch["valid"][:, index]].mean()) for index in range(ACTION_HORIZON)]
        endpoint = ACTION_HORIZON - 1
        endpoint_mask = batch["valid"][:, endpoint]
        endpoint_mae = float(torch.abs(predicted_absolute[:, endpoint] - target_absolute[:, endpoint])[endpoint_mask].mean()) if bool(endpoint_mask.any()) else None
        finite = bool(torch.isfinite(predicted_delta).all())
    reloaded = OperationMLPAbsolute(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        reload_error = float(torch.max(torch.abs(reloaded(batch["point_cloud"], batch["target_mask"], batch["affordance"], batch["state"], batch["context"]) - predicted_delta_norm)))
    checks = {
        "input_rows_1024": arrays["point_cloud"].shape == (1024, 1024, 3),
        "prediction_shape_exact": tuple(predicted_delta.shape) == (1024, ACTION_HORIZON, ACTION_DIM),
        "finite_prediction": finite,
        "checkpoint_reload_max_error_le_1e_6": reload_error <= 1e-6,
        "zero_affordance": not bool(np.count_nonzero(arrays["zero_affordance"])),
        "valid_mask_present": bool(arrays["action_valid"].any()),
    }
    passed = all(checks.values())
    out_dir = Path(JOINTTRAIN_ARCH6_O100C_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dyn64_rows": int(arrays["point_cloud"].shape[0]),
        "normalized_delta_mae": normalized_mae,
        "raw_absolute_mae": raw_mae,
        "repeat_last_absolute_mae": repeat_mae,
        "relative_to_repeat_last": raw_mae / max(repeat_mae, 1e-12),
        "endpoint_absolute_mae": endpoint_mae,
        "horizon_absolute_mae": horizon_values,
        "checkpoint_reload_max_error": reload_error,
    }
    atomic_json(out_dir / "run_manifest.json", {"schema_version": 1, "run_id": RUN_ID, "model": "O-MLP-CMDDELTA", "input": str((root / "dyn64_input.npz").resolve()), "input_sha256": sha256_file(root / "dyn64_input.npz"), "checkpoint": str(checkpoint_path.resolve()), "normalizer": str((Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json").resolve())})
    atomic_json(out_dir / "offline_metrics.json", {"schema_version": 1, **metrics})
    summary = {"schema_version": 1, "run_id": RUN_ID, "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "implementation_failure", "claim_supported": "yes" if passed else "no", "metrics": metrics, "checks": checks, "decision": "O100C DYN64 offline screen is valid; compare matched decoders." if passed else "O100C input/checkpoint evaluation invalid; do not compare decoders.", "evidence": {"manifest": "run_manifest.json", "metrics": "offline_metrics.json"}, "event_id": f"{RUN_ID}_terminal"}
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-O100C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

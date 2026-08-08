#!/usr/bin/env python3
"""A6-D010 absolute-qpos adapter roundtrip on two frozen replay trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model.datasets import action_from_npz, select_trajectory_phase
from model.online_eval import load_result_json, replay_action
from model.train_trajectory import normalize_action, unnormalize_action
from path_config import (
    ARTICU_CURATED_ACCEPTED_SAMPLES,
    JOINTTRAIN_ARCH2_C020_SHARED_REPLAY_SUMMARY,
    JOINTTRAIN_ARCH6_D010_RESULT_ROOT,
)


HORIZON = 32
EXECUTE_PREFIX = 8


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def chunk_roundtrip(action: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> tuple[np.ndarray, float]:
    chunks: list[np.ndarray] = []
    for start in range(0, action.shape[0], EXECUTE_PREFIX):
        stop = min(action.shape[0], start + HORIZON)
        chunk = action[start:stop]
        valid = min(EXECUTE_PREFIX, action.shape[0] - start)
        if chunk.shape[0] < HORIZON:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], HORIZON - chunk.shape[0], axis=0)], axis=0)
        tensor = torch.from_numpy(chunk.astype(np.float32))
        decoded = unnormalize_action(normalize_action(tensor, mean, std), mean, std).cpu().numpy()
        chunks.append(decoded[:valid])
    stitched = np.concatenate(chunks, axis=0).astype(np.float32)
    error = float(np.max(np.abs(stitched - action)))
    return stitched, error


def sample_sizes() -> dict[tuple[str, str, str], float]:
    rows = [json.loads(line) for line in Path(ARTICU_CURATED_ACCEPTED_SAMPLES).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(str(row["shape_id"]), str(row["link_name"]), str(row["repeat"])): float(row["size"]) for row in rows}


def trajectory_identity(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    single = parts.index("single")
    return parts[single + 1], parts[single + 2], parts[single + 3]


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_D010_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path(JOINTTRAIN_ARCH2_C020_SHARED_REPLAY_SUMMARY).read_text(encoding="utf-8"))[:2]
    sizes = sample_sizes()
    source_actions: list[np.ndarray] = []
    records: list[dict] = []
    for row in frozen:
        trajectory = Path(row["trajectory_npz"])
        with np.load(trajectory, allow_pickle=True) as data:
            full_action = action_from_npz(data, source="joint_command_qpos_repaired", include_finger=False)
            operation = select_trajectory_phase(data, full_action, "operation")
        if operation.ndim != 2 or operation.shape[1] != 9 or not np.isfinite(operation).all():
            raise ValueError(f"invalid canonical operation action: {trajectory} {operation.shape}")
        source_actions.append(operation)
        records.append({"trajectory": str(trajectory), "historical_replay_success": bool(row["eval_success"]), "operation_length": int(operation.shape[0]), "source_sha256": sha256_array(operation)})
    stacked = np.concatenate(source_actions, axis=0).astype(np.float32)
    mean = torch.from_numpy(stacked.mean(axis=0, keepdims=True))
    std = torch.from_numpy(stacked.std(axis=0, keepdims=True)).clamp_min(1e-4)
    all_pass = True
    for record, operation in zip(records, source_actions):
        decoded, max_error = chunk_roundtrip(operation, mean, std)
        trajectory = Path(record["trajectory"])
        result = load_result_json(trajectory)
        identity = trajectory_identity(trajectory)
        kwargs = {"trajectory_npz": str(trajectory), "link_name": str(result["link_name"]), "size": sizes[identity], "steps_per_waypoint": 1, "success_open_ratio": 0.4, "replay_start_phase": "operation", "operation_controller_mode": "never", "replay_drive_mode": "drive", "action_already_operation": True, "contact_static_friction": 2.0, "contact_dynamic_friction": 2.0, "contact_restitution": 0.0, "finger_stiffness": 4000.0, "finger_damping": 800.0}
        source_replay = replay_action(operation, **kwargs)
        decoded_replay = replay_action(decoded, **kwargs)
        source_success = bool(source_replay.get("passed"))
        decoded_success = bool(decoded_replay.get("passed"))
        parity = decoded_success == source_success
        record.update({"decoded_sha256": sha256_array(decoded), "max_abs_roundtrip_error": max_error, "source_operation_replay_success": source_success, "decoded_replay_success": decoded_success, "replay_parity": parity, "source_final_target_open_ratio": source_replay.get("final_target_open_ratio"), "decoded_final_target_open_ratio": decoded_replay.get("final_target_open_ratio")})
        all_pass = all_pass and max_error <= 1e-6 and parity and source_success and decoded_success
    (out_dir / "adapter_contract.json").write_text(json.dumps({"schema_version": 1, "action_source": "joint_command_qpos_repaired", "action_dimension": 9, "representation": "absolute", "horizon": HORIZON, "execute_prefix": EXECUTE_PREFIX, "normalizer_mean": mean.reshape(-1).tolist(), "normalizer_std": std.reshape(-1).tolist(), "records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"schema_version": 1, "run_id": "a6_d010_adapter_roundtrip_v1", "complete": True, "terminal": True, "status": "passed" if all_pass else "failed", "failure_class": None if all_pass else "data_contract_failure", "claim_supported": "yes" if all_pass else "no", "evidence": {"adapter_contract": "adapter_contract.json", "trajectory_count": len(records), "new_full_replay": False, "operation_only_replay_count": 2 * len(records)}, "checks": {"all_9d_finite": True, "roundtrip_max_abs_le_1e_6": all(record["max_abs_roundtrip_error"] <= 1e-6 for record in records), "replay_parity_2_of_2": sum(record["replay_parity"] for record in records) == 2, "source_success_2_of_2": sum(record["source_operation_replay_success"] for record in records) == 2, "decoded_success_2_of_2": sum(record["decoded_replay_success"] for record in records) == 2}, "decision": "D010 adapter roundtrip passes; authorize D020 and D030." if all_pass else "D010 adapter or operation replay parity failed; do not materialize.", "remaining_work": ["A6-D020 sample/normalizer freeze", "A6-D030 DYN64 materialization"] if all_pass else ["repair D010 without changing H/K or executor"], "next_run_ids": ["a6_d020_sample_normalizer_v1", "a6_d030_dyn64_materialization_v1"] if all_pass else [], "event_id": "a6_d010_adapter_roundtrip_v1_terminal"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

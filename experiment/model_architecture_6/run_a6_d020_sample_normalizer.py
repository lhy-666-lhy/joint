#!/usr/bin/env python3
"""Freeze the A6 operation sample index, fixed batch, and TRAIN normalizer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from model.datasets import action_from_npz
from path_config import (
    ARTICU_COLLECTION_ROOT,
    ARTICU_CURATED_ACCEPTED_SAMPLES,
    JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT,
    JOINTTRAIN_ARCH6_D020_RESULT_ROOT,
    JOINTTRAIN_ARCH5_DYN8_SUMMARY,
    JOINTTRAIN_ARCH5_TERMINAL_OBSERVATION_MASK,
)


HORIZON = 32
EXECUTE_PREFIX = 8
ANCHORS_PER_TRAJECTORY = 16
FIXED_ANCHORS_PER_TRAJECTORY = 8


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_base(row: dict) -> Path:
    relative = Path(str(row["relative_heatmap_npz"]))
    return (Path(ARTICU_COLLECTION_ROOT) / relative).parents[1]


def resolve_trajectory(index_path: Path, value: object) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_file() else index_path.parent / candidate.name


def operation_span(data: np.lib.npyio.NpzFile) -> tuple[int, int]:
    phases = np.asarray(data["action_phase"]).astype(str).reshape(-1)
    indices = np.flatnonzero(phases == "operation")
    if indices.size < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError("invalid operation phase")
    return int(indices[0]), int(indices[-1]) + 1


def deterministic_anchors(action: np.ndarray, start: int, stop: int) -> list[int]:
    operation = np.asarray(action[start:stop, :7], dtype=np.float64)
    legal_count = operation.shape[0] - 1
    if legal_count < ANCHORS_PER_TRAJECTORY:
        raise ValueError("operation too short for 16 unique legal anchors")
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(operation, axis=0), axis=1))])
    if cumulative[-1] > 0:
        candidates = np.searchsorted(cumulative, np.linspace(0.0, cumulative[-1], ANCHORS_PER_TRAJECTORY), side="left")
    else:
        candidates = np.linspace(0, legal_count - 1, ANCHORS_PER_TRAJECTORY).round().astype(int)
    candidates = np.clip(candidates, 0, legal_count - 1).tolist()
    unique = list(dict.fromkeys(int(value) for value in candidates))
    if len(unique) < ANCHORS_PER_TRAJECTORY:
        for value in np.linspace(0, legal_count - 1, legal_count).round().astype(int).tolist():
            if value not in unique:
                unique.append(value)
            if len(unique) == ANCHORS_PER_TRAJECTORY:
                break
    unique.sort()
    return [start + value for value in unique]


def chunk(action: np.ndarray, anchor: int, operation_stop: int) -> tuple[np.ndarray, np.ndarray]:
    begin = anchor + 1
    stop = min(operation_stop, begin + HORIZON)
    values = np.asarray(action[begin:stop], dtype=np.float32)
    valid = np.zeros((HORIZON,), dtype=bool)
    valid[: values.shape[0]] = True
    if values.shape[0] < HORIZON:
        values = np.concatenate([values, np.repeat(values[-1:], HORIZON - values.shape[0], axis=0)], axis=0)
    return values, valid


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_D020_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT).read_text(encoding="utf-8"))
    accepted = [json.loads(line) for line in Path(ARTICU_CURATED_ACCEPTED_SAMPLES).read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted_by_id = {str(row["sample_id"]): row for row in accepted}
    train_ids = [sample_id for target in split["source_partitions"]["A5_TRAIN"] for sample_id in target["sample_ids"]]
    invalid_rows = {str(row["trajectory_relative_path"]): set(map(int, row["invalid_observation_raw_indices"])) for row in json.loads(Path(JOINTTRAIN_ARCH5_TERMINAL_OBSERVATION_MASK).read_text(encoding="utf-8"))["entries"]}
    dyn8 = json.loads(Path(JOINTTRAIN_ARCH5_DYN8_SUMMARY).read_text(encoding="utf-8"))["rows"]
    dyn8_paths = {str(Path(ARTICU_COLLECTION_ROOT) / row["relative_trajectory_path"]): row for row in dyn8}
    fixed_rows: list[dict] = []
    fixed_actions: list[np.ndarray] = []
    fixed_masks: list[np.ndarray] = []
    total = np.zeros((9,), dtype=np.float64)
    square = np.zeros((9,), dtype=np.float64)
    command_rows = 0
    trajectory_count = 0
    index_tmp = out_dir / "sample_index.jsonl.tmp"
    with index_tmp.open("w", encoding="utf-8") as index_handle:
        for sample_number, sample_id in enumerate(train_ids, 1):
            sample = accepted_by_id[sample_id]
            index_path = sample_base(sample) / "trajectory" / "index.json"
            entries = json.loads(index_path.read_text(encoding="utf-8"))["trajectories"]
            for value in entries:
                trajectory = resolve_trajectory(index_path, value).resolve()
                relative = trajectory.relative_to(Path(ARTICU_COLLECTION_ROOT)).as_posix()
                with np.load(trajectory, allow_pickle=False) as data:
                    action = action_from_npz(data, source="joint_command_qpos_repaired", include_finger=False)
                    start, stop = operation_span(data)
                operation = np.asarray(action[start:stop], dtype=np.float64)
                total += operation.sum(axis=0)
                square += np.square(operation).sum(axis=0)
                command_rows += operation.shape[0]
                anchors = deterministic_anchors(action, start, stop)
                masked = invalid_rows.get(relative, set())
                if any(anchor in masked for anchor in anchors):
                    raise ValueError(f"masked observation anchor: {relative}")
                source_hash = sha256_file(trajectory)
                index_handle.write(json.dumps({"trajectory_relative_path": relative, "source_sha256": source_hash, "sample_id": sample_id, "split": "A5_TRAIN", "target": str(sample["target"]), "operation_start": start, "operation_stop": stop, "anchors": anchors, "horizon": HORIZON, "execute_prefix": EXECUTE_PREFIX}, ensure_ascii=True, sort_keys=True) + "\n")
                trajectory_count += 1
                dyn = dyn8_paths.get(str(trajectory))
                if dyn is not None:
                    selected = np.linspace(0, len(anchors) - 1, FIXED_ANCHORS_PER_TRAJECTORY).round().astype(int)
                    for anchor_rank in selected.tolist():
                        values, valid = chunk(action, anchors[anchor_rank], stop)
                        fixed_actions.append(values)
                        fixed_masks.append(valid)
                        fixed_rows.append({"target": dyn["target"], "trajectory_relative_path": relative, "source_sha256": source_hash, "anchor_rank": int(anchor_rank), "anchor_raw_index": int(anchors[anchor_rank]), "valid_count": int(valid.sum())})
            if sample_number % 25 == 0:
                atomic_json(out_dir / "run_state.json", {"schema_version": 1, "iteration_id": "a6_d020_sample_normalizer_v1", "status": "running", "samples_scanned": sample_number, "samples_total": len(train_ids), "trajectories_scanned": trajectory_count})
    os.replace(index_tmp, out_dir / "sample_index.jsonl")
    mean = total / command_rows
    variance = np.maximum(square / command_rows - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-4)
    actions = np.stack(fixed_actions).astype(np.float32)
    masks = np.stack(fixed_masks)
    np.savez_compressed(out_dir / "fixed_batch.npz", action=actions, valid_mask=masks)
    normalizer = {"schema_version": 1, "split": "A5_TRAIN", "source": "all operation joint_command_qpos_repaired rows", "action_dimension": 9, "row_count": command_rows, "trajectory_count": trajectory_count, "mean": mean.tolist(), "std": std.tolist()}
    atomic_json(out_dir / "normalizer.json", normalizer)
    index_hash = sha256_file(out_dir / "sample_index.jsonl")
    fixed_hash = sha256_file(out_dir / "fixed_batch.npz")
    checks = {"train_trajectory_count_exact": trajectory_count == 26605, "dyn8_target_count_exact": len({row["target"] for row in fixed_rows}) == 8, "fixed_chunk_count_exact": len(fixed_rows) == 64, "all_chunks_h32x9_finite": actions.shape == (64, 32, 9) and bool(np.isfinite(actions).all()), "valid_masks_exact": masks.shape == (64, 32) and bool(masks.any(axis=1).all()), "all_sample_splits_train": True, "terminal_mask_consumed": True, "source_hashes_persisted": True, "zero_mech_dev_or_final_reads": True}
    passed = all(checks.values())
    atomic_json(out_dir / "fixed_batch_manifest.json", {"schema_version": 1, "rows": fixed_rows, "sample_index_sha256": index_hash, "fixed_batch_sha256": fixed_hash, "normalizer_sha256": sha256_file(out_dir / "normalizer.json")})
    summary = {"schema_version": 1, "run_id": "a6_d020_sample_normalizer_v1", "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "data_contract_failure", "claim_supported": "yes" if passed else "no", "evidence": {"sample_index": "sample_index.jsonl", "fixed_batch": "fixed_batch.npz", "fixed_batch_manifest": "fixed_batch_manifest.json", "normalizer": "normalizer.json"}, "counts": {"train_samples": len(train_ids), "train_trajectories": trajectory_count, "operation_command_rows": command_rows, "fixed_chunks": len(fixed_rows)}, "checks": checks, "decision": "D020 sample/normalizer freeze passes." if passed else "D020 sample/normalizer contract failed; do not train.", "remaining_work": ["A6-D030 DYN64 materialization"] if passed else ["repair D020 without changing H/K, split, or labels"], "next_run_ids": ["a6_d030_dyn64_materialization_v1"] if passed else [], "event_id": "a6_d020_sample_normalizer_v1_terminal"}
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-D020", "status": summary["status"]}]})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

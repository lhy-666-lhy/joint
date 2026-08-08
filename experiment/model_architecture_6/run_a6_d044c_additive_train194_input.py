#!/usr/bin/env python3
"""Preserve D042C exactly and append only D043C targets absent from D042C."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT,
    JOINTTRAIN_ARCH6_D040C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D043C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D044C_RESULT_ROOT,
)

ARRAY_NAMES = (
    "point_cloud",
    "target_mask",
    "zero_affordance",
    "state_history",
    "context",
    "absolute_action_target",
    "command_delta_target",
    "action_valid",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-new-targets", type=int, default=0)
    args = parser.parse_args()

    d042_path = Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz"
    d043_path = Path(JOINTTRAIN_ARCH6_D043C_RESULT_ROOT) / "full" / "train194_input.npz"
    base_arrays = load_npz(d042_path)
    candidate_arrays = load_npz(d043_path)
    if tuple(base_arrays) != ARRAY_NAMES or tuple(candidate_arrays) != ARRAY_NAMES:
        raise ValueError("D042C/D043C array schema mismatch")

    base_rows = json.load(
        open(Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT) / "full" / "input_manifest.json")
    )["rows"]
    candidate_rows = json.load(
        open(Path(JOINTTRAIN_ARCH6_D043C_RESULT_ROOT) / "full" / "input_manifest.json")
    )["rows"]
    base_targets = {row["target"] for row in base_rows}
    available_new_targets = sorted(
        {row["target"] for row in candidate_rows} - base_targets
    )
    selected_new_targets = available_new_targets
    if args.limit_new_targets:
        selected_new_targets = available_new_targets[: args.limit_new_targets]
    selected_set = set(selected_new_targets)
    selected_indices = np.asarray(
        [index for index, row in enumerate(candidate_rows) if row["target"] in selected_set],
        dtype=np.int64,
    )

    combined = {
        name: np.concatenate([base_arrays[name], candidate_arrays[name][selected_indices]], axis=0)
        for name in ARRAY_NAMES
    }
    rows = [
        {
            **row,
            "split": "A5_TRAIN",
            "observation_source": "D042C_preserved_prefix",
            "lineage": "D042C",
        }
        for row in base_rows
    ] + [
        {**candidate_rows[index], "lineage": "D043C_new_target_append"}
        for index in selected_indices.tolist()
    ]

    target_counts = Counter(row["target"] for row in rows)
    expected_new_targets = args.limit_new_targets or 130
    expected_targets = 64 + expected_new_targets
    expected_rows = 1024 + 8 * expected_new_targets
    cal_targets = {
        entry["target"]
        for entry in json.load(open(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT))[
            "source_partitions"
        ]["A5_CAL"]
    }
    checks = {
        "base_rows_1024": len(base_rows) == 1024,
        "base_targets_64": len(base_targets) == 64,
        "new_targets_exact": len(selected_set) == expected_new_targets,
        "new_targets_disjoint_from_base": not bool(selected_set & base_targets),
        "targets_exact": len(target_counts) == expected_targets,
        "rows_exact": len(rows) == expected_rows,
        "base_prefix_exact": all(
            np.array_equal(combined[name][:1024], base_arrays[name])
            for name in ARRAY_NAMES
        ),
        "base_rows_16_each": all(target_counts[target] == 16 for target in base_targets),
        "new_rows_8_each": all(target_counts[target] == 8 for target in selected_set),
        "zero_cal_target_overlap": not bool(set(target_counts) & cal_targets),
        "zero_contact": not bool(np.count_nonzero(combined["context"][:, :34])),
        "task_metadata_finite": bool(np.isfinite(combined["context"][:, 34:]).all()),
        "point_state_label_finite": all(
            np.isfinite(combined[name]).all()
            for name in ("point_cloud", "state_history", "command_delta_target")
        ),
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_D044C_RESULT_ROOT) / (
        f"probe_{args.limit_new_targets}" if args.limit_new_targets else "full"
    )
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "train194_additive_input.npz"
    temporary = out / "train194_additive_input.tmp.npz"
    np.savez_compressed(temporary, **combined)
    os.replace(temporary, input_path)
    manifest = {
        "schema_version": 1,
        "run_id": "a6_d044c_additive_train194_zero_contact_input_v1",
        "intervention": (
            "exact D042C prefix plus D043C rows whose targets are absent from D042C"
        ),
        "rows": rows,
        "array_shapes": {name: list(value.shape) for name, value in combined.items()},
        "source_hashes": {
            "d042c_train": sha256_file(d042_path),
            "d043c_train": sha256_file(d043_path),
        },
        "input_sha256": sha256_file(input_path),
    }
    summary = {
        "schema_version": 1,
        "run_id": "a6_d044c_additive_train194_zero_contact_input_v1",
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "counts": {
            "base_targets": len(base_targets),
            "new_targets": len(selected_set),
            "targets": len(target_counts),
            "rows": len(rows),
        },
        "checks": checks,
        "decision": (
            "additive target-coverage input valid; train matched MLP/PAR"
            if passed
            else "D044C additive input invalid"
        ),
    }
    atomic_json(out / "input_manifest.json", manifest)
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

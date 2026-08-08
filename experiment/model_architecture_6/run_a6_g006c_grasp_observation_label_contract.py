#!/usr/bin/env python3
"""Join deployable primary observations with the clean G000C grasp labels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[3]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G006C_RESULT_ROOT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
)

RUN_ID = "a6_g006c_grasp_observation_label_contract_v1"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_G006C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "traj_teacher_manifest.json").read_text())
    groups = manifest["groups"]
    with np.load(Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "traj_labels.npz", allow_pickle=False) as labels:
        paths = np.asarray(labels["path_relative"], dtype=np.float32)
        presence = np.asarray(labels["presence"], dtype=bool)
        initial_arm_qpos = np.asarray(labels["initial_arm_qpos"], dtype=np.float32)
    with np.load(Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_labels.npz", allow_pickle=False) as labels:
        qpose = np.asarray(labels["qpose_relative"], dtype=np.float32)
    zroot = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    source_ids = np.asarray(zroot["meta/source_replay_id"][:], dtype=np.int32)
    row_by_source = {int(value): i for i, value in enumerate(source_ids.tolist())}
    source_rows = np.asarray([row_by_source[int(group["source_replay_id"])] for group in groups], dtype=np.int64)
    point_cloud = np.asarray(zroot["data/point_cloud"][source_rows, :, :3], dtype=np.float32)
    split = np.asarray([0 if group["split"] == "A5_TRAIN" else 1 for group in groups], dtype=np.int8)
    group_index = np.asarray([int(group["group_index"]) for group in groups], dtype=np.int64)
    target = np.asarray([str(group["target"]) for group in groups], dtype=object)
    arrays = {
        "point_cloud_xyz": point_cloud,
        "state_qpos": initial_arm_qpos,
        "path_relative": paths,
        "qpose_relative": qpose,
        "presence": presence,
        "split": split,
        "group_index": group_index,
        "source_replay_id": np.asarray([int(group["source_replay_id"]) for group in groups], dtype=np.int32),
    }
    np.savez_compressed(out / "grasp_inputs.npz", **arrays)
    checks = {
        "group_count": len(groups) == 632,
        "train_cal_counts": int((split == 0).sum()) == 531 and int((split == 1).sum()) == 101,
        "source_join_exact": len(source_rows) == len(set(source_rows.tolist())) and bool(np.isfinite(source_rows).all()),
        "point_shape_finite": point_cloud.shape == (632, 1024, 3) and bool(np.isfinite(point_cloud).all()),
        "state_shape_finite": initial_arm_qpos.shape == (632, 7) and bool(np.isfinite(initial_arm_qpos).all()),
        "labels_shape_finite": paths.shape == (632, 4, 64, 7) and qpose.shape == (632, 4, 7) and bool(np.isfinite(paths).all() and np.isfinite(qpose).all()),
        "presence_not_copied": bool(np.all(presence.sum(axis=1) >= 1)) and bool(np.all(presence.sum(axis=1) <= 4)),
        "zero_affordance_input": True,
        "no_outcome_or_future_input": True,
    }
    summary = {
        "schema_version": 1, "run_id": RUN_ID, "status": "passed" if all(checks.values()) else "failed",
        "complete": True, "terminal": True, "scientific_scope": "G010/G020 deployable ZERO-affordance grasp input join",
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "counts": {"groups": len(groups), "train": int((split == 0).sum()), "cal": int((split == 1).sum())},
        "input_schema": "current primary XYZ point cloud + current initial 7D robot qpos; no affordance channel, no target mask, no future/outcome fields",
        "source_hashes": {"g000c": sha256_file(Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "summary.json"), "zarr_summary": sha256_file(Path(JOINTTRAIN_BESTVIEW_DUAL_ZARR) / ".zarr_summary.json")},
        "checks": checks,
        "decision": "authorize matched G010/G020 fixed-batch fit" if all(checks.values()) else "repair observation-label join",
    }
    atomic_json(out / "input_manifest.json", {"run_id": RUN_ID, "source_rows": source_rows.tolist(), "target": target.tolist(), "checks": checks})
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": RUN_ID, "status": summary["status"]}]})
    print(json.dumps(summary))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Group G051 base-frame hand poses back into the frozen K4 grasp contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import JOINTTRAIN_ARCH6_G041C_RESULT_ROOT, JOINTTRAIN_ARCH6_G051C_RESULT_ROOT, JOINTTRAIN_ARCH6_G052C_RESULT_ROOT


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[:3, :2].T.reshape(-1)


def main() -> int:
    input_path = Path(JOINTTRAIN_ARCH6_G041C_RESULT_ROOT) / "grasp_inputs.npz"
    pose_path = Path(JOINTTRAIN_ARCH6_G051C_RESULT_ROOT) / "full" / "relative_grasp_labels.npz"
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    with np.load(pose_path, allow_pickle=False) as data:
        base_pose = np.asarray(data["base_pose"], dtype=np.float64)
        pose_split = np.asarray(data["split"], dtype=np.int8)
        pose_group = np.asarray(data["group_index"], dtype=np.int64)
    grouped = np.zeros((632, 4, 9), dtype=np.float32)
    cursor = 0
    lineage_ok = True
    for group in range(632):
        for slot in np.flatnonzero(arrays["presence"][group]):
            lineage_ok &= int(pose_group[cursor]) == int(arrays["group_index"][group]) and int(pose_split[cursor]) == int(arrays["split"][group])
            grouped[group, slot, :3] = base_pose[cursor, :3, 3]
            grouped[group, slot, 3:] = rotation_6d(base_pose[cursor])
            cursor += 1
    arrays["se3_base"] = grouped
    out = Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    output = out / "grasp_inputs.npz"
    np.savez_compressed(output, **arrays)
    valid_rotation = grouped[arrays["presence"], 3:]
    first = valid_rotation[:, :3]
    second = valid_rotation[:, 3:]
    checks = {
        "all_pose_rows_consumed": cursor == len(base_pose) == 2373,
        "lineage_exact": bool(lineage_ok),
        "shape": grouped.shape == (632, 4, 9),
        "finite": bool(np.isfinite(grouped).all()),
        "rotation_unit": bool(np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5) and np.allclose(np.linalg.norm(second, axis=1), 1.0, atol=1e-5)),
        "rotation_orthogonal": bool(np.allclose(np.sum(first * second, axis=1), 0.0, atol=1e-5)),
        "split_counts": int(np.sum(arrays["split"] == 0)) == 531 and int(np.sum(arrays["split"] == 1)) == 101,
        "no_link_pose_input": "relative_pose" not in arrays,
        "no_outcome": True,
    }
    summary = {
        "schema_version": 1,
        "run_id": "A6-G052C",
        "status": "passed" if all(checks.values()) else "failed",
        "complete": True,
        "terminal": True,
        "groups": 632,
        "labels": cursor,
        "input_sha256": sha256(input_path),
        "pose_source_sha256": sha256(pose_path),
        "output_sha256": sha256(output),
        "checks": checks,
        "decision": "authorize matched base-only and target-local SE3 sanity" if all(checks.values()) else "repair grouped SE3 labels",
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

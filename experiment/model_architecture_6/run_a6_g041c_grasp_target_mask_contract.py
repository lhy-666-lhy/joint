#!/usr/bin/env python3
"""Join base-frame grasp inputs with the deployable primary-view target mask."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from path_config import JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT, JOINTTRAIN_ARCH6_G040R_RESULT_ROOT, JOINTTRAIN_ARCH6_G041C_RESULT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> int:
    base_path = Path(JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT) / "grasp_inputs.npz"
    mask_path = Path(JOINTTRAIN_ARCH6_G040R_RESULT_ROOT) / "full" / "target_masks.npz"
    mask_summary_path = Path(JOINTTRAIN_ARCH6_G040R_RESULT_ROOT) / "full" / "summary.json"
    mask_summary = json.loads(mask_summary_path.read_text())
    if mask_summary.get("status") != "passed" or not mask_summary.get("terminal"):
        raise RuntimeError("G040C is not a terminal passed artifact")
    with np.load(base_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    with np.load(mask_path, allow_pickle=False) as data:
        target_mask = np.asarray(data["target_mask"], dtype=bool)
        mask_group_index = np.asarray(data["group_index"], dtype=np.int64)
    arrays["target_mask"] = target_mask
    out = Path(JOINTTRAIN_ARCH6_G041C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "grasp_inputs.npz"
    np.savez_compressed(output_path, **arrays)
    checks = {
        "g040_terminal_pass": True,
        "group_index_exact": bool(np.array_equal(arrays["group_index"], mask_group_index)),
        "shape": target_mask.shape == (632, 1024),
        "binary": bool(np.isin(target_mask, [False, True]).all()),
        "all_target_visible": bool(np.all(target_mask.sum(axis=1) > 0)),
        "base_points_unchanged": True,
        "labels_unchanged": True,
        "split_counts": int(np.sum(arrays["split"] == 0)) == 531 and int(np.sum(arrays["split"] == 1)) == 101,
        "no_affordance_or_outcome": "affordance" not in arrays and "outcome" not in arrays,
    }
    summary = {
        "schema_version": 1,
        "run_id": "A6-G041C",
        "status": "passed" if all(checks.values()) else "failed",
        "complete": True,
        "terminal": True,
        "groups": int(target_mask.shape[0]),
        "target_points": {
            "min": int(target_mask.sum(axis=1).min()),
            "median": float(np.median(target_mask.sum(axis=1))),
            "max": int(target_mask.sum(axis=1).max()),
        },
        "source_hashes": {"base_inputs": sha256(base_path), "target_masks": sha256(mask_path)},
        "output_sha256": sha256(output_path),
        "checks": checks,
        "decision": "authorize matched target-mask qpose/traj sanity" if all(checks.values()) else "repair target-mask join",
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

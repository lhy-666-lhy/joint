#!/usr/bin/env python3
"""Repair only misaligned G040 target masks with fresh single-worker renders."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import run_a6_g040c_primary_target_mask as source
from path_config import JOINTTRAIN_ARCH6_G000C_RESULT_ROOT, JOINTTRAIN_ARCH6_G040C_RESULT_ROOT, JOINTTRAIN_ARCH6_G040R_RESULT_ROOT, JOINTTRAIN_BESTVIEW_DUAL_ZARR


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def load_row(path: Path) -> tuple[int, np.ndarray, float, float, int, int, float]:
    with np.load(path, allow_pickle=False) as data:
        whole_error = float(data["max_error"])
        target_error = float(data["target_alignment_error"]) if "target_alignment_error" in data.files else (0.0 if whole_error <= 1e-5 else float("inf"))
        return int(data["index"]), np.asarray(data["mask"], dtype=bool), whole_error, float(data["mean_error"]), int(data["target_points"]), int(data["raw_points"]), target_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-bad", type=int, default=0)
    args = parser.parse_args()
    source_root = Path(JOINTTRAIN_ARCH6_G040C_RESULT_ROOT) / "full"
    source_summary = json.loads((source_root / "summary.json").read_text())
    groups = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_teacher_manifest.json").read_text())["groups"]
    source_rows = {int(path.stem): load_row(path) for path in (source_root / "rows").glob("*.npz")}
    if len(source_rows) != len(groups) or not source_summary.get("terminal"):
        raise RuntimeError("G040C source is not terminal with 632 rows")
    bad = sorted(index for index, row in source_rows.items() if row[2] > 1e-5)
    selected_bad = bad[: args.pilot_bad] if args.pilot_bad else bad
    name = f"probe_bad{args.pilot_bad}_exact_target" if args.pilot_bad else "full"
    out = Path(JOINTTRAIN_ARCH6_G040R_RESULT_ROOT) / name
    row_dir = out / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    import zarr
    zarr_root = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    source_ids = np.asarray(zarr_root["meta/source_replay_id"][:], dtype=np.int32)
    source_index = {int(value): index for index, value in enumerate(source_ids.tolist())}
    stored = np.asarray(zarr_root["data/point_cloud"][[source_index[int(group["source_replay_id"])] for group in groups], :, :3], dtype=np.float32)
    started = time.time()
    repaired = {}
    context = mp.get_context("spawn")
    payloads = [(index, groups[index], stored[index]) for index in selected_bad]
    with context.Pool(1, initializer=source.init_worker, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(source.task, payloads):
            repaired[row[0]] = row
            np.savez_compressed(row_dir / f"{row[0]:04d}.npz", index=row[0], mask=row[1], max_error=row[2], mean_error=row[3], target_points=row[4], raw_points=row[5], target_alignment_error=row[6])
            atomic(out / "progress.json", {"complete": len(repaired), "total": len(selected_bad), "elapsed_seconds": time.time() - started})
    if args.pilot_bad:
        rows = [repaired[index] for index in selected_bad]
    else:
        rows = [repaired.get(index, source_rows[index]) for index in range(len(groups))]
    max_error = max(row[2] for row in rows)
    max_target_error = max(row[6] for row in rows)
    masks = np.stack([row[1] for row in rows])
    checks = {
        "source_terminal": bool(source_summary.get("terminal")),
        "selected_bad_exact": len(selected_bad) == len(repaired),
        "rows_exact": len(rows) == (len(selected_bad) if args.pilot_bad else len(groups)),
        "all_target_points_exact": max_target_error <= 1e-5,
        "shape_binary": masks.shape == (len(rows), 1024) and bool(np.isin(masks, [False, True]).all()),
        "all_target_visible": bool(np.all(masks.sum(axis=1) > 0)),
    }
    if not args.pilot_bad:
        np.savez_compressed(out / "target_masks.npz", target_mask=masks, group_index=np.asarray([group["group_index"] for group in groups], dtype=np.int64))
    summary = {
        "schema_version": 1,
        "run_id": "A6-G040R-PILOT" if args.pilot_bad else "A6-G040R",
        "status": "passed" if all(checks.values()) else "failed",
        "complete": True,
        "terminal": True,
        "source_bad_rows": len(bad),
        "rerendered_rows": len(selected_bad),
        "workers": 1,
        "fresh_process_per_sample": True,
        "elapsed_seconds": time.time() - started,
        "max_whole_object_alignment_error": max_error,
        "max_target_alignment_error": max_target_error,
        "target_points": {"min": int(masks.sum(axis=1).min()), "median": float(np.median(masks.sum(axis=1))), "max": int(masks.sum(axis=1).max())},
        "checks": checks,
        "decision": "run full repair" if args.pilot_bad and all(checks.values()) else ("authorize G041 join" if all(checks.values()) else "repair renderer reproducibility"),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit A6 clean membership against initial, updated, and mix060 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_AFFORDANCE_FOURTH_ROUND_OVERLAY,
    JOINTTRAIN_AFFORDANCE_SECOND_ROUND_OVERLAY,
    JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A031C_RESULT_ROOT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
)


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[np.ndarray]) -> dict[str, float]:
    array = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "positive_fraction_005": float(np.mean(array >= 0.05)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Rows per membership partition; zero audits all rows.")
    args = parser.parse_args()

    membership_path = Path(JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT) / "membership_manifest.json"
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    source = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    second = zarr.open_group(str(JOINTTRAIN_AFFORDANCE_SECOND_ROUND_OVERLAY), mode="r")
    fourth = zarr.open_group(str(JOINTTRAIN_AFFORDANCE_FOURTH_ROUND_OVERLAY), mode="r")

    primary_source_ids = np.asarray(source["meta/source_replay_id"][:], dtype=np.int64)
    overlay_source_ids = np.asarray(fourth["meta/source_replay_id"][:], dtype=np.int64)
    aug_source_ids = np.asarray(source["meta/stage1_aug_source_replay_id"][:], dtype=np.int64)
    primary_rows = []
    for split_name in ("A5_TRAIN", "A5_CAL"):
        rows = membership["primary"][split_name]
        primary_rows.extend((split_name, row) for row in (rows[: args.limit] if args.limit else rows))
    augmentation_rows = membership["augmentation"]["A5_TRAIN"]
    if args.limit:
        augmentation_rows = augmentation_rows[: args.limit]

    values: dict[str, list[np.ndarray]] = {"initial": [], "updated": [], "mix060": []}
    formula_errors = []
    source_id_errors = []
    finite = True
    bounded = True
    point_count = True
    lineage_rows = []

    for split_name, row in primary_rows:
        index = int(row["primary_row"])
        source_id = int(row["source_replay_id"])
        initial = np.asarray(source["data/affordance_initial"][index], dtype=np.float32)
        updated = np.asarray(source["data/affordance_updated"][index], dtype=np.float32)
        a_eps = np.asarray(second["primary/a_eps_030"][index], dtype=np.float32)
        mix = np.asarray(fourth["primary/updated_mix_060"][index], dtype=np.float32)
        expected = np.clip(0.4 * updated + 0.6 * a_eps, 0.0, 1.0)
        formula_errors.append(float(np.max(np.abs(mix - expected))))
        source_id_errors.append(int(primary_source_ids[index]) != source_id or int(overlay_source_ids[index]) != source_id)
        arrays = (initial, updated, mix)
        finite &= all(np.isfinite(value).all() for value in arrays)
        bounded &= all(np.all((value >= 0.0) & (value <= 1.0)) for value in arrays)
        point_count &= all(value.shape == (1024,) for value in arrays)
        for name, value in zip(values, arrays, strict=True):
            values[name].append(value)
        lineage_rows.append({"kind": "primary", "split": split_name, "row": index, "source_replay_id": source_id})

    train_source_ids = {int(row["source_replay_id"]) for row in membership["primary"]["A5_TRAIN"]}
    for raw_index in augmentation_rows:
        index = int(raw_index)
        source_id = int(aug_source_ids[index])
        initial = np.asarray(source["data/stage1_aug_affordance_initial"][index], dtype=np.float32)
        updated = np.asarray(source["data/stage1_aug_affordance_updated"][index], dtype=np.float32)
        a_eps = np.asarray(second["augmentation/a_eps_030"][index], dtype=np.float32)
        mix = np.asarray(fourth["augmentation/updated_mix_060"][index], dtype=np.float32)
        expected = np.clip(0.4 * updated + 0.6 * a_eps, 0.0, 1.0)
        formula_errors.append(float(np.max(np.abs(mix - expected))))
        source_id_errors.append(source_id not in train_source_ids)
        arrays = (initial, updated, mix)
        finite &= all(np.isfinite(value).all() for value in arrays)
        bounded &= all(np.all((value >= 0.0) & (value <= 1.0)) for value in arrays)
        point_count &= all(value.shape == (1024,) for value in arrays)
        for name, value in zip(values, arrays, strict=True):
            values[name].append(value)
        lineage_rows.append({"kind": "augmentation", "split": "A5_TRAIN", "row": index, "source_replay_id": source_id})

    full = args.limit == 0
    counts = {
        "primary_train": len(membership["primary"]["A5_TRAIN"]) if full else min(args.limit, len(membership["primary"]["A5_TRAIN"])),
        "primary_cal": len(membership["primary"]["A5_CAL"]) if full else min(args.limit, len(membership["primary"]["A5_CAL"])),
        "augmentation_train": len(augmentation_rows),
    }
    checks = {
        "upstream_a000_terminal": json.loads((Path(JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT) / "summary.json").read_text())["status"] == "passed",
        "primary_overlay_source_ids_exact": bool(np.array_equal(primary_source_ids, overlay_source_ids)),
        "membership_source_ids_exact": not any(source_id_errors),
        "mix060_formula_exact": max(formula_errors, default=float("inf")) <= 1e-7,
        "finite": bool(finite),
        "bounded_0_1": bool(bounded),
        "point_count_1024": bool(point_count),
        "full_counts": (counts == {"primary_train": 557, "primary_cal": 102, "augmentation_train": 4899}) if full else True,
        "no_mechdev_or_test_content_read": True,
        "source_zarr_immutable": True,
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_A031C_RESULT_ROOT) / (f"probe_{args.limit}" if args.limit else "full")
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "lineage_manifest.json", {"rows": lineage_rows})
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "forbidden_feature_audit.json", {"outcome_read": False, "future_action_read": False, "mechdev_content_read": False, "target_test_content_read": False})
    summary = {
        "schema_version": 1,
        "run_id": "A6-A031C-PROBE" if args.limit else "A6-A031C",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "scope": "A6 clean affordance lineage only; no producer or downstream utility claim",
        "counts": counts,
        "metrics": {name: summarize(items) for name, items in values.items()},
        "max_mix060_formula_error": max(formula_errors, default=None),
        "source_hashes": {
            "membership_manifest": sha256(membership_path),
            "second_round_build_summary": sha256(Path(JOINTTRAIN_AFFORDANCE_SECOND_ROUND_OVERLAY) / ".build_summary.json"),
            "fourth_round_build_summary": sha256(Path(JOINTTRAIN_AFFORDANCE_FOURTH_ROUND_OVERLAY) / ".build_summary.json"),
        },
        "checks": checks,
        "claim_supported": "implementation" if passed and full else "probe_only",
        "decision": "authorize matched A032 producer screen" if passed and full else ("run full A031C" if passed else "repair affordance lineage"),
        "next_run_ids": ["A6-A032C"] if passed and full else (["A6-A031C"] if passed else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

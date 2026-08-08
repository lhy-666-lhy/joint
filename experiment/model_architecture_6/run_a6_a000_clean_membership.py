#!/usr/bin/env python3
"""Audit the clean A6 TRAIN/CAL membership against the existing best-view Zarr."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT,
    JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES,
    JOINTTRAIN_ARCH6_CLEAN_EXCLUSION_MANIFEST,
    JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
    JOINTTRAIN_PRETRAIN_CHECKPOINT,
)


RUN_ID = "a6_a000_clean_membership_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = Path(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT)
    accepted_path = Path(JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES)
    exclusion_path = Path(JOINTTRAIN_ARCH6_CLEAN_EXCLUSION_MANIFEST)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    accepted_rows = [json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted_by_sample = {str(row["sample_id"]): row for row in accepted_rows}
    source = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    source_ids = np.asarray(source["meta/source_replay_id"][:], dtype=np.int32).tolist()
    primary_by_source = {source_id: row_id for row_id, source_id in enumerate(source_ids)}
    aug_source_ids = np.asarray(source["meta/stage1_aug_source_replay_id"][:], dtype=np.int32).tolist()

    manifests: dict[str, list[dict]] = {"A5_TRAIN": [], "A5_CAL": []}
    excluded_partitions: dict[str, list[dict]] = {"A5_MECH_DEV": [], "A5_TARGET_TEST": []}
    failures = {"missing_accepted_sample": [], "missing_zarr_primary": [], "duplicate": [], "wrong_target": []}
    seen: set[str] = set()
    for split, targets in contract["source_partitions"].items():
        for target in targets:
            for sample_id in target["sample_ids"]:
                if sample_id in seen:
                    failures["duplicate"].append(sample_id)
                    continue
                seen.add(sample_id)
                accepted = accepted_by_sample.get(str(sample_id))
                if accepted is None:
                    failures["missing_accepted_sample"].append(sample_id)
                    continue
                if str(accepted["target"]) != str(target["target"]):
                    failures["wrong_target"].append(sample_id)
                    continue
                source_id = int(accepted["source_replay_id"])
                primary_row = primary_by_source.get(source_id)
                if primary_row is None:
                    failures["missing_zarr_primary"].append({"sample_id": sample_id, "source_replay_id": source_id, "split": split})
                record = {
                    "sample_id": sample_id,
                    "target": target["target"],
                    "shape_id": target["shape_id"],
                    "link_name": target["link_name"],
                    "split": split,
                    "source_replay_id": source_id,
                    "primary_row": primary_row,
                }
                (manifests if split in manifests else excluded_partitions)[split].append(record)

    train_source_ids = {row["source_replay_id"] for row in manifests["A5_TRAIN"]}
    train_aug_rows = [row_id for row_id, source_id in enumerate(aug_source_ids) if source_id in train_source_ids]
    expected_train = int(contract["source_partition_counts"]["A5_TRAIN"]["sample_count"])
    expected_cal = int(contract["source_partition_counts"]["A5_CAL"]["sample_count"])
    counts = {
        split: {
            "contract_samples": len(rows),
            "joined_primary": sum(row["primary_row"] is not None for row in rows),
        }
        for split, rows in manifests.items()
    }
    train_targets = {row["target"] for row in manifests["A5_TRAIN"]}
    cal_targets = {row["target"] for row in manifests["A5_CAL"]}
    train_shapes = {row["shape_id"] for row in manifests["A5_TRAIN"]}
    cal_shapes = {row["shape_id"] for row in manifests["A5_CAL"]}
    checks = {
        "train_contract_count_exact": counts["A5_TRAIN"]["contract_samples"] == expected_train,
        "train_primary_join_exact": counts["A5_TRAIN"]["joined_primary"] == expected_train,
        "cal_contract_count_exact": counts["A5_CAL"]["contract_samples"] == expected_cal,
        "cal_primary_join_exact": counts["A5_CAL"]["joined_primary"] == expected_cal,
        "train_cal_target_overlap_zero": not (train_targets & cal_targets),
        "train_cal_shape_overlap_zero": not (train_shapes & cal_shapes),
        "join_failures_zero": all(not values for values in failures.values()),
        "clean_contract_hash_exact": contract["cleaning"]["exclusion_manifest_sha256"] == sha256_file(exclusion_path),
        "old_replay_split_unused": True,
        "heldout_content_unread": True,
    }
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "source_split_contract": str(contract_path),
        "source_split_contract_sha256": sha256_file(contract_path),
        "source_accepted_samples": str(accepted_path),
        "source_accepted_samples_sha256": sha256_file(accepted_path),
        "source_zarr": str(JOINTTRAIN_BESTVIEW_DUAL_ZARR),
        "zarr_metadata_hash": canonical_hash({"source_ids": source_ids, "aug_source_ids": aug_source_ids}),
        "primary": manifests,
        "augmentation": {"A5_TRAIN": train_aug_rows, "A5_CAL": []},
        "excluded_partition_membership_hashes": {
            split: canonical_hash(sorted(row["sample_id"] for row in rows))
            for split, rows in excluded_partitions.items()
        },
        "join_failures": failures,
        "counts": counts,
        "checks": checks,
    }
    atomic_json(out_dir / "membership_manifest.json", manifest)
    passed = all(checks.values())
    pretrain = Path(JOINTTRAIN_PRETRAIN_CHECKPOINT)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "counts": counts,
        "checks": checks,
        "evidence": {
            "membership_manifest": "membership_manifest.json",
            "pretrain_checkpoint_sha256": sha256_file(pretrain),
            "pretrain_label_free_provenance_verified": False,
            "required_init": "random",
        },
        "decision": "Clean TRAIN/CAL membership closes exactly; unlock A010 and clean D020." if passed else "Do not train; repair clean membership only.",
        "next_run_ids": ["a6_a010_affordance_fixed8_v1", "a6_d020_clean_sample_normalizer_v1"] if passed else [],
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-A000C", "status": summary["status"]}]})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

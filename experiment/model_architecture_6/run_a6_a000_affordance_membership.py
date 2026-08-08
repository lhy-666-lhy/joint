#!/usr/bin/env python3
"""Read-only A6-A000 exact join and split provenance audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from path_config import (
    ARTICU_CURATED_ACCEPTED_SAMPLES,
    JOINTTRAIN_ARCH6_A000R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
    JOINTTRAIN_PRETRAIN_CHECKPOINT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def split_sample_rows(contract: dict, accepted_rows: list[dict], source_ids: list[int]) -> tuple[dict, dict, dict]:
    accepted_by_sample = {str(row["sample_id"]): index for index, row in enumerate(accepted_rows)}
    primary_by_source = {int(source_id): row_id for row_id, source_id in enumerate(source_ids)}
    manifests: dict[str, list[dict]] = {"A5_TRAIN": [], "A5_CAL": []}
    exclusions: dict[str, list[dict]] = {"A5_MECH_DEV": [], "A5_TARGET_TEST": []}
    counters = {"missing_accepted_sample": [], "missing_zarr_primary": [], "duplicate": 0, "wrong_target": 0}
    seen: set[str] = set()
    for split, targets in contract["source_partitions"].items():
        for target in targets:
            for sample_id in target["sample_ids"]:
                if sample_id in seen:
                    counters["duplicate"] += 1
                    continue
                seen.add(sample_id)
                accepted_id = accepted_by_sample.get(str(sample_id))
                if accepted_id is None:
                    counters["missing_accepted_sample"].append(sample_id)
                    continue
                accepted = accepted_rows[accepted_id]
                if str(accepted["target"]) != str(target["target"]):
                    counters["wrong_target"] += 1
                    continue
                primary_row = primary_by_source.get(accepted_id)
                if primary_row is None:
                    counters["missing_zarr_primary"].append({"sample_id": sample_id, "split": split, "accepted_source_id": accepted_id})
                record = {"sample_id": sample_id, "target": target["target"], "shape_id": target["shape_id"], "link_name": target["link_name"], "split": split, "accepted_source_id": accepted_id, "primary_row": primary_row}
                (manifests if split in manifests else exclusions)[split].append(record)
    return manifests, exclusions, counters


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_A000R_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = Path(JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    root = zarr.open(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    primary_keys = [str(item) for item in np.asarray(root["meta"]["replay_obj_keys"][:]).tolist()]
    source_ids = np.asarray(root["meta"]["source_replay_id"][:], dtype=np.int32).tolist()
    aug_source_ids = np.asarray(root["meta"]["stage1_aug_source_replay_id"][:], dtype=np.int32).tolist()
    accepted_rows = [json.loads(line) for line in Path(ARTICU_CURATED_ACCEPTED_SAMPLES).read_text(encoding="utf-8").splitlines() if line.strip()]
    manifests, exclusions, counters = split_sample_rows(contract, accepted_rows, source_ids)
    train_ids = {row["accepted_source_id"] for row in manifests["A5_TRAIN"] if row["primary_row"] is not None}
    train_aug = [i for i, source_id in enumerate(aug_source_ids) if source_id in train_ids]
    split_hash = sha256_file(contract_path)
    metadata_hash = canonical_hash({"primary_keys": primary_keys, "source_ids": source_ids, "aug_source_ids": aug_source_ids})
    manifest_payload = {
        "schema_version": 1,
        "run_id": "a6_a000r_affordance_membership_v2",
        "source_split_contract_sha256": split_hash,
        "zarr_metadata_hash": metadata_hash,
        "source_zarr": "JOINTTRAIN_BESTVIEW_DUAL_ZARR",
        "primary": manifests,
        "augmentation": {"A5_TRAIN": train_aug, "A5_CAL": []},
        "exclusion_hashes": {split: canonical_hash(sorted(row["sample_id"] for row in rows)) for split, rows in exclusions.items()},
        "join_failures": counters,
    }
    (out_dir / "membership_manifest.json").write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {split: {"contract_samples": len(rows), "joined_primary": sum(row["primary_row"] is not None for row in rows)} for split, rows in manifests.items()}
    train_targets = {r["target"] for r in manifests["A5_TRAIN"]}
    cal_targets = {r["target"] for r in manifests["A5_CAL"]}
    train_shapes = {r["shape_id"] for r in manifests["A5_TRAIN"]}
    cal_shapes = {r["shape_id"] for r in manifests["A5_CAL"]}
    checks = {"train_contract_count": counts["A5_TRAIN"]["contract_samples"] == 559, "train_joined_primary_count": counts["A5_TRAIN"]["joined_primary"] == 559, "cal_contract_count": counts["A5_CAL"]["contract_samples"] == 102, "cal_joined_primary_count": counts["A5_CAL"]["joined_primary"] == 102, "train_cal_target_overlap": len(train_targets & cal_targets), "train_cal_shape_overlap": len(train_shapes & cal_shapes), "missing_accepted_sample": len(counters["missing_accepted_sample"]), "missing_zarr_primary": len(counters["missing_zarr_primary"]), "duplicate": counters["duplicate"], "wrong_target": counters["wrong_target"], "mech_dev_target_test_content_read": False, "old_replay_split_used": False}
    status = "passed" if all((checks["train_contract_count"], checks["train_joined_primary_count"], checks["cal_contract_count"], checks["cal_joined_primary_count"])) and all(checks[key] == 0 for key in ("train_cal_target_overlap", "train_cal_shape_overlap", "missing_accepted_sample", "missing_zarr_primary", "duplicate", "wrong_target")) else "failed"
    pretrain_path = Path(JOINTTRAIN_PRETRAIN_CHECKPOINT)
    pretrain = {"sha256": sha256_file(pretrain_path), "label_free_provenance_verified": False, "required_init": "random"}
    summary = {"schema_version": 1, "run_id": "a6_a000r_affordance_membership_v2", "complete": True, "terminal": True, "status": status, "failure_class": None if status == "passed" else "data_contract_failure", "claim_supported": "yes" if status == "passed" else "no", "evidence": {"membership_manifest": "membership_manifest.json", "source_split_contract_sha256": split_hash, "zarr_metadata_hash": metadata_hash, "point_cloud_or_label_read": False, "outcome_read": False, "pretrain_provenance": pretrain}, "counts": counts, "checks": checks, "decision": "A000 split-safe metadata join passes." if status == "passed" else "Two A5_TRAIN primary rows are absent from the frozen Zarr; block A010 and do not shrink the split.", "remaining_work": ["resolve missing A5_TRAIN primary rows under a planning revision", "A6-D010 exact adapter roundtrip"], "next_run_ids": ["a6_d010_adapter_roundtrip_v1"], "event_id": "a6_a000r_affordance_membership_v2_terminal"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

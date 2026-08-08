#!/usr/bin/env python3
"""Publish the A6 data-cleaning revision without deleting raw collection data.

The existing best-view affordance Zarr already excludes three source replay IDs.
This run makes that exclusion explicit in the A6 split and sample manifests so
the action, point-cloud, and affordance consumers use the same population.
"""

from __future__ import annotations

import copy
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
    ARTICU_COLLECTION_ROOT,
    ARTICU_CURATED_ACCEPTED_SAMPLES,
    JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT,
    JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES,
    JOINTTRAIN_ARCH6_CLEAN_EXCLUSION_MANIFEST,
    JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT,
    JOINTTRAIN_ARCH6_DATA_CLEAN_RESULT_ROOT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
    JOINTTRAIN_REPLAY_MANIFEST,
)


RUN_ID = "a6_r100_data_clean_contract_v1"
REVISION_ID = "20260806T-data-clean-70-251-349"
EXCLUSIONS = (
    {
        "source_replay_id": 70,
        "sample_id": "11550/link_1/repeat_1/base_0000",
        "split": "A5_TRAIN",
        "reason": "visible_positive_points_below_16_in_all_debug_views",
        "visible_positive_debug": {"views": 10, "min": 2, "max": 3, "mean": 2.1, "required_min": 16},
    },
    {
        "source_replay_id": 251,
        "sample_id": "45397/link_1/repeat_1/base_0000",
        "split": "A5_TRAIN",
        "reason": "visible_positive_points_below_16_in_all_debug_views",
        "visible_positive_debug": {"views": 10, "min": 1, "max": 4, "mean": 2.5, "required_min": 16},
    },
    {
        "source_replay_id": 349,
        "sample_id": "45670/link_1/repeat_1/base_0000",
        "split": "A5_MECH_DEV",
        "reason": "visible_positive_points_below_16_in_all_debug_views",
        "visible_positive_debug": {"views": 10, "min": 2, "max": 4, "mean": 3.0, "required_min": 16},
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sample_base(row: dict) -> Path:
    return (Path(ARTICU_COLLECTION_ROOT) / str(row["relative_heatmap_npz"])).parents[1]


def partition_counts(partitions: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, rows in partitions.items():
        result[name] = {
            "shape_count": len({str(row["shape_id"]) for row in rows}),
            "target_count": len({str(row["target"]) for row in rows}),
            "sample_count": sum(int(row["sample_count"]) for row in rows),
        }
    return result


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_DATA_CLEAN_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    old_contract_path = Path(JOINTTRAIN_ARCH6_A000_SPLIT_CONTRACT)
    accepted_path = Path(ARTICU_CURATED_ACCEPTED_SAMPLES)
    replay_manifest_path = Path(JOINTTRAIN_REPLAY_MANIFEST)
    old_contract = json.loads(old_contract_path.read_text(encoding="utf-8"))
    accepted_rows = [json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    replay_rows = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    excluded_ids = {int(item["source_replay_id"]) for item in EXCLUSIONS}
    excluded_samples = {str(item["sample_id"]) for item in EXCLUSIONS}
    accepted_by_sample = {str(row["sample_id"]): (index, row) for index, row in enumerate(accepted_rows)}

    source_zarr = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    primary_source_ids = np.asarray(source_zarr["meta/source_replay_id"][:], dtype=np.int32)
    primary_ids = set(primary_source_ids.tolist())
    episode_local_ids = np.asarray(source_zarr["meta/episode_replay_ids"][:], dtype=np.int32)
    episode_source_ids = set(primary_source_ids[episode_local_ids].tolist())
    exclusion_rows: list[dict] = []
    checks: dict[str, bool] = {}
    for item in EXCLUSIONS:
        source_id = int(item["source_replay_id"])
        sample_id = str(item["sample_id"])
        accepted_entry = accepted_by_sample.get(sample_id)
        if accepted_entry is None:
            raise ValueError(f"excluded sample is absent from accepted_samples: {sample_id}")
        accepted_index, accepted = accepted_entry
        replay = replay_rows[source_id]
        if str(replay["obj_key"]) != f"{accepted['shape_id']}_{accepted['link_name']}":
            raise ValueError(f"source replay/object mismatch for {source_id}")
        base = sample_base(accepted)
        heatmap_path = base / "heatmap" / "heatmap_data.npz"
        trajectory_index = base / "trajectory" / "index.json"
        with np.load(heatmap_path, allow_pickle=True) as heatmap:
            scores = np.asarray(heatmap["scores"], dtype=np.float32).reshape(-1)
            candidate_success = np.asarray(heatmap["candidate_success"], dtype=bool).reshape(-1)
            point_count = int(np.asarray(heatmap["points"]).reshape(-1, 3).shape[0])
            positive_score_count = int(np.count_nonzero(scores > 0.0))
            success_count = int(np.count_nonzero(candidate_success))
        trajectory_names = json.loads(trajectory_index.read_text(encoding="utf-8"))["trajectories"]
        trajectory_paths = []
        for value in trajectory_names:
            candidate = Path(str(value))
            path = candidate if candidate.is_file() else trajectory_index.parent / candidate.name
            if not path.is_file():
                raise FileNotFoundError(path)
            trajectory_paths.append(path)
        exclusion_rows.append(
            {
                **item,
                "accepted_source_id": accepted_index,
                "target": accepted["target"],
                "point_count": point_count,
                "positive_score_count": positive_score_count,
                "success_count": success_count,
                "trajectory_count": len(trajectory_paths),
                "trajectory_relative_paths": [path.relative_to(Path(ARTICU_COLLECTION_ROOT)).as_posix() for path in trajectory_paths],
                "raw_sample_root": str(base),
                "derived_pointcloud_present": source_id in primary_ids,
                "derived_action_present": source_id in episode_source_ids,
            }
        )

    checks["all_exclusion_rows_valid"] = len(exclusion_rows) == len(EXCLUSIONS)
    checks["derived_primary_excludes_exact_ids"] = not (primary_ids & excluded_ids) and len(primary_ids) == len(replay_rows) - len(EXCLUSIONS)
    checks["derived_actions_exclude_exact_ids"] = not (episode_source_ids & excluded_ids)
    checks["raw_data_retained"] = all(Path(row["raw_sample_root"]).is_dir() for row in exclusion_rows)
    checks["all_excluded_rows_have_positive_labels"] = all(row["positive_score_count"] > 0 and row["success_count"] > 0 for row in exclusion_rows)

    exclusion_manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "revision_id": REVISION_ID,
        "parent_split_contract": str(old_contract_path),
        "parent_split_contract_sha256": sha256_file(old_contract_path),
        "source_zarr": str(JOINTTRAIN_BESTVIEW_DUAL_ZARR),
        "source_zarr_summary_sha256": sha256_file(Path(JOINTTRAIN_BESTVIEW_DUAL_ZARR) / ".zarr_summary.json"),
        "policy": "quarantine raw collection; exclude the exact repeat/base from all A6 derived action, point-cloud, and affordance manifests",
        "raw_data_deleted": False,
        "excluded_source_replay_ids": sorted(excluded_ids),
        "rows": exclusion_rows,
        "checks": checks,
    }
    atomic_json(Path(JOINTTRAIN_ARCH6_CLEAN_EXCLUSION_MANIFEST), exclusion_manifest)
    exclusion_hash = sha256_file(Path(JOINTTRAIN_ARCH6_CLEAN_EXCLUSION_MANIFEST))

    clean_accepted = [
        {**row, "source_replay_id": index}
        for index, row in enumerate(accepted_rows)
        if str(row["sample_id"]) not in excluded_samples
    ]
    write_jsonl(Path(JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES), clean_accepted)

    clean_contract = copy.deepcopy(old_contract)
    clean_contract["schema_version"] = 1
    clean_contract["contract_version"] = "A6-DATA-v1.2"
    clean_contract["iteration_id"] = RUN_ID
    for split, targets in clean_contract["source_partitions"].items():
        retained_targets = []
        for target in targets:
            sample_ids = [sample_id for sample_id in target["sample_ids"] if sample_id not in excluded_samples]
            if not sample_ids:
                continue
            target["sample_ids"] = sample_ids
            target["sample_count"] = len(sample_ids)
            target["split"] = split
            retained_targets.append(target)
        clean_contract["source_partitions"][split] = retained_targets
    clean_contract["source_partition_counts"] = partition_counts(clean_contract["source_partitions"])
    old_total = sum(int(value["sample_count"]) for value in old_contract["source_partition_counts"].values())
    clean_total = sum(int(value["sample_count"]) for value in clean_contract["source_partition_counts"].values())
    clean_contract["coverage"] = {
        "parent_source_sample_count": old_total,
        "clean_source_sample_count": clean_total,
        "excluded_sample_count": len(EXCLUSIONS),
        "excluded_source_replay_ids": sorted(excluded_ids),
    }
    clean_contract["cleaning"] = {
        "revision_id": REVISION_ID,
        "exclusion_manifest": str(JOINTTRAIN_ARCH6_CLEAN_EXCLUSION_MANIFEST),
        "exclusion_manifest_sha256": exclusion_hash,
        "clean_accepted_samples": str(JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES),
        "raw_data_deleted": False,
    }
    clean_contract["checks"] = {
        **checks,
        "excluded_samples_absent_from_clean_contract": all(
            sample_id not in {
                value
                for targets in clean_contract["source_partitions"].values()
                for target in targets
                for value in target["sample_ids"]
            }
            for sample_id in excluded_samples
        ),
        "clean_accepted_count_exact": len(clean_accepted) == len(accepted_rows) - len(EXCLUSIONS),
    }
    atomic_json(Path(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT), clean_contract)
    passed = all(clean_contract["checks"].values())
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "revision_id": REVISION_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "counts": {
            "parent_samples": old_total,
            "clean_samples": clean_total,
            "excluded_samples": len(EXCLUSIONS),
            "clean_train_samples": clean_contract["source_partition_counts"]["A5_TRAIN"]["sample_count"],
            "clean_cal_samples": clean_contract["source_partition_counts"]["A5_CAL"]["sample_count"],
            "clean_mech_dev_samples": clean_contract["source_partition_counts"]["A5_MECH_DEV"]["sample_count"],
            "clean_target_test_samples": clean_contract["source_partition_counts"]["A5_TARGET_TEST"]["sample_count"],
        },
        "checks": clean_contract["checks"],
        "decision": "Use the clean contract and existing derived Zarr; do not run a recovery producer or physically delete raw collection data." if passed else "Do not train; repair the clean contract audit.",
        "evidence": {
            "exclusion_manifest": Path(JOINTTRAIN_ARCH6_CLEAN_EXCLUSION_MANIFEST).name,
            "clean_accepted_samples": Path(JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES).name,
            "clean_split_contract": Path(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT).name,
        },
        "next_run_ids": ["a6_a000_clean_membership_v1", "a6_d020_clean_sample_normalizer_v1"] if passed else [],
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-R100", "status": summary["status"]}]})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze clean A6 grasp trajectory, open-qpose, and closed-state labels."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (  # noqa: E402
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH5_C020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES,
    JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT,
    JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,
)
from a6_grasp_path_utils import resample_joint_path  # noqa: E402


RUN_ID = "a6_g000c_grasp_label_set_contract_v3"
ALLOWED_SPLITS = ("A5_TRAIN", "A5_CAL")
K = 4
PATH_LENGTH = 64
EXPECTED_ORIGINAL_GROUPS = {"A5_TRAIN": 557, "A5_CAL": 102}
EXPECTED_GROUPS = {"A5_TRAIN": 531, "A5_CAL": 101}
EXPECTED_EXCLUDED_ZERO_TEACHER_GROUPS = {"A5_TRAIN": 26, "A5_CAL": 1}
EXPECTED_SOURCE_TRAJECTORIES = {"A5_TRAIN": 26505, "A5_CAL": 4918}
EXPECTED_REPLAY_PASS_TRAJECTORIES = {"A5_TRAIN": 22242, "A5_CAL": 3952}
EXPECTED_SELECTED_TEACHERS = 2373
LOADED_TRAJECTORY_FIELDS = (
    "grasp_plan_qpath",
    "operation_start_robot_qpos",
    "base_pose",
    "initial_object_qpos",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_teacher_values(values: list[Any], k: int = K) -> list[Any]:
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("trajectory index contains duplicate entries")
    return list(values[:k])


def resolve_trajectory(index_path: Path, value: Any) -> Path:
    candidate = Path(str(value))
    path = candidate if candidate.is_file() else index_path.parent / candidate.name
    path = path.resolve()
    if not path.is_file() or path.parent != index_path.parent.resolve():
        raise FileNotFoundError(path)
    return path


def replay_quality_file(collection_root: Path, initial: dict[str, Any]) -> Path:
    name = (
        f"{initial['shape_id']}_{initial['link_name']}_"
        f"size_{initial['size']}"
    )
    return collection_root / "replay_quality" / name / "replay_quality.json"


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested
            for child in value.values()
            for nested in recursive_keys(child)
        }
    if isinstance(value, list):
        return {nested for child in value for nested in recursive_keys(child)}
    return set()


def main() -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = now()
    running = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": False,
        "terminal": False,
        "status": "running",
        "resource_mode": "cpu",
        "pid": os.getpid(),
        "started_at": started_at,
    }
    atomic_json(out_dir / "run_state.json", running)
    atomic_json(
        out_dir / "queue_state.json",
        {**running, "jobs": [{"id": "A6-G000C", "status": "running"}]},
    )
    atomic_json(
        out_dir / "command.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    )

    try:
        collection_root = Path(ARTICU_COLLECTION_ROOT)
        split_path = Path(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT)
        accepted_path = Path(JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES)
        parent_summary_path = Path(JOINTTRAIN_ARCH5_C020_RESULT_ROOT) / "summary.json"
        parent_report_path = Path(JOINTTRAIN_ARCH5_C020_RESULT_ROOT) / "label_contract_report.json"
        split_contract = json.loads(split_path.read_text(encoding="utf-8"))
        accepted_rows = load_jsonl(accepted_path)
        accepted_by_id = {str(row["sample_id"]): row for row in accepted_rows}
        parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))

        raw_group_specs: list[tuple[str, dict[str, Any], str]] = []
        for split in ALLOWED_SPLITS:
            for target in sorted(
                split_contract["source_partitions"][split], key=lambda row: row["target"]
            ):
                for sample_id in sorted(target["sample_ids"]):
                    raw_group_specs.append((split, target, str(sample_id)))

        replay_cache: dict[Path, tuple[set[str], set[str]]] = {}
        replay_source_hashes: dict[str, str] = {}
        group_specs: list[dict[str, Any]] = []
        excluded_groups: list[dict[str, Any]] = []
        original_group_counts = {split: 0 for split in ALLOWED_SPLITS}
        excluded_group_counts = {split: 0 for split in ALLOWED_SPLITS}
        for split, target_row, sample_id in raw_group_specs:
            original_group_counts[split] += 1
            sample_dir = collection_root / "data" / "single" / Path(sample_id)
            initial_path = sample_dir / "initial_state.json"
            index_path = sample_dir / "trajectory" / "index.json"
            initial = json.loads(initial_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            values = list(index["trajectories"])
            if int(index["trajectory_count"]) != len(values) or not values:
                raise ValueError(f"invalid trajectory index: {index_path}")
            quality_path = replay_quality_file(collection_root, initial)
            if quality_path not in replay_cache:
                quality = json.loads(quality_path.read_text(encoding="utf-8"))
                all_paths = {
                    str(Path(row["trajectory_npz"]).resolve())
                    for row in quality["rows"]
                }
                pass_paths = {
                    str(Path(row["trajectory_npz"]).resolve())
                    for row in quality["rows"]
                    if row["quality_label"] == "fixed_gate_pass"
                }
                replay_cache[quality_path] = (all_paths, pass_paths)
                replay_source_hashes[
                    quality_path.relative_to(collection_root).as_posix()
                ] = sha256_file(quality_path)
            all_paths, pass_paths = replay_cache[quality_path]
            resolved = [(value, str(resolve_trajectory(index_path, value))) for value in values]
            missing = [path for _, path in resolved if path not in all_paths]
            if missing:
                raise ValueError(
                    f"trajectory index has rows absent from replay quality: {missing[:3]}"
                )
            replay_pass_values = [value for value, path in resolved if path in pass_paths]
            selected = select_teacher_values(replay_pass_values)
            if not selected:
                excluded_group_counts[split] += 1
                excluded_groups.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "target": str(target_row["target"]),
                        "raw_candidate_count": len(values),
                        "replay_pass_candidate_count": 0,
                        "reason": "zero_fixed_gate_pass_trajectories",
                    }
                )
                continue
            group_specs.append(
                {
                    "split": split,
                    "target_row": target_row,
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                    "initial_path": initial_path,
                    "index_path": index_path,
                    "initial": initial,
                    "values": values,
                    "replay_pass_values": replay_pass_values,
                    "selected": selected,
                    "replay_quality_path": quality_path,
                }
            )

        group_count = len(group_specs)
        initial_arm = np.zeros((group_count, 7), dtype=np.float32)
        presence = np.zeros((group_count, K), dtype=bool)
        path_absolute = np.zeros(
            (group_count, K, PATH_LENGTH, 7), dtype=np.float32
        )
        qpose_absolute = np.zeros((group_count, K, 7), dtype=np.float32)
        closed_absolute = np.zeros((group_count, K, 7), dtype=np.float32)
        trajectory_groups: list[dict[str, Any]] = []
        qpose_groups: list[dict[str, Any]] = []
        split_group_counts = {split: 0 for split in ALLOWED_SPLITS}
        split_source_counts = {split: 0 for split in ALLOWED_SPLITS}
        split_replay_pass_counts = {split: 0 for split in ALLOWED_SPLITS}
        split_teacher_counts = {split: 0 for split in ALLOWED_SPLITS}
        source_counts: list[int] = []
        raw_lengths: list[float] = []
        arc_lengths: list[float] = []
        open_closed_l2: list[float] = []
        initial_qpos_errors: list[float] = []
        base_errors: list[float] = []
        object_errors: list[float] = []
        teacher_ids: set[str] = set()

        for group_index, spec in enumerate(group_specs):
            split = str(spec["split"])
            target_row = spec["target_row"]
            sample_id = str(spec["sample_id"])
            accepted = accepted_by_id.get(sample_id)
            if accepted is None or str(accepted["target"]) != str(target_row["target"]):
                raise ValueError(f"clean sample/target mismatch: {sample_id}")
            initial_path = Path(spec["initial_path"])
            index_path = Path(spec["index_path"])
            initial = spec["initial"]
            values = list(spec["values"])
            replay_pass_values = list(spec["replay_pass_values"])
            selected = list(spec["selected"])
            robot = np.asarray(initial["robot_default_full_qpos"], dtype=np.float32)[:7]
            base = np.asarray(initial["base_pose"], dtype=np.float32)
            initial_object = np.asarray(initial["initial_object_qpos"], dtype=np.float32)
            if robot.shape != (7,) or base.shape != (4,) or not np.isfinite(
                np.concatenate((robot, base, initial_object))
            ).all():
                raise ValueError(f"invalid initial state: {initial_path}")
            initial_arm[group_index] = robot
            split_group_counts[split] += 1
            split_source_counts[split] += len(values)
            split_replay_pass_counts[split] += len(replay_pass_values)
            split_teacher_counts[split] += len(selected)
            source_counts.append(len(replay_pass_values))
            state_payload = {
                "sample_id": sample_id,
                "target": str(target_row["target"]),
                "source_replay_id": int(accepted["source_replay_id"]),
                "relative_heatmap_npz": str(accepted["relative_heatmap_npz"]),
                "initial_arm_qpos": robot.tolist(),
                "initial_object_qpos": initial_object.tolist(),
                "base_pose": base.tolist(),
            }
            group_id = sha256_json(state_payload)
            trajectory_candidates: list[dict[str, Any]] = []
            qpose_candidates: list[dict[str, Any]] = []

            for slot, value in enumerate(selected):
                trajectory_path = resolve_trajectory(index_path, value)
                with np.load(trajectory_path, allow_pickle=False) as data:
                    missing = [field for field in LOADED_TRAJECTORY_FIELDS if field not in data.files]
                    if missing:
                        raise ValueError(f"missing trajectory fields {missing}: {trajectory_path}")
                    qpath = np.asarray(data["grasp_plan_qpath"], dtype=np.float32)
                    closed = np.asarray(
                        data["operation_start_robot_qpos"], dtype=np.float32
                    )[:7]
                    trajectory_base = np.asarray(data["base_pose"], dtype=np.float32)
                    trajectory_object = np.asarray(
                        data["initial_object_qpos"], dtype=np.float32
                    )
                resampled = resample_joint_path(qpath)
                arc_length = float(np.linalg.norm(np.diff(qpath, axis=0), axis=1).sum())
                initial_error = float(np.max(np.abs(qpath[0] - robot)))
                base_error = float(np.max(np.abs(trajectory_base - base)))
                object_error = float(np.max(np.abs(trajectory_object - initial_object)))
                if closed.shape != (7,) or not np.isfinite(closed).all():
                    raise ValueError(f"invalid closed operation state: {trajectory_path}")
                label_hash = sha256_arrays(qpath, closed, trajectory_base, trajectory_object)
                teacher_id = sha256_json(
                    {"group_id": group_id, "slot": slot, "label_sha256": label_hash}
                )
                if teacher_id in teacher_ids:
                    raise ValueError(f"duplicate teacher id: {teacher_id}")
                teacher_ids.add(teacher_id)
                presence[group_index, slot] = True
                path_absolute[group_index, slot] = resampled
                qpose_absolute[group_index, slot] = qpath[-1]
                closed_absolute[group_index, slot] = closed
                relative = trajectory_path.relative_to(collection_root).as_posix()
                trajectory_candidates.append(
                    {
                        "slot": slot,
                        "teacher_id": teacher_id,
                        "trajectory_relative_path": relative,
                        "label_sha256": label_hash,
                        "raw_qpath_length": int(qpath.shape[0]),
                        "raw_qpath_arc_length": arc_length,
                        "replay_quality_label": "fixed_gate_pass",
                    }
                )
                qpose_candidates.append(
                    {
                        "slot": slot,
                        "teacher_id": teacher_id,
                        "label_sha256": label_hash,
                    }
                )
                raw_lengths.append(float(qpath.shape[0]))
                arc_lengths.append(arc_length)
                open_closed_l2.append(float(np.linalg.norm(qpath[-1] - closed)))
                initial_qpos_errors.append(initial_error)
                base_errors.append(base_error)
                object_errors.append(object_error)

            common = {
                "group_index": group_index,
                "group_id": group_id,
                "sample_id": sample_id,
                "source_replay_id": int(accepted["source_replay_id"]),
                "split": split,
                "target": str(target_row["target"]),
                "raw_candidate_count": len(values),
                "available_candidate_count": len(replay_pass_values),
                "selected_candidate_count": len(selected),
                "observation_identity_source": "clean sample_id/source_replay_id",
            }
            trajectory_groups.append({**common, "candidates": trajectory_candidates})
            qpose_groups.append({**common, "candidates": qpose_candidates})
            if (group_index + 1) % 50 == 0:
                atomic_json(
                    out_dir / "run_state.json",
                    {**running, "groups_completed": group_index + 1, "groups_total": group_count},
                )

        path_relative = path_absolute - initial_arm[:, None, None, :]
        qpose_relative = qpose_absolute - initial_arm[:, None, :]
        trajectory_payload = {
            "initial_arm_qpos": initial_arm,
            "path_absolute": path_absolute,
            "path_relative": path_relative,
            "presence": presence,
        }
        qpose_payload = {
            "initial_arm_qpos": initial_arm,
            "qpose_absolute": qpose_absolute,
            "qpose_relative": qpose_relative,
            "presence": presence,
        }
        closed_payload = {
            "q_operation_start_closed": closed_absolute,
            "presence": presence,
        }
        trajectory_manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "route": "G-TRAJ",
            "selection": "fixed_gate_pass filter, then first K in collection trajectory index order",
            "path_resampling": "7D cumulative joint-space arc length with exact endpoints",
            "k": K,
            "path_length": PATH_LENGTH,
            "groups": trajectory_groups,
        }
        qpose_manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "route": "G-QPOSE",
            "terminal_semantics": "open terminal grasp goal",
            "k": K,
            "groups": qpose_groups,
        }
        atomic_npz(out_dir / "traj_labels.npz", trajectory_payload)
        atomic_npz(out_dir / "qpose_labels.npz", qpose_payload)
        atomic_npz(out_dir / "closed_operation_labels.npz", closed_payload)
        atomic_json(out_dir / "traj_teacher_manifest.json", trajectory_manifest)
        atomic_json(out_dir / "qpose_teacher_manifest.json", qpose_manifest)
        atomic_json(
            out_dir / "excluded_zero_teacher_groups.json",
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "counts": excluded_group_counts,
                "groups": excluded_groups,
            },
        )
        atomic_json(
            out_dir / "replay_quality_sources.json",
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "sources": replay_source_hashes,
            },
        )

        with np.load(out_dir / "traj_labels.npz", allow_pickle=False) as reloaded_traj:
            traj_reload_exact = all(
                np.array_equal(reloaded_traj[name], value)
                for name, value in trajectory_payload.items()
            )
            traj_keys = list(reloaded_traj.files)
        with np.load(out_dir / "qpose_labels.npz", allow_pickle=False) as reloaded_qpose:
            qpose_reload_exact = all(
                np.array_equal(reloaded_qpose[name], value)
                for name, value in qpose_payload.items()
            )
            qpose_keys = list(reloaded_qpose.files)

        active_paths = path_absolute[presence]
        active_qposes = qpose_absolute[presence]
        active_closed = closed_absolute[presence]
        active_initial = np.broadcast_to(
            initial_arm[:, None, :], (group_count, K, 7)
        )[presence]
        path_roundtrip_error = float(
            np.max(
                np.abs(
                    path_relative[presence] + active_initial[:, None, :] - active_paths
                )
            )
        )
        qpose_roundtrip_error = float(
            np.max(np.abs(qpose_relative[presence] + active_initial - active_qposes))
        )
        false_path_zero = bool(np.count_nonzero(path_absolute[~presence]) == 0)
        false_qpose_zero = bool(np.count_nonzero(qpose_absolute[~presence]) == 0)
        parent_passed = parent_summary.get("status") == "success" and all(
            bool(value) for value in parent_summary.get("checks", {}).values()
        )
        qpose_manifest_keys = recursive_keys(qpose_manifest)
        checks = {
            "parent_a5_label_contract_passed": parent_passed,
            "clean_split_contract_passed": all(
                bool(value) for value in split_contract["checks"].values()
            ),
            "original_clean_group_counts_exact": original_group_counts
            == EXPECTED_ORIGINAL_GROUPS,
            "excluded_zero_teacher_group_counts_exact": excluded_group_counts
            == EXPECTED_EXCLUDED_ZERO_TEACHER_GROUPS,
            "allowed_group_counts_exact": split_group_counts == EXPECTED_GROUPS,
            "allowed_source_trajectory_counts_exact": split_source_counts
            == EXPECTED_SOURCE_TRAJECTORIES,
            "selected_teacher_count_exact": int(presence.sum())
            == EXPECTED_SELECTED_TEACHERS,
            "replay_pass_trajectory_counts_exact": split_replay_pass_counts
            == EXPECTED_REPLAY_PASS_TRAJECTORIES,
            "all_selected_teachers_replay_pass": all(
                candidate["replay_quality_label"] == "fixed_gate_pass"
                for group in trajectory_groups
                for candidate in group["candidates"]
            ),
            "group_count_exact": group_count == sum(EXPECTED_GROUPS.values()),
            "one_group_per_clean_sample": len({row["sample_id"] for row in trajectory_groups})
            == group_count,
            "candidate_slots_not_duplicated": len(teacher_ids) == int(presence.sum()),
            "missing_slots_not_label_copies": false_path_zero and false_qpose_zero,
            "resampled_start_exact": bool(np.array_equal(active_paths[:, 0], active_initial)),
            "terminal_qpose_exact": bool(np.array_equal(active_paths[:, -1], active_qposes)),
            "relative_path_roundtrip_le_1e_6": path_roundtrip_error <= 1e-6,
            "relative_qpose_roundtrip_le_1e_6": qpose_roundtrip_error <= 1e-6,
            "open_terminal_distinct_from_closed": bool(
                np.min(np.linalg.norm(active_qposes - active_closed, axis=1)) > 0.0
            ),
            "same_initial_state_for_group_candidates": max(
                initial_qpos_errors + base_errors + object_errors
            )
            <= 1e-6,
            "all_arrays_finite": all(
                np.isfinite(value).all()
                for value in (
                    initial_arm,
                    path_absolute,
                    path_relative,
                    qpose_absolute,
                    qpose_relative,
                    closed_absolute,
                )
            ),
            "all_raw_paths_positive_arc": min(arc_lengths) > 0.0,
            "traj_reload_exact": traj_reload_exact,
            "qpose_reload_exact": qpose_reload_exact,
            "qpose_npz_physically_excludes_paths": not any(
                "path" in name.lower() for name in qpose_keys
            ),
            "qpose_manifest_excludes_trajectory_paths": not any(
                "path" in name.lower() for name in qpose_manifest_keys
            ),
            "train_cal_only": set(split_group_counts) == set(ALLOWED_SPLITS),
        }
        metrics = {
            "group_counts": split_group_counts,
            "source_trajectory_counts": split_source_counts,
            "replay_pass_trajectory_counts": split_replay_pass_counts,
            "selected_teacher_counts": split_teacher_counts,
            "selected_teacher_count": int(presence.sum()),
            "groups_with_fewer_than_k": int(np.count_nonzero(presence.sum(axis=1) < K)),
            "available_candidate_count": percentile_summary(
                [float(value) for value in source_counts]
            ),
            "raw_qpath_length": percentile_summary(raw_lengths),
            "raw_qpath_arc_length": percentile_summary(arc_lengths),
            "open_closed_l2": percentile_summary(open_closed_l2),
            "initial_qpos_max_abs_error": max(initial_qpos_errors),
            "base_pose_max_abs_error": max(base_errors),
            "initial_object_qpos_max_abs_error": max(object_errors),
            "relative_path_roundtrip_max_abs": path_roundtrip_error,
            "relative_qpose_roundtrip_max_abs": qpose_roundtrip_error,
        }
        forbidden_audit = {
            "model_input_forbidden_fields": [],
            "loaded_trajectory_fields": list(LOADED_TRAJECTORY_FIELDS),
            "result_json_read": False,
            "replay_quality_gate_read_for_teacher_filter": True,
            "task_progress_or_final_outcome_read": False,
            "mech_dev_data_read": False,
            "target_test_data_read": False,
            "qpose_npz_fields": qpose_keys,
            "qpose_manifest_fields": sorted(qpose_manifest_keys),
            "qpose_consumer_allowed_artifacts": [
                "qpose_labels.npz",
                "qpose_teacher_manifest.json",
            ],
            "traj_consumer_allowed_artifacts": [
                "traj_labels.npz",
                "traj_teacher_manifest.json",
            ],
        }
        atomic_json(out_dir / "forbidden_feature_audit.json", forbidden_audit)
        passed = all(checks.values())
        completed_at = now()
        summary = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "complete": True,
            "terminal": True,
            "status": "passed" if passed else "failed",
            "failure_class": None if passed else "data_contract_failure",
            "scientific_scope": "clean TRAIN/CAL grasp K4 label-set materialization",
            "implementation_revision": "v3 excludes replay-fail teachers and zero-valid-teacher groups",
            "selection_policy": "filter fixed_gate_pass, then first K by collection index; no task-outcome ranking",
            "metrics": metrics,
            "checks": checks,
            "sources": {
                "clean_split_contract_sha256": sha256_file(split_path),
                "clean_accepted_samples_sha256": sha256_file(accepted_path),
                "parent_label_summary_sha256": sha256_file(parent_summary_path),
                "parent_label_report_sha256": sha256_file(parent_report_path),
                "replay_quality_sources_sha256": sha256_json(replay_source_hashes),
            },
            "artifacts": {
                name: sha256_file(out_dir / name)
                for name in (
                    "traj_labels.npz",
                    "qpose_labels.npz",
                    "closed_operation_labels.npz",
                    "traj_teacher_manifest.json",
                    "qpose_teacher_manifest.json",
                    "forbidden_feature_audit.json",
                    "excluded_zero_teacher_groups.json",
                    "replay_quality_sources.json",
                )
            },
            "decision": (
                "authorize A6-G005C exact-paired GT-TRAJ versus GT-QPOSE interface test"
                if passed
                else "repair G000C label materialization without changing grasp representation"
            ),
            "next_run_ids": ["A6-G005C"] if passed else [],
            "started_at": started_at,
            "completed_at": completed_at,
        }
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(
            out_dir / "queue_state.json",
            {**summary, "jobs": [{"id": "A6-G000C", "status": summary["status"]}]},
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if passed else 2
    except Exception as exc:
        completed_at = now()
        failure = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "complete": True,
            "terminal": True,
            "status": "failed",
            "failure_class": "implementation_crash",
            "error": f"{type(exc).__name__}: {exc}",
            "decision": "repair G000C implementation without changing label semantics",
            "started_at": started_at,
            "completed_at": completed_at,
        }
        atomic_json(out_dir / "summary.json", failure)
        atomic_json(out_dir / "run_state.json", failure)
        atomic_json(
            out_dir / "queue_state.json",
            {**failure, "jobs": [{"id": "A6-G000C", "status": "failed"}]},
        )
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

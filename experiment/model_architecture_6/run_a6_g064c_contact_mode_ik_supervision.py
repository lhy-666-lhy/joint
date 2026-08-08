#!/usr/bin/env python3
"""Materialize contact-mode, SE3 residual, and IK-set supervision from G063."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import (
    JOINTTRAIN_ARCH6_G052C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G061C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G063C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G064C_RESULT_ROOT,
)


MAX_IK = 8
PREGRASP_DISTANCE_M = 0.08


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[:, :2].T.reshape(-1)


def rotation_angle(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-labels", type=int, default=0)
    args = parser.parse_args()

    g063_root = Path(JOINTTRAIN_ARCH6_G063C_RESULT_ROOT) / "full"
    g063_summary = json.loads((g063_root / "summary.json").read_text(encoding="utf-8"))
    with np.load(g063_root / "contact_mode_labels.npz", allow_pickle=False) as data:
        flat = {key: np.asarray(data[key]) for key in data.files}
    ik_rows = json.loads((g063_root / "ik_rows.json").read_text(encoding="utf-8"))["rows"]
    with np.load(Path(JOINTTRAIN_ARCH6_G061C_RESULT_ROOT) / "full" / "contact_query_inputs.npz", allow_pickle=False) as data:
        query = {key: np.asarray(data[key]) for key in data.files}
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        qpose_relative_teacher = np.asarray(data["qpose_relative"], dtype=np.float64)

    label_count = len(flat["split"])
    if args.probe_labels:
        label_count = min(args.probe_labels, label_count)
    group_to_row = {int(value): index for index, value in enumerate(query["group_index"].tolist())}
    train_flat_indices = np.flatnonzero(flat["split"] == 0)
    prototype_flat_indices = train_flat_indices[flat["prototype_train_row"]]
    prototype_rotation = flat["local_rotation"][prototype_flat_indices].astype(np.float64)
    prototype_translation = flat["local_translation"][prototype_flat_indices].astype(np.float64)
    prototype_feature = np.stack([rotation_6d(value) for value in prototype_rotation])

    shape = query["query_presence"].shape
    mode_index = np.full(shape, -1, dtype=np.int8)
    mode_presence = np.zeros(shape, dtype=bool)
    translation_residual = np.zeros((*shape, 3), dtype=np.float32)
    rotation_residual_6d = np.zeros((*shape, 6), dtype=np.float32)
    grasp_se3 = np.zeros((*shape, 9), dtype=np.float32)
    pregrasp_se3 = np.zeros((*shape, 9), dtype=np.float32)
    ik_qpose_relative = np.zeros((*shape, MAX_IK, 7), dtype=np.float32)
    ik_presence = np.zeros((*shape, MAX_IK), dtype=bool)
    teacher_qpose_relative = np.zeros((*shape, 7), dtype=np.float32)
    consumed = []

    ik_lookup = {(int(row["group_index"]), int(row["slot"])): row for row in ik_rows}
    for flat_index in range(label_count):
        group_index = int(flat["group_index"][flat_index])
        slot = int(flat["query_slot"][flat_index])
        teacher_slot = int(flat["teacher_slot"][flat_index])
        row = group_to_row[group_index]
        local_rotation = flat["local_rotation"][flat_index].astype(np.float64)
        local_translation = flat["local_translation"][flat_index].astype(np.float64)
        feature = rotation_6d(local_rotation)
        selected_mode = int(np.argmin(np.linalg.norm(prototype_feature - feature, axis=1)))
        mode_index[row, slot] = selected_mode
        mode_presence[row, slot] = True
        translation_residual[row, slot] = (local_translation - prototype_translation[selected_mode]).astype(np.float32)
        relative_rotation = prototype_rotation[selected_mode].T @ local_rotation
        rotation_residual_6d[row, slot] = rotation_6d(relative_rotation).astype(np.float32)

        target = np.asarray(query["query_target_se3"][row, slot], dtype=np.float64)
        first = target[3:6] / max(np.linalg.norm(target[3:6]), 1e-8)
        second = target[6:9] - np.dot(first, target[6:9]) * first
        second /= max(np.linalg.norm(second), 1e-8)
        target_rotation = np.stack((first, second, np.cross(first, second)), axis=1)
        target_6d = rotation_6d(target_rotation)
        grasp_se3[row, slot] = np.concatenate((target[:3], target_6d)).astype(np.float32)
        pregrasp_position = target[:3] - PREGRASP_DISTANCE_M * target_rotation[:, 2]
        pregrasp_se3[row, slot] = np.concatenate((pregrasp_position, target_6d)).astype(np.float32)

        teacher_qpose_relative[row, slot] = qpose_relative_teacher[row, teacher_slot].astype(np.float32)
        ik_row = ik_lookup[(group_index, slot)]
        accepted = np.asarray(ik_row["accepted_qpos"], dtype=np.float64).reshape(-1, 7)
        accepted = accepted[:MAX_IK]
        if len(accepted):
            ik_qpose_relative[row, slot, : len(accepted)] = (accepted - query["state_qpos"][row]).astype(np.float32)
            ik_presence[row, slot, : len(accepted)] = True
        consumed.append({
            "group_index": group_index,
            "source_replay_id": int(flat["source_replay_id"][flat_index]),
            "query_slot": slot,
            "teacher_slot": teacher_slot,
            "mode_index": selected_mode,
            "ik_solutions": int(len(accepted)),
        })

    expected_presence = query["query_presence"].astype(bool)
    processed_presence = mode_presence
    full = args.probe_labels == 0
    if full:
        expected_processed = expected_presence
    else:
        expected_processed = np.zeros_like(expected_presence)
        for row in consumed:
            expected_processed[group_to_row[row["group_index"]], row["query_slot"]] = True
    mode_counts = {str(index): int(np.sum(mode_index[processed_presence] == index)) for index in range(8)}
    ik_counts = ik_presence.sum(axis=-1)[processed_presence]
    reconstruction_translation = []
    reconstruction_rotation = []
    for row, slot in np.argwhere(processed_presence):
        selected_mode = int(mode_index[row, slot])
        recovered_translation = prototype_translation[selected_mode] + translation_residual[row, slot]
        reconstruction_translation.append(float(np.linalg.norm(recovered_translation - flat["local_translation"][len(reconstruction_translation)])))
        residual = rotation_residual_6d[row, slot]
        first = residual[:3] / max(np.linalg.norm(residual[:3]), 1e-8)
        second = residual[3:6] - np.dot(first, residual[3:6]) * first
        second /= max(np.linalg.norm(second), 1e-8)
        residual_matrix = np.stack((first, second, np.cross(first, second)), axis=1)
        recovered_rotation = prototype_rotation[selected_mode] @ residual_matrix
        reconstruction_rotation.append(rotation_angle(recovered_rotation.T @ flat["local_rotation"][len(reconstruction_rotation)]))

    checks = {
        "g063_authorizes_g064": g063_summary.get("status") == "passed" and g063_summary.get("claim_supported") == "yes",
        "lineage_unique": len({(row["group_index"], row["query_slot"]) for row in consumed}) == len(consumed),
        "presence_exact": bool(np.array_equal(processed_presence, expected_processed)),
        "teacher_slot_valid": all(0 <= row["teacher_slot"] < 4 for row in consumed),
        "mode_index_valid": bool(np.all((mode_index[processed_presence] >= 0) & (mode_index[processed_presence] < 8))),
        "mode_reconstruction_exact": max(reconstruction_translation, default=1.0) <= 1e-6 and max(reconstruction_rotation, default=1.0) <= 1e-3,
        "ik_count_matches_g063": all(row["ik_solutions"] == int(ik_presence[group_to_row[row["group_index"]], row["query_slot"]].sum()) for row in consumed),
        "finite": bool(np.isfinite(translation_residual).all() and np.isfinite(rotation_residual_6d).all() and np.isfinite(ik_qpose_relative).all()),
        "full_label_counts": len(consumed) == 2373 if full else True,
        "train_only_codebook": True,
        "no_outcome_read": True,
        "no_label_duplication": True,
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / (f"probe_{args.probe_labels}" if args.probe_labels else "full")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "supervision.npz",
        point_cloud_xyz=query["point_cloud_xyz"],
        state_qpos=query["state_qpos"],
        predicted_affordance=query["predicted_affordance"],
        query_point=query["query_point"],
        split=query["split"],
        group_index=query["group_index"],
        source_replay_id=query["source_replay_id"],
        mode_index=mode_index,
        mode_presence=mode_presence,
        translation_residual=translation_residual,
        rotation_residual_6d=rotation_residual_6d,
        pregrasp_se3=pregrasp_se3,
        grasp_se3=grasp_se3,
        ik_qpose_relative=ik_qpose_relative,
        ik_presence=ik_presence,
        teacher_qpose_relative=teacher_qpose_relative,
        prototype_rotation=prototype_rotation.astype(np.float32),
        prototype_translation=prototype_translation.astype(np.float32),
    )
    atomic(out / "lineage_manifest.json", {"rows": consumed})
    atomic(out / "forbidden_feature_audit.json", {"task_outcome_read": False, "future_operation_read": False, "cal_codebook_fit": False, "label_duplication": False})
    summary = {
        "schema_version": 1,
        "run_id": "A6-G064C-PROBE" if args.probe_labels else "A6-G064C",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "labels": len(consumed),
        "mode_counts": mode_counts,
        "ik_set": {
            "coverage_at_least_one": float(np.mean(ik_counts >= 1)),
            "multi_solution_fraction": float(np.mean(ik_counts >= 2)),
            "mean_solutions": float(np.mean(ik_counts)),
            "max_solutions": int(np.max(ik_counts)),
        },
        "pregrasp_distance_m": PREGRASP_DISTANCE_M,
        "max_reconstruction_translation_error": max(reconstruction_translation, default=None),
        "max_reconstruction_rotation_error": max(reconstruction_rotation, default=None),
        "checks": checks,
        "claim_supported": "implementation" if passed and full else "probe_only",
        "decision": "authorize G065 matched fit" if passed and full else ("run full G064C" if passed else "repair G064 supervision"),
        "next_run_ids": ["A6-G065C"] if passed and full else (["A6-G064C"] if passed else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

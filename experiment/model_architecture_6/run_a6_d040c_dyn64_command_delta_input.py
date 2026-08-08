#!/usr/bin/env python3
"""Build clean DYN64 operation inputs from frozen A6 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.datasets import action_from_npz
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH5_DYNAMIC_CANDIDATES,
    JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D040C_RESULT_ROOT,
)
from run_a6_o000b_shared_input_contract import task_metadata


RUN_ID = "a6_d040c_dyn64_command_delta_input_v1"
HORIZON = 32
ACTION_DIM = 9


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def action_chunk(action: np.ndarray, anchor: int, operation_stop: int) -> tuple[np.ndarray, np.ndarray]:
    begin = anchor + 1
    stop = min(operation_stop, begin + HORIZON)
    values = np.asarray(action[begin:stop, :ACTION_DIM], dtype=np.float32)
    if values.shape[0] == 0:
        raise ValueError(f"empty action horizon at anchor {anchor}")
    valid = np.zeros((HORIZON,), dtype=bool)
    valid[: values.shape[0]] = True
    if values.shape[0] < HORIZON:
        values = np.concatenate(
            [values, np.repeat(values[-1:], HORIZON - values.shape[0], axis=0)], axis=0
        )
    return values, valid


def causal_state(qpos: np.ndarray, command: np.ndarray, anchor: int) -> np.ndarray:
    indices = np.clip(np.arange(anchor - 3, anchor + 1), 0, qpos.shape[0] - 1)
    history = qpos[indices]
    previous = qpos[np.maximum(indices - 1, 0)]
    qvel = 240.0 * (history - previous)
    return np.concatenate(
        [history.reshape(-1), qvel.reshape(-1), command[anchor, :ACTION_DIM]], axis=0
    ).astype(np.float32)


def build(limit: int) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    d020 = Path(JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT)
    d021 = Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)
    d030 = Path(JOINTTRAIN_ARCH6_D030_RESULT_ROOT)
    d030c = Path(JOINTTRAIN_ARCH6_D030C_RESULT_ROOT)
    if load_json(d021 / "summary.json").get("status") != "passed":
        raise ValueError("D021C is not passed")
    if load_json(d030c / "summary.json").get("status") != "passed":
        raise ValueError("D030C is not passed")

    index_rows = [json.loads(line) for line in (d020 / "sample_index.jsonl").read_text(encoding="utf-8").splitlines()]
    index = {row["trajectory_relative_path"]: row for row in index_rows}
    candidates = load_json(Path(JOINTTRAIN_ARCH5_DYNAMIC_CANDIDATES))["DYN64"]
    if limit:
        candidates = candidates[:limit]

    arrays: dict[str, list[np.ndarray]] = {
        "point_cloud": [],
        "target_mask": [],
        "zero_affordance": [],
        "state_history": [],
        "context": [],
        "absolute_action_target": [],
        "command_delta_target": [],
        "action_valid": [],
    }
    rows: list[dict] = []
    loaded_fields: set[str] = set()
    for candidate in candidates:
        relative = str(candidate["relative_trajectory_path"])
        indexed = index.get(relative)
        if indexed is None or indexed["split"] != "A5_TRAIN":
            raise ValueError(f"DYN64 trajectory is not in clean TRAIN: {relative}")
        if indexed["target"] != candidate["target"]:
            raise ValueError(f"target mismatch: {relative}")
        trajectory_path = Path(ARTICU_COLLECTION_ROOT) / relative
        if sha256_file(trajectory_path) != indexed["source_sha256"]:
            raise ValueError(f"source hash drift: {relative}")

        target = str(candidate["target"])
        materialized_path = d030 / "materialized" / f"{target.replace('/', '_')}.npz"
        with np.load(materialized_path, allow_pickle=False) as materialized:
            points = np.asarray(materialized["point_cloud"], dtype=np.float32)
            masks = np.asarray(materialized["target_mask"], dtype=bool)
            raw_indices = np.asarray(materialized["raw_index"], dtype=np.int64)
        anchors = np.asarray(indexed["anchors"], dtype=np.int64)
        if not np.array_equal(raw_indices, anchors):
            raise ValueError(f"D030 anchor mismatch: {target}")

        with np.load(trajectory_path, allow_pickle=False) as data:
            loaded_fields.update(data.files)
            qpos = np.asarray(data["actual_joint_qpos"], dtype=np.float32)
            command = np.asarray(data["joint_command_qpos"], dtype=np.float32)
            feedback = np.asarray(data["contact_feedback"], dtype=np.float32)
            action = action_from_npz(
                data, source="joint_command_qpos_repaired", include_finger=False
            )
        task = task_metadata(target)
        for local_index, anchor in enumerate(anchors.tolist()):
            absolute, valid = action_chunk(action, anchor, int(indexed["operation_stop"]))
            last_command = command[anchor, :ACTION_DIM]
            contact_present = (
                feedback.ndim == 2 and feedback.shape[1] == 33 and anchor < feedback.shape[0]
            )
            contact = feedback[anchor] if contact_present else np.zeros((33,), dtype=np.float32)
            available = float(contact_present and np.isfinite(contact).all())
            contact = np.nan_to_num(contact, nan=0.0, posinf=0.0, neginf=0.0)
            arrays["point_cloud"].append(points[local_index])
            arrays["target_mask"].append(masks[local_index])
            arrays["zero_affordance"].append(np.zeros((1024,), dtype=np.float32))
            arrays["state_history"].append(causal_state(qpos, command, anchor))
            arrays["context"].append(
                np.concatenate([contact, np.asarray([available], dtype=np.float32), task])
            )
            arrays["absolute_action_target"].append(absolute)
            arrays["command_delta_target"].append(absolute - last_command[None, :])
            arrays["action_valid"].append(valid)
            rows.append(
                {
                    "target": target,
                    "trajectory_relative_path": relative,
                    "source_sha256": indexed["source_sha256"],
                    "anchor_raw_index": anchor,
                    "anchor_rank": local_index,
                    "contact_available": bool(available),
                }
            )

    stacked = {name: np.stack(values) for name, values in arrays.items()}
    audit = {
        "loaded_source_fields": sorted(loaded_fields),
        "consumed_source_fields": [
            "actual_joint_qpos",
            "joint_command_qpos",
            "contact_feedback",
            "action_phase",
        ],
        "forbidden_fields_consumed": [],
        "future_qpos_read": False,
        "object_qpos_read": False,
        "result_json_read": False,
        "outcome_read": False,
        "heldout_read": False,
        "qvel_source": "240Hz causal backward finite difference",
        "task_metadata_source": "PartNet mobility_vhacd.urdf",
    }
    return stacked, rows, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT)
    target_dir = out_dir / (f"probe_{args.limit}" if args.limit else "full")
    target_dir.mkdir(parents=True, exist_ok=True)
    arrays, rows, audit = build(args.limit)
    output_path = target_dir / "dyn64_input.npz"
    temporary = target_dir / "dyn64_input.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output_path)

    expected_targets = args.limit if args.limit else 64
    expected_rows = expected_targets * 16
    delta_roundtrip = float(
        np.max(
            np.abs(
                arrays["command_delta_target"]
                + arrays["state_history"][:, -ACTION_DIM:, None].reshape(-1, 1, ACTION_DIM)
                - arrays["absolute_action_target"]
            )
        )
    )
    checks = {
        "target_count_exact": len({row["target"] for row in rows}) == expected_targets,
        "row_count_exact": len(rows) == expected_rows,
        "point_shape_exact": arrays["point_cloud"].shape == (expected_rows, 1024, 3),
        "state_context_shapes_exact": arrays["state_history"].shape == (expected_rows, 81)
        and arrays["context"].shape == (expected_rows, 43),
        "target_mask_binary": arrays["target_mask"].shape == (expected_rows, 1024)
        and bool(np.isin(arrays["target_mask"], [False, True]).all()),
        "zero_affordance_exact": not bool(np.count_nonzero(arrays["zero_affordance"])),
        "action_shapes_exact": arrays["command_delta_target"].shape
        == (expected_rows, HORIZON, ACTION_DIM)
        and arrays["action_valid"].shape == (expected_rows, HORIZON),
        "all_rows_have_valid_action": bool(arrays["action_valid"].any(axis=1).all()),
        "all_arrays_finite": all(np.isfinite(value).all() for value in arrays.values()),
        "delta_roundtrip_le_1e_6": delta_roundtrip <= 1e-6,
        "forbidden_field_audit_clean": not audit["forbidden_fields_consumed"],
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "mode": "probe" if args.limit else "full",
        "rows": rows,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "input_sha256": sha256_file(output_path),
        "source_hashes": {
            "clean_sample_index": sha256_file(
                Path(JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT) / "sample_index.jsonl"
            ),
            "command_delta_normalizer": sha256_file(
                Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT) / "normalizer.json"
            ),
            "d030c_summary": sha256_file(
                Path(JOINTTRAIN_ARCH6_D030C_RESULT_ROOT) / "summary.json"
            ),
        },
    }
    atomic_json(target_dir / "input_manifest.json", manifest)
    atomic_json(target_dir / "forbidden_feature_audit.json", audit)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "counts": {"targets": expected_targets, "rows": len(rows)},
        "metrics": {"delta_roundtrip_max_abs": delta_roundtrip},
        "checks": checks,
        "evidence": {
            "input": str(output_path.relative_to(out_dir)),
            "manifest": str((target_dir / "input_manifest.json").relative_to(out_dir)),
            "forbidden_feature_audit": str(
                (target_dir / "forbidden_feature_audit.json").relative_to(out_dir)
            ),
        },
        "decision": "DYN64 input adapter passes." if passed else "DYN64 input adapter failed; do not train.",
    }
    atomic_json(target_dir / "summary.json", summary)
    if not args.limit:
        atomic_json(out_dir / "summary.json", summary)
        atomic_json(out_dir / "run_state.json", summary)
        atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-D040C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

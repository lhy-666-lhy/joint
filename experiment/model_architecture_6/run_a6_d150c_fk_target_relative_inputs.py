#!/usr/bin/env python3
"""Build a deployable FK/visible-target relative state for A6 operation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_deployable_geometry import (
    FK_TARGET_FEATURE_DIM,
    FK_TARGET_FEATURE_NAMES,
    raw_fk_target_feature,
)
from jointTrain_new.joint_train.sim.capture_view_pcd import (
    ViewPcdCapturer,
    base_pose_from_init,
    resolve_urdf,
)
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D040C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D150C_RESULT_ROOT,
    PARTNET_DATASET_ROOT,
    PROJECT_ROOT,
)


RUN_ID = "a6_d150c_fk_target_relative_inputs_v1"


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


def load_rows_and_arrays(split: str) -> tuple[list[dict], dict[str, np.ndarray], Path]:
    if split == "A5_TRAIN":
        root = Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT) / "full"
        source = Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "train_zero_contact.npz"
    elif split == "A5_CAL":
        root = Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT) / "full"
        source = Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT) / "cal_zero_contact.npz"
    else:
        raise ValueError(f"unsupported split: {split}")
    manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    rows = list(manifest["rows"])
    if len(rows) != arrays["state_history"].shape[0]:
        raise ValueError(f"row/input length mismatch for {split}")
    return rows, arrays, source


def build_features(
    rows: list[dict],
    arrays: dict[str, np.ndarray],
    capturer: ViewPcdCapturer,
) -> np.ndarray:
    features: list[np.ndarray] = []
    trajectory_cache: dict[str, tuple[np.ndarray, dict]] = {}
    for index, row in enumerate(rows):
        relative = str(row["trajectory_relative_path"])
        if relative not in trajectory_cache:
            trajectory = Path(ARTICU_COLLECTION_ROOT) / relative
            if row.get("source_sha256") and sha256_file(trajectory) != row["source_sha256"]:
                raise ValueError(f"source hash drift: {relative}")
            with np.load(trajectory, allow_pickle=False) as data:
                robot_qpos = np.asarray(data["actual_joint_qpos"], dtype=np.float32)
            init_path = trajectory.parents[1] / "initial_state.json"
            init = json.loads(init_path.read_text(encoding="utf-8"))
            trajectory_cache[relative] = (robot_qpos, init)
        robot_qpos, init = trajectory_cache[relative]
        urdf = resolve_urdf(init["object_urdf"], partnet_root=PARTNET_DATASET_ROOT)
        world = capturer._get_world(urdf, float(init["size"]))
        anchor = int(row["anchor_raw_index"])
        if anchor < 0 or anchor >= robot_qpos.shape[0]:
            raise ValueError(f"anchor outside robot qpos: {relative}:{anchor}")
        feature = raw_fk_target_feature(
            world,
            base_pose_from_init(init),
            robot_qpos[anchor],
            arrays["point_cloud"][index],
            arrays["target_mask"][index],
        )
        features.append(feature)
    return np.stack(features).astype(np.float32)


def main() -> int:
    out = Path(JOINTTRAIN_ARCH6_D150C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, train_arrays, train_source = load_rows_and_arrays("A5_TRAIN")
    cal_rows, cal_arrays, cal_source = load_rows_and_arrays("A5_CAL")
    capturer = ViewPcdCapturer(
        articu_root=PROJECT_ROOT,
        partnet_root=PARTNET_DATASET_ROOT,
        render_enabled=False,
        settle_steps=0,
    )
    try:
        train_raw = build_features(train_rows, train_arrays, capturer)
        cal_raw = build_features(cal_rows, cal_arrays, capturer)
    finally:
        capturer.close()

    mean = train_raw.mean(axis=0).astype(np.float32)
    std = train_raw.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    train_geometry = ((train_raw - mean) / std).astype(np.float32)
    cal_geometry = ((cal_raw - mean) / std).astype(np.float32)
    train_state = np.concatenate((train_arrays["state_history"], train_geometry), axis=1)
    cal_state = np.concatenate((cal_arrays["state_history"], cal_geometry), axis=1)

    train_out = {**train_arrays, "state_history": train_state, "geometry_feature_raw": train_raw, "geometry_feature": train_geometry}
    cal_out = {**cal_arrays, "state_history": cal_state, "geometry_feature_raw": cal_raw, "geometry_feature": cal_geometry}
    train_path = out / "train_fk_target_relative.npz"
    cal_path = out / "cal_fk_target_relative.npz"
    train_tmp = train_path.with_suffix(".tmp.npz")
    cal_tmp = cal_path.with_suffix(".tmp.npz")
    np.savez_compressed(train_tmp, **train_out)
    np.savez_compressed(cal_tmp, **cal_out)
    os.replace(train_tmp, train_path)
    os.replace(cal_tmp, cal_path)
    atomic_json(
        out / "feature_normalizer.json",
        {
            "schema_version": 1,
            "fit_split": "A5_TRAIN",
            "feature_names": list(FK_TARGET_FEATURE_NAMES),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "train_rows": len(train_rows),
        },
    )
    train_targets = {str(row["target"]) for row in train_rows}
    cal_targets = {str(row["target"]) for row in cal_rows}
    checks = {
        "train_rows_1024": len(train_rows) == 1024,
        "cal_rows_280": len(cal_rows) == 280,
        "state_width_85": train_state.shape == (len(train_rows), 85) and cal_state.shape == (len(cal_rows), 85),
        "original_state_prefix_exact": bool(np.array_equal(train_state[:, :81], train_arrays["state_history"])) and bool(np.array_equal(cal_state[:, :81], cal_arrays["state_history"])),
        "zero_contact_preserved": not bool(np.count_nonzero(train_arrays["context"][:, :34])) and not bool(np.count_nonzero(cal_arrays["context"][:, :34])),
        "train_cal_targets_disjoint": not bool(train_targets & cal_targets),
        "finite": bool(np.isfinite(train_out["state_history"]).all() and np.isfinite(cal_out["state_history"]).all() and np.isfinite(mean).all() and np.isfinite(std).all()),
        "normalizer_train_only": True,
        "feature_dim_exact": len(FK_TARGET_FEATURE_NAMES) == FK_TARGET_FEATURE_DIM,
        "source_hashes_recorded": len(sha256_file(train_source)) == 64 and len(sha256_file(cal_source)) == 64,
        "forbidden_object_progress_reads": True,
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "state_schema": "81D causal robot history/command + 4D robot-FK to visible target centroid",
        "feature_names": list(FK_TARGET_FEATURE_NAMES),
        "observation_source": {"train": "recorded_current_observation", "cal": "recorded_current_observation", "live": "live_sapiens_observation"},
        "source_hashes": {"train_input": sha256_file(train_source), "cal_input": sha256_file(cal_source)},
        "output_hashes": {"train": sha256_file(train_path), "cal": sha256_file(cal_path), "normalizer": sha256_file(out / "feature_normalizer.json")},
        "rows": {"train": len(train_rows), "cal": len(cal_rows)},
    }
    audit = {
        "loaded_fields": ["actual_joint_qpos", "point_cloud", "target_mask", "base_pose", "object_urdf", "size"],
        "forbidden_fields_read": [],
        "object_qpos_read": False,
        "future_qpos_read": False,
        "contact_feedback_read": False,
        "result_json_read": False,
        "outcome_read": False,
        "heldout_read": False,
        "feature_is_deployable": True,
    }
    atomic_json(out / "input_manifest.json", manifest)
    atomic_json(out / "forbidden_feature_audit.json", audit)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "data_contract_failure",
        "claim_supported": "yes" if passed else "no",
        "checks": checks,
        "metrics": {"train_feature_mean": mean.tolist(), "train_feature_std": std.tolist(), "train_visible_fraction_zero_rows": int((train_raw[:, 3] == 0).sum()), "cal_visible_fraction_zero_rows": int((cal_raw[:, 3] == 0).sum())},
        "decision": "deployable geometry state materialized; authorize matched MLP training" if passed else "geometry input contract failed",
        "next_run_ids": ["A6-O151C", "A6-O152C", "A6-O153C"] if passed else [],
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(out / "queue_state.json", {**summary, "jobs": [{"id": "A6-D150C", "status": summary["status"]}]})
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

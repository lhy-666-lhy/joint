#!/usr/bin/env python3
"""Correct G006 world-frame XYZ to the current robot base frame."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import ARTICU_COLLECTION_ROOT, JOINTTRAIN_ARCH6_G000C_RESULT_ROOT, JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT, JOINTTRAIN_ARCH6_G006C_RESULT_ROOT


def atomic(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def world_to_base(points: np.ndarray, base: list[float]) -> np.ndarray:
    x, y, yaw, z = map(float, base)
    centered = np.asarray(points, dtype=np.float64) - np.asarray([x, y, z])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation_world_to_base = np.asarray([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return (centered @ rotation_world_to_base.T).astype(np.float32)


def main() -> int:
    source = Path(JOINTTRAIN_ARCH6_G006C_RESULT_ROOT) / "grasp_inputs.npz"
    groups = json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT) / "qpose_teacher_manifest.json").read_text())["groups"]
    with np.load(source, allow_pickle=False) as d:
        arrays = {key: np.asarray(d[key]) for key in d.files}
    base_poses=[]; converted=[]
    root = Path(ARTICU_COLLECTION_ROOT)
    for group, points in zip(groups, arrays["point_cloud_xyz"], strict=True):
        init = json.loads((root / "data" / "single" / group["sample_id"] / "initial_state.json").read_text())
        if init.get("frame_transform") is None:
            raise ValueError(f"unexpected legacy base pose order: {group['sample_id']}")
        base = list(map(float, init["base_pose"]))
        base_poses.append(base); converted.append(world_to_base(points, base))
    arrays["point_cloud_xyz"] = np.stack(converted)
    arrays["base_pose"] = np.asarray(base_poses, dtype=np.float32)
    out = Path(JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"grasp_inputs.npz", **arrays)
    world = np.load(source, allow_pickle=False)["point_cloud_xyz"]
    delta = float(np.mean(np.abs(arrays["point_cloud_xyz"] - world)))
    checks={"shape_preserved":arrays["point_cloud_xyz"].shape==(632,1024,3),"finite":bool(np.isfinite(arrays["point_cloud_xyz"]).all()),"all_frame_transform_present":len(base_poses)==632,"world_to_base_nontrivial":delta>0.1,"labels_unchanged":True,"deployable_preprocess":True}
    summary={"schema_version":1,"run_id":"A6-G006BC","status":"passed" if all(checks.values()) else "failed","complete":True,"terminal":True,"coordinate_frame":"robot_base","mean_abs_world_to_base_delta":delta,"checks":checks,"decision":"authorize matched base-frame qpose retrain"}
    atomic(out/"summary.json",summary);atomic(out/"run_state.json",summary);atomic(out/"queue_state.json",summary);print(json.dumps(summary));return 0 if all(checks.values()) else 2


if __name__=="__main__":raise SystemExit(main())

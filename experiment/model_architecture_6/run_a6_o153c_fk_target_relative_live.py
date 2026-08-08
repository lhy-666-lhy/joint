#!/usr/bin/env python3
"""Run the exact A6 live8 protocol with the geometry-state MLP arm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o126c_zero_contact_live_probe as live
from a6_deployable_geometry import (
    FK_TARGET_FEATURE_NAMES,
    normalize_fk_target_feature,
    raw_fk_target_feature,
)
from a6_operation_models import OperationMLPAbsolute
from jointTrain_new.joint_train.sim.capture_view_pcd import base_pose_from_init
from path_config import (
    JOINTTRAIN_ARCH6_D150C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O151C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O153C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


def geometry_state(world, init, cloud, mask, state):
    raw = raw_fk_target_feature(
        world,
        base_pose_from_init(init),
        np.asarray(world.robot.get_qpos(), dtype=np.float32),
        cloud,
        mask,
    )
    normalizer = json.loads(
        (Path(JOINTTRAIN_ARCH6_D150C_RESULT_ROOT) / "feature_normalizer.json").read_text(
            encoding="utf-8"
        )
    )
    feature = normalize_fk_target_feature(raw, normalizer["mean"], normalizer["std"])
    return np.concatenate((state, feature), axis=0).astype(np.float32)


def main() -> int:
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "fk_target_mlp": (
            lambda: OperationMLPAbsolute(state_dim=85),
            Path(JOINTTRAIN_ARCH6_O151C_RESULT_ROOT) / "last.pt",
        ),
        "repeat_last": (None, None),
    }
    live.STATE_APPENDERS = {"fk_target_mlp": geometry_state}
    live.MODEL_INPUT_FEATURES = {
        "baseline_mlp": [],
        "fk_target_mlp": list(FK_TARGET_FEATURE_NAMES),
        "repeat_last": [],
    }
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = str(JOINTTRAIN_ARCH6_O153C_RESULT_ROOT)
    code = live.main()
    # The probe writes the selected target directory; only enrich its terminal record.
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--execute-prefix", type=int, required=True)
    parsed, _ = parser.parse_known_args()
    out = Path(JOINTTRAIN_ARCH6_O153C_RESULT_ROOT) / f"probe_calls_{parsed.max_calls}_target_{parsed.target_index}"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary["run_id"] = "a6_o153c_fk_target_relative_live8_v1"
    summary["scientific_scope"] = "A5_CAL source-horizon live8 baseline versus FK/visible-target MLP"
    summary["mode"] = "K8"
    summary["checkpoint_scope"] = {
        "baseline_mlp": "A6-O127C TRAIN1024 frozen",
        "fk_target_mlp": "A6-O151C TRAIN1024 frozen",
        "repeat_last": "control",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

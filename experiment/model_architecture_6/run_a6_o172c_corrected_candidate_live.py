#!/usr/bin/env python3
"""Re-evaluate valid A6 MLP checkpoints under the corrected live protocol."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o126c_zero_contact_live_probe as live
from a6_deployable_geometry import FK_TARGET_FEATURE_NAMES
from a6_operation_models import OperationMLPAbsolute, OperationMLPGeometryResidual
from path_config import (
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O137C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O143C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O144C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O154C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O172C_RESULT_ROOT,
)
from run_a6_o153c_fk_target_relative_live import geometry_state


def main() -> int:
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "additive_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O137C_RESULT_ROOT) / "last.pt",
        ),
        "perturb_1x": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O143C_RESULT_ROOT) / "last.pt",
        ),
        "perturb_3x": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O144C_RESULT_ROOT) / "last.pt",
        ),
        "geometry_residual": (
            OperationMLPGeometryResidual,
            Path(JOINTTRAIN_ARCH6_O154C_RESULT_ROOT) / "last.pt",
        ),
        "repeat_last": (None, None),
    }
    live.STATE_APPENDERS = {"geometry_residual": geometry_state}
    live.MODEL_INPUT_FEATURES = {
        arm: list(FK_TARGET_FEATURE_NAMES) if arm == "geometry_residual" else []
        for arm in live.ARMS
    }
    live.RUN_ID = "a6_o172c_corrected_candidate_live8_v1"
    live.SCIENTIFIC_SCOPE = (
        "corrected live8 screen of previously trained valid A6 MLP candidates"
    )
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = JOINTTRAIN_ARCH6_O172C_RESULT_ROOT
    return live.main()


if __name__ == "__main__":
    raise SystemExit(main())


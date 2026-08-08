#!/usr/bin/env python3
"""Run fresh-world live8 for the averaged recovery residual."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o126c_zero_contact_live_probe as live
from a6_operation_models import OperationMLPAbsolute, OperationMLPRecoveryResidual
from path_config import (
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O185C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O189C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O197C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O199C_RESULT_ROOT,
)


def main() -> int:
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "residual_seed1": (
            OperationMLPRecoveryResidual,
            Path(JOINTTRAIN_ARCH6_O185C_RESULT_ROOT) / "last.pt",
        ),
        "residual_seed2": (
            OperationMLPRecoveryResidual,
            Path(JOINTTRAIN_ARCH6_O189C_RESULT_ROOT) / "last.pt",
        ),
        "residual_average": (
            OperationMLPRecoveryResidual,
            Path(JOINTTRAIN_ARCH6_O197C_RESULT_ROOT) / "last.pt",
        ),
    }
    live.STATE_APPENDERS = {}
    live.MODEL_INPUT_FEATURES = {arm: [] for arm in live.ARMS}
    live.RUN_ID = "a6_o199c_recovery_residual_weight_average_live8_v1"
    live.SCIENTIFIC_SCOPE = "fresh-world recovery residual weight-average comparison"
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = JOINTTRAIN_ARCH6_O199C_RESULT_ROOT
    return live.main()


if __name__ == "__main__":
    raise SystemExit(main())

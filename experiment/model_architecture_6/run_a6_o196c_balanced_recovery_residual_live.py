#!/usr/bin/env python3
"""Run fresh-world live8 for equal-exposure residual sampling."""

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
    JOINTTRAIN_ARCH6_O194C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O196C_RESULT_ROOT,
)


def main() -> int:
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "random_seed1": (
            OperationMLPRecoveryResidual,
            Path(JOINTTRAIN_ARCH6_O185C_RESULT_ROOT) / "last.pt",
        ),
        "balanced_seed1": (
            OperationMLPRecoveryResidual,
            Path(JOINTTRAIN_ARCH6_O194C_RESULT_ROOT) / "seed1" / "last.pt",
        ),
        "balanced_seed2": (
            OperationMLPRecoveryResidual,
            Path(JOINTTRAIN_ARCH6_O194C_RESULT_ROOT) / "seed2" / "last.pt",
        ),
    }
    live.STATE_APPENDERS = {}
    live.MODEL_INPUT_FEATURES = {arm: [] for arm in live.ARMS}
    live.RUN_ID = "a6_o196c_balanced_recovery_residual_live8_v1"
    live.SCIENTIFIC_SCOPE = (
        "fresh-world equal-exposure recovery residual stability comparison"
    )
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = JOINTTRAIN_ARCH6_O196C_RESULT_ROOT
    return live.main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the averaged recovery residual checkpoint on frozen A5_CAL."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o186c_recovery_residual_cal as evaluate
from a6_operation_models import OperationMLPAbsolute, OperationMLPRecoveryResidual
from path_config import (
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O185C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O189C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O197C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O198C_RESULT_ROOT,
)


def main() -> int:
    evaluate.RUN_ID = "a6_o198c_recovery_residual_weight_average_cal_v1"
    evaluate.QUEUE_RUN_ID = "A6-O198C"
    evaluate.NEXT_RUN_IDS = ["A6-O199C"]
    evaluate.JOINTTRAIN_ARCH6_O186C_RESULT_ROOT = JOINTTRAIN_ARCH6_O198C_RESULT_ROOT
    evaluate.MODEL_SPECS = {
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
    evaluate.PAIRWISE_COMPARISONS = (
        ("residual_average", "baseline_mlp"),
        ("residual_average", "residual_seed1"),
        ("residual_average", "residual_seed2"),
        ("residual_seed2", "residual_seed1"),
    )
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main())

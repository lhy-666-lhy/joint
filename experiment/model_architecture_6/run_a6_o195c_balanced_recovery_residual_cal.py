#!/usr/bin/env python3
"""Audit balanced recovery residual checkpoints on frozen A5_CAL."""

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
    JOINTTRAIN_ARCH6_O194C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O195C_RESULT_ROOT,
)


def main() -> int:
    evaluate.RUN_ID = "a6_o195c_balanced_recovery_residual_cal_v1"
    evaluate.QUEUE_RUN_ID = "A6-O195C"
    evaluate.NEXT_RUN_IDS = ["A6-O196C"]
    evaluate.JOINTTRAIN_ARCH6_O186C_RESULT_ROOT = JOINTTRAIN_ARCH6_O195C_RESULT_ROOT
    evaluate.MODEL_SPECS = {
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
    evaluate.PAIRWISE_COMPARISONS = (
        ("balanced_seed1", "baseline_mlp"),
        ("balanced_seed2", "baseline_mlp"),
        ("balanced_seed1", "random_seed1"),
        ("balanced_seed2", "random_seed1"),
        ("balanced_seed2", "balanced_seed1"),
    )
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main())

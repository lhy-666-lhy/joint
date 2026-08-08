#!/usr/bin/env python3
"""Run the frozen O185C residual recipe with a second sampling seed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o185c_recovery_residual_train as train
from path_config import JOINTTRAIN_ARCH6_O189C_RESULT_ROOT


def main() -> int:
    train.SEED = 20260806
    train.RUN_ID = "a6_o189c_recovery_residual_seed2_train_v1"
    train.QUEUE_RUN_ID = "A6-O189C"
    train.NEXT_RUN_IDS = ["A6-O190C", "A6-O191C"]
    train.JOINTTRAIN_ARCH6_O185C_RESULT_ROOT = JOINTTRAIN_ARCH6_O189C_RESULT_ROOT
    return train.main()


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Train one equal-exposure recovery residual seed for A6-O194C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o185c_recovery_residual_train as train
from path_config import JOINTTRAIN_ARCH6_O194C_RESULT_ROOT


SEEDS = {1: 20260807, 2: 20260808}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-index", type=int, choices=sorted(SEEDS), required=True)
    args = parser.parse_args()
    seed = SEEDS[args.seed_index]
    train.RUN_ID = f"a6_o194c_balanced_recovery_residual_seed{args.seed_index}_train_v1"
    train.QUEUE_RUN_ID = f"A6-O194C-seed{args.seed_index}"
    train.NEXT_RUN_IDS = ["A6-O195C"]
    train.OUTPUT_ROOT = Path(JOINTTRAIN_ARCH6_O194C_RESULT_ROOT) / f"seed{args.seed_index}"
    train.SEED = seed
    train.SAMPLING_MODE = "epoch_balanced_without_replacement"
    train.SCIENTIFIC_SCOPE = (
        "equal-exposure prefix and recovery sampling over frozen O127C residual"
    )
    return train.main()


if __name__ == "__main__":
    raise SystemExit(main())

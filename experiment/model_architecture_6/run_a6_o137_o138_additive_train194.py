#!/usr/bin/env python3
"""Train MLP/PAR on the additive D044C target-coverage input."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o132_o133_zero_contact_train194 as trainer
from path_config import (
    JOINTTRAIN_ARCH6_D044C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O137C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O138C_RESULT_ROOT,
)


def main() -> int:
    trainer.DATA_ROOT = JOINTTRAIN_ARCH6_D044C_RESULT_ROOT
    trainer.INPUT_FILENAME = "train194_additive_input.npz"
    trainer.INPUT_SCHEMA = "D044C-additive-zero-contact"
    trainer.TRAINING_TARGETS = 194
    trainer.SCIENTIFIC_SCOPE = (
        "A5_TRAIN D042C-exact-prefix plus 130 new targets, 2064 rows"
    )
    trainer.ROW_CHECK_NAME = "train_rows_2064"
    trainer.EXPECTED_ROWS = 2064
    trainer.ROOTS = {
        "mlp": JOINTTRAIN_ARCH6_O137C_RESULT_ROOT,
        "parallel": JOINTTRAIN_ARCH6_O138C_RESULT_ROOT,
    }
    trainer.RUN_IDS = {
        "mlp": "a6_o137c_mlp_additive_train194_v1",
        "parallel": "a6_o138c_parallel_additive_train194_v1",
    }
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())

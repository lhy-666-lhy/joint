#!/usr/bin/env python3
"""Run the frozen CAL evaluator for the corrected geometry residual."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o152c_fk_target_relative_cal as evaluator
from a6_operation_models import OperationMLPGeometryResidual
from path_config import (
    JOINTTRAIN_ARCH6_O154C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O155C_RESULT_ROOT,
)


def main() -> int:
    evaluator.RUN_ID = "a6_o155c_geometry_residual_cal_v1"
    evaluator.OUTPUT_ROOT = JOINTTRAIN_ARCH6_O155C_RESULT_ROOT
    evaluator.GEOMETRY_CHECKPOINT_ROOT = JOINTTRAIN_ARCH6_O154C_RESULT_ROOT
    evaluator.GEOMETRY_FACTORY = lambda state_dim: OperationMLPGeometryResidual()
    evaluator.GEOMETRY_STATE_DIM = 85
    evaluator.NEXT_RUN_ID = "A6-O156C"
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())

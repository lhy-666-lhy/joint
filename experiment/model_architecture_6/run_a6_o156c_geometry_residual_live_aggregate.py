#!/usr/bin/env python3
"""Aggregate the corrected A6 geometry-residual live8 run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o153c_fk_target_relative_live_aggregate as aggregate
from path_config import JOINTTRAIN_ARCH6_O156C_RESULT_ROOT


def main() -> int:
    aggregate.RUN_ID = "a6_o156c_geometry_residual_live8_v1"
    aggregate.OUTPUT_ROOT = JOINTTRAIN_ARCH6_O156C_RESULT_ROOT
    aggregate.POSITIVE_NEXT_RUN_ID = "A6-O157C"
    aggregate.NEGATIVE_NEXT_RUN_ID = "A6-O160C"
    return aggregate.main()


if __name__ == "__main__":
    raise SystemExit(main())

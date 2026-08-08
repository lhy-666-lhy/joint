#!/usr/bin/env python3
"""Run corrected live8 for time/progress aligned recovery policies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o126c_zero_contact_live_probe as live
from a6_operation_models import OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O181C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O182C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O184C_RESULT_ROOT,
)


def main() -> int:
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "time_aligned": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O181C_RESULT_ROOT) / "last.pt",
        ),
        "progress_aligned": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O182C_RESULT_ROOT) / "last.pt",
        ),
        "repeat_last": (None, None),
    }
    live.STATE_APPENDERS = {}
    live.MODEL_INPUT_FEATURES = {arm: [] for arm in live.ARMS}
    live.RUN_ID = "a6_o184c_recovery_alignment_live8_v1"
    live.SCIENTIFIC_SCOPE = (
        "corrected live8 baseline versus full-horizon recovery alignments"
    )
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = JOINTTRAIN_ARCH6_O184C_RESULT_ROOT
    return live.main()


if __name__ == "__main__":
    raise SystemExit(main())


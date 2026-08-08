#!/usr/bin/env python3
"""Aggregate the additive TRAIN194 live8 target-coverage experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o136c_train194_live8_aggregate as aggregate
from path_config import (
    JOINTTRAIN_ARCH6_O140C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O141C_RESULT_ROOT,
)


def main() -> int:
    aggregate.JOINTTRAIN_ARCH6_O135C_RESULT_ROOT = JOINTTRAIN_ARCH6_O140C_RESULT_ROOT
    aggregate.JOINTTRAIN_ARCH6_O136C_RESULT_ROOT = JOINTTRAIN_ARCH6_O141C_RESULT_ROOT
    aggregate.RUN_ID = "a6_o141c_additive_coverage_decision_v1"
    aggregate.SCIENTIFIC_SCOPE = (
        "8-target A5_CAL source-horizon live test of additive target coverage"
    )
    aggregate.EFFECT_FIELD = "coverage_effect"
    aggregate.PROGRESS_EFFECT_KEY = "progress_additive_minus_train1024"
    aggregate.CONTACT_EFFECT_KEY = "contact_additive_minus_train1024"
    aggregate.DECISION = (
        "additive target-coverage effect measured; choose next strategy from paired evidence"
    )
    return aggregate.main()


if __name__ == "__main__":
    raise SystemExit(main())

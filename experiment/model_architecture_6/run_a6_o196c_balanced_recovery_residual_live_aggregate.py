#!/usr/bin/env python3
"""Aggregate fresh-world live8 for equal-exposure residual sampling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o193c_fresh_world_fixed_budget_live_aggregate as aggregate
from path_config import JOINTTRAIN_ARCH6_O196C_RESULT_ROOT


def main() -> int:
    aggregate.RUN_ID = "a6_o196c_balanced_recovery_residual_live8_v1"
    aggregate.OUTPUT_ROOT = JOINTTRAIN_ARCH6_O196C_RESULT_ROOT
    aggregate.SCIENTIFIC_SCOPE = (
        "fresh-world equal-exposure recovery residual stability comparison"
    )
    aggregate.DECISION = "balanced residual screen complete; assess sampler stability"
    aggregate.ARMS = (
        "baseline_mlp",
        "random_seed1",
        "balanced_seed1",
        "balanced_seed2",
    )
    aggregate.PAIRWISE_COMPARISONS = (
        ("random_seed1", "baseline_mlp"),
        ("balanced_seed1", "baseline_mlp"),
        ("balanced_seed2", "baseline_mlp"),
        ("balanced_seed1", "random_seed1"),
        ("balanced_seed2", "random_seed1"),
        ("balanced_seed2", "balanced_seed1"),
    )
    return aggregate.main()


if __name__ == "__main__":
    raise SystemExit(main())

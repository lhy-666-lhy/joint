#!/usr/bin/env python3
"""Aggregate fresh-world live8 for the averaged recovery residual."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o193c_fresh_world_fixed_budget_live_aggregate as aggregate
from path_config import JOINTTRAIN_ARCH6_O199C_RESULT_ROOT


def main() -> int:
    aggregate.RUN_ID = "a6_o199c_recovery_residual_weight_average_live8_v1"
    aggregate.OUTPUT_ROOT = JOINTTRAIN_ARCH6_O199C_RESULT_ROOT
    aggregate.SCIENTIFIC_SCOPE = "fresh-world recovery residual weight-average comparison"
    aggregate.DECISION = "weight-average screen complete; freeze residual direction decision"
    aggregate.ARMS = (
        "baseline_mlp",
        "residual_seed1",
        "residual_seed2",
        "residual_average",
    )
    aggregate.PAIRWISE_COMPARISONS = (
        ("residual_seed1", "baseline_mlp"),
        ("residual_seed2", "baseline_mlp"),
        ("residual_average", "baseline_mlp"),
        ("residual_average", "residual_seed1"),
        ("residual_average", "residual_seed2"),
        ("residual_seed2", "residual_seed1"),
    )
    return aggregate.main()


if __name__ == "__main__":
    raise SystemExit(main())

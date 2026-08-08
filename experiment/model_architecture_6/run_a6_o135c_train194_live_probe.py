#!/usr/bin/env python3
"""Run the frozen O126 live loop with TRAIN194 MLP/PAR checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o126c_zero_contact_live_probe as live
from a6_operation_models import OperationMLPAbsolute, OperationParallelAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_O132C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O133C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O135C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-calls", type=int, default=3)
    parser.add_argument("--target-index", type=int, default=0)
    args, _ = parser.parse_known_args()
    live.ARMS = {
        "mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O132C_RESULT_ROOT) / "last.pt",
        ),
        "parallel": (
            OperationParallelAbsolute,
            Path(JOINTTRAIN_ARCH6_O133C_RESULT_ROOT) / "last.pt",
        ),
        "repeat_last": (None, None),
    }
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = JOINTTRAIN_ARCH6_O135C_RESULT_ROOT
    code = live.main()
    out = Path(JOINTTRAIN_ARCH6_O135C_RESULT_ROOT) / (
        f"probe_calls_{args.max_calls}_target_{args.target_index}"
    )
    summary = json.load(open(out / "summary.json"))
    summary["run_id"] = "a6_o135c_train194_live_probe_v1"
    summary["scientific_scope"] = (
        "A5_CAL source-horizon live closed loop with TRAIN194 checkpoints"
    )
    summary["checkpoint_scope"] = {
        "mlp": "A6-O132C TRAIN194",
        "parallel": "A6-O133C TRAIN194",
        "repeat_last": "control",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

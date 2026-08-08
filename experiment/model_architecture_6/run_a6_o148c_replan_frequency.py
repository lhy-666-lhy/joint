#!/usr/bin/env python3
"""Run frozen O127C with K8/K4 while preserving the physics-step budget."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o126c_zero_contact_live_probe as live
from a6_operation_models import OperationMLPAbsolute
from path_config import (
    JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O148C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--execute-prefix", type=int, choices=(2, 4, 8), required=True)
    args, _ = parser.parse_known_args()
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "repeat_last": (None, None),
    }
    mode_root = Path(JOINTTRAIN_ARCH6_O148C_RESULT_ROOT) / f"k{args.execute_prefix}"
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = str(mode_root)
    code = live.main()
    out = mode_root / f"probe_calls_{args.max_calls}_target_{args.target_index}"
    summary = json.load(open(out / "summary.json"))
    summary["run_id"] = "a6_o148c_replan_prefix_k4_v1"
    summary["scientific_scope"] = (
        "O127C replan-frequency comparison with fixed maximum physics-step budget"
    )
    summary["mode"] = f"K{args.execute_prefix}"
    summary["checkpoint_scope"] = {
        "baseline_mlp": "A6-O127C TRAIN1024 frozen",
        "repeat_last": "control",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

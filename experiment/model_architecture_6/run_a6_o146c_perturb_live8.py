#!/usr/bin/env python3
"""Run baseline and consistently perturbed MLPs in the same live8 protocol."""
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
    JOINTTRAIN_ARCH6_O143C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O144C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O146C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-calls", type=int, default=3)
    parser.add_argument("--target-index", type=int, default=0)
    args, _ = parser.parse_known_args()
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "perturb_1x": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O143C_RESULT_ROOT) / "last.pt",
        ),
        "perturb_3x": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O144C_RESULT_ROOT) / "last.pt",
        ),
        "repeat_last": (None, None),
    }
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = JOINTTRAIN_ARCH6_O146C_RESULT_ROOT
    code = live.main()
    out = Path(JOINTTRAIN_ARCH6_O146C_RESULT_ROOT) / (
        f"probe_calls_{args.max_calls}_target_{args.target_index}"
    )
    summary = json.load(open(out / "summary.json"))
    summary["run_id"] = "a6_o146c_perturb_live8_v1"
    summary["scientific_scope"] = (
        "same source-horizon live closed loop for baseline/1x/3x MLPs"
    )
    summary["checkpoint_scope"] = {
        "baseline_mlp": "A6-O127C TRAIN1024",
        "perturb_1x": "A6-O143C TRAIN1024 perturb 1x",
        "perturb_3x": "A6-O144C TRAIN1024 perturb 3x",
        "repeat_last": "control",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

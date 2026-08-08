#!/usr/bin/env python3
"""Run the unchanged A6 live8 protocol for the recovery-trained MLP."""

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
    JOINTTRAIN_ARCH6_O161C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O163C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


def main() -> int:
    live.ARMS = {
        "baseline_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT) / "last.pt",
        ),
        "recovery_mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O161C_RESULT_ROOT) / "last.pt",
        ),
        "repeat_last": (None, None),
    }
    live.STATE_APPENDERS = {}
    live.MODEL_INPUT_FEATURES = {
        "baseline_mlp": [],
        "recovery_mlp": [],
        "repeat_last": [],
    }
    live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT = str(JOINTTRAIN_ARCH6_O163C_RESULT_ROOT)
    code = live.main()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--execute-prefix", type=int, required=True)
    args, _ = parser.parse_known_args()
    out = Path(JOINTTRAIN_ARCH6_O163C_RESULT_ROOT) / f"probe_calls_{args.max_calls}_target_{args.target_index}"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary["run_id"] = "a6_o163c_recovery_live8_v1"
    summary["scientific_scope"] = "A5_CAL source-horizon live8 baseline versus TRAIN-only recovery MLP"
    summary["mode"] = "K8"
    summary["checkpoint_scope"] = {
        "baseline_mlp": "A6-O127C TRAIN1024 frozen",
        "recovery_mlp": "A6-O161C TRAIN1088 frozen",
        "repeat_last": "control",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

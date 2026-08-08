#!/usr/bin/env python3
"""Evaluate frozen TRAIN194 MLP/PAR checkpoints on the unchanged A5_CAL input."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_a6_o125c_zero_contact_cal_screen as evaluator
from a6_operation_models import (
    OperationCausalAbsolute,
    OperationMLPAbsolute,
    OperationParallelAbsolute,
)
from path_config import (
    JOINTTRAIN_ARCH6_O124C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O132C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O133C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_O134C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


def main() -> int:
    evaluator.ARMS = {
        "mlp": (
            OperationMLPAbsolute,
            Path(JOINTTRAIN_ARCH6_O132C_RESULT_ROOT) / "last.pt",
        ),
        "parallel": (
            OperationParallelAbsolute,
            Path(JOINTTRAIN_ARCH6_O133C_RESULT_ROOT) / "last.pt",
        ),
        "causal": (
            OperationCausalAbsolute,
            Path(JOINTTRAIN_ARCH6_O124C_RESULT_ROOT) / "last.pt",
        ),
    }
    evaluator.JOINTTRAIN_ARCH6_O125C_RESULT_ROOT = JOINTTRAIN_ARCH6_O134C_RESULT_ROOT
    code = evaluator.main()
    out = Path(JOINTTRAIN_ARCH6_O134C_RESULT_ROOT)
    summary = json.load(open(out / "summary.json"))
    summary["run_id"] = "a6_o134c_train194_cal_screen_v1"
    summary["scientific_scope"] = (
        "A5_CAL zero-contact screen: MLP/PAR trained on TRAIN194; "
        "causal fixed64 diagnostic"
    )
    summary["checkpoint_scope"] = {
        "mlp": "TRAIN194",
        "parallel": "TRAIN194",
        "causal": "fixed64 diagnostic",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    print(json.dumps(summary))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

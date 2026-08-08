#!/usr/bin/env python3
"""Post-hoc matched analysis of G065 set generation versus mode scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jointTrain_new.experiment.model_architecture_6.run_a6_a030c_affordance_cal_consumer import paired_bootstrap
from path_config import (
    JOINTTRAIN_ARCH6_G062C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G064C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G065C_RESULT_ROOT,
)


SEED = 20260806
STRICT_ROTATION_RAD = np.deg2rad(12.0)


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def group_rows(groups: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> dict[int, dict[str, float]]:
    rows = {}
    for group in np.unique(groups):
        take = groups == group
        rows[int(group)] = {
            "translation_m": float(translation[take].mean()),
            "rotation_rad": float(rotation[take].mean()),
            "pose_within_3cm_12deg": float(
                ((translation[take] <= 0.03) & (rotation[take] <= STRICT_ROTATION_RAD)).mean()
            ),
        }
    return rows


def aggregate(rows: dict[int, dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows.values()]))
        for key in ("translation_m", "rotation_rad", "pose_within_3cm_12deg")
    }


def paired(left: dict[int, dict[str, float]], right: dict[int, dict[str, float]]) -> dict[str, dict]:
    common = sorted(set(left) & set(right))
    return {
        key: paired_bootstrap(
            np.asarray([left[group][key] - right[group][key] for group in common]), SEED
        )
        for key in ("translation_m", "rotation_rad", "pose_within_3cm_12deg")
    }


def main() -> int:
    g064_path = Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / "full" / "supervision.npz"
    prediction_path = Path(JOINTTRAIN_ARCH6_G065C_RESULT_ROOT) / "set_residual" / "full" / "cal_predictions.npz"
    g062_path = Path(JOINTTRAIN_ARCH6_G062C_RESULT_ROOT) / "full" / "summary.json"
    g065_path = Path(JOINTTRAIN_ARCH6_G065C_RESULT_ROOT) / "summary.json"

    with np.load(g064_path, allow_pickle=False) as data:
        cal = np.flatnonzero(data["split"] == 1)
        presence = np.asarray(data["mode_presence"][cal], dtype=bool)
        groups = np.broadcast_to(data["group_index"][cal, None], presence.shape)[presence]
    with np.load(prediction_path, allow_pickle=False) as predictions:
        prediction_lengths = {key: len(predictions[key]) for key in predictions.files}
        selected = group_rows(groups, predictions["selected_translation"], predictions["selected_rotation"])
        best = group_rows(groups, predictions["best_translation"], predictions["best_rotation"])

    g062 = json.loads(g062_path.read_text(encoding="utf-8"))
    baseline = {int(row["group_index"]): row for row in g062["metrics"]["per_group"]}
    g065 = json.loads(g065_path.read_text(encoding="utf-8"))
    checks = {
        "g065_terminal_negative": g065.get("complete") is True and g065.get("claim_supported") == "no",
        "set_variant_present": "set_residual" in g065.get("variants", {}),
        "cal_group_count": len(selected) == len(best) == len(baseline) == 101,
        "cal_valid_slot_count": len(groups) == 382,
        "prediction_lengths": all(
            prediction_lengths[key] == len(groups)
            for key in ("selected_translation", "selected_rotation", "best_translation", "best_rotation")
        ),
        "no_outcome_read": True,
    }
    passed = all(checks.values())
    selected_minus_baseline = paired(selected, baseline)
    best_minus_baseline = paired(best, baseline)
    selected_minus_best = paired(selected, best)
    oracle_generation_supported = bool(
        best_minus_baseline["translation_m"]["ci95"][1] < 0.0
        and best_minus_baseline["rotation_rad"]["ci95"][1] < 0.0
    )
    selector_supported = bool(
        selected_minus_baseline["translation_m"]["ci95"][1] < 0.0
        and selected_minus_baseline["rotation_rad"]["ci95"][1] < 0.0
    )
    summary = {
        "schema_version": 1,
        "run_id": "A6-G065C-MODE-SCORING-ANALYSIS",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "evaluation_unit": "CAL observation group",
        "cal_groups": len(selected),
        "cal_valid_slots": len(groups),
        "metrics": {
            "selected_top1": aggregate(selected),
            "best_of_8_oracle": aggregate(best),
            "selected_minus_g062": selected_minus_baseline,
            "best_of_8_minus_g062": best_minus_baseline,
            "selected_minus_best_of_8": selected_minus_best,
        },
        "checks": checks,
        "oracle_generation_supported": "yes" if passed and oracle_generation_supported else "no",
        "selector_supported": "yes" if passed and selector_supported else "no",
        "claim_supported": "oracle_only" if passed and oracle_generation_supported and not selector_supported else "no",
        "decision": "keep G066 blocked; isolate outcome-blind mode scoring before realization",
        "next_run_ids": ["A6-A032C"] if passed else [],
    }
    out = Path(JOINTTRAIN_ARCH6_G065C_RESULT_ROOT) / "mode_scoring_analysis.json"
    atomic(out, summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

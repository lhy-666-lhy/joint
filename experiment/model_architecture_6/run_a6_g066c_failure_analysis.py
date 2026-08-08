#!/usr/bin/env python3
"""Diagnose whether G066 fails at realization coverage or score alignment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jointTrain_new.experiment.model_architecture_6.run_a6_g066c_selector_realization import (
    ROTATION_SCALE,
    TRANSLATION_SCALE,
    group_rows,
    paired,
)
from path_config import JOINTTRAIN_ARCH6_G062C_RESULT_ROOT, JOINTTRAIN_ARCH6_G066C_RESULT_ROOT


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def masked_best(cost: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.where(mask, cost, np.inf)
    selected = value.argmin(axis=-1)
    available = mask.any(axis=-1)
    return np.where(available, selected, fallback)


def main() -> int:
    root = Path(JOINTTRAIN_ARCH6_G066C_RESULT_ROOT)
    with np.load(root / "score" / "full" / "candidate_predictions.npz", allow_pickle=False) as data:
        candidates = {key: np.asarray(data[key]) for key in data.files}
    with np.load(root / "score" / "full" / "evaluation_labels.npz", allow_pickle=False) as data:
        labels = {key: np.asarray(data[key]) for key in data.files}
    with np.load(root / "realize" / "full" / "realization.npz", allow_pickle=False) as data:
        realize = {key: np.asarray(data[key]) for key in data.files}
    cost = labels["translation_error"] / TRANSLATION_SCALE + labels["rotation_error"] / ROTATION_SCALE
    oracle = cost.argmin(axis=-1)
    ik_mask = realize["ik_presence"].any(axis=-1)
    planner_mask = realize["planner_success"].any(axis=-1)
    ik_oracle = masked_best(cost, ik_mask, oracle)
    planner_oracle = masked_best(cost, planner_mask, ik_oracle)
    selectors = {
        "full_oracle": oracle,
        "ik_feasible_oracle": ik_oracle,
        "planner_feasible_oracle": planner_oracle,
        "s0_mode_logit": candidates["s0_selected"],
        "s1_calibrated_risk": candidates["s1_selected"],
        "s2_ik_fk_planner": realize["s2_selected"],
    }
    metrics = {}
    per_group = {}
    valid = labels["presence"]
    for name, selected in selectors.items():
        translation = np.take_along_axis(labels["translation_error"], selected[..., None], axis=-1).squeeze(-1)
        rotation = np.take_along_axis(labels["rotation_error"], selected[..., None], axis=-1).squeeze(-1)
        rows = group_rows(candidates["group_index"], valid, translation, rotation)
        per_group[name] = rows
        metrics[name] = {
            "translation_m": float(translation[valid].mean()),
            "rotation_rad": float(rotation[valid].mean()),
            "pose_within_3cm_12deg": float(
                ((translation[valid] <= TRANSLATION_SCALE) & (rotation[valid] <= ROTATION_SCALE)).mean()
            ),
        }
    baseline = json.loads((Path(JOINTTRAIN_ARCH6_G062C_RESULT_ROOT) / "full" / "summary.json").read_text())
    baseline_groups = {int(row["group_index"]): row for row in baseline["metrics"]["per_group"]}
    valid_oracle = oracle[valid]
    oracle_ik = np.take_along_axis(ik_mask, oracle[..., None], axis=-1).squeeze(-1)
    oracle_plan = np.take_along_axis(planner_mask, oracle[..., None], axis=-1).squeeze(-1)
    plan_length = np.where(realize["planner_success"], realize["path_length"], np.nan)
    best_path = np.nanmin(plan_length, axis=-1)
    finite = np.isfinite(best_path) & np.broadcast_to(valid[..., None], best_path.shape)
    correlation = spearmanr(cost[finite], best_path[finite]).statistic if finite.any() else float("nan")
    ik_count = realize["ik_presence"].sum(axis=-1)
    valid_candidates = np.broadcast_to(valid[..., None], ik_count.shape)
    ik_correlation = spearmanr(cost[valid_candidates], ik_count[valid_candidates]).statistic
    comparisons = {
        "ik_feasible_oracle_minus_full_oracle": paired(per_group["ik_feasible_oracle"], per_group["full_oracle"]),
        "planner_feasible_oracle_minus_full_oracle": paired(per_group["planner_feasible_oracle"], per_group["full_oracle"]),
        "planner_feasible_oracle_minus_g062": paired(per_group["planner_feasible_oracle"], baseline_groups),
    }
    checks = {
        "g066_terminal_negative": json.loads((root / "summary.json").read_text())["claim_supported"] == "no",
        "group_count_101": len(candidates["group_index"]) == 101,
        "valid_labels_382": int(valid.sum()) == 382,
        "selection_shape": oracle.shape == (101, 4),
        "finite_metrics": all(np.isfinite(value) for item in metrics.values() for value in item.values()),
        "no_outcome_read": True,
        "diagnostic_only": True,
    }
    passed = all(checks.values())
    planner_oracle_vs_g062 = comparisons["planner_feasible_oracle_minus_g062"]
    feasible_generation_survives = bool(
        planner_oracle_vs_g062["translation_m"]["ci95"][1] < 0.0
        and planner_oracle_vs_g062["rotation_rad"]["ci95"][1] < 0.0
    )
    summary = {
        "schema_version": 1,
        "run_id": "A6-G066C-FAILURE-ANALYSIS",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "metrics": metrics,
        "coverage": {
            "oracle_pose_has_legal_ik": float(oracle_ik[valid].mean()),
            "oracle_pose_has_planner_path": float(oracle_plan[valid].mean()),
            "selected_mode_exact_oracle_rate": {
                name: float((selected[valid] == valid_oracle).mean())
                for name, selected in selectors.items()
                if name.startswith("s")
            },
        },
        "correlation": {
            "pose_cost_vs_best_path_length_spearman": float(correlation),
            "pose_cost_vs_ik_solution_count_spearman": float(ik_correlation),
        },
        "comparisons": comparisons,
        "checks": checks,
        "feasible_generation_survives": "yes" if feasible_generation_survives else "no",
        "claim_supported": "diagnostic_only",
        "decision": (
            "revise generator with assignment-aligned quality head; do not tune feasibility rank"
            if feasible_generation_survives
            else "repair IK/planner realization before any score revision"
        ),
        "next_run_ids": [],
    }
    atomic(root / "failure_analysis.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

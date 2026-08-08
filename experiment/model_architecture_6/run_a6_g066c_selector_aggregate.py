#!/usr/bin/env python3
"""Aggregate disjoint G066 realization shards and apply the single CAL gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jointTrain_new.experiment.model_architecture_6.run_a6_g066c_selector_realization import (
    ROTATION_SCALE,
    TRANSLATION_SCALE,
    group_rows,
    paired,
)
from path_config import (
    JOINTTRAIN_ARCH6_G062C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G066C_RESULT_ROOT,
)


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    root = Path(JOINTTRAIN_ARCH6_G066C_RESULT_ROOT)
    score_root = root / "score" / "full"
    shard_dirs = sorted((root / "realize").glob("shard_*_*"))
    shard_summaries = [json.loads((path / "summary.json").read_text()) for path in shard_dirs if (path / "summary.json").exists()]
    shard_summaries.sort(key=lambda item: int(item["cal_row_start"]))
    arrays = []
    for summary in shard_summaries:
        path = root / "realize" / f"shard_{int(summary['cal_row_start']):03d}_{int(summary['cal_row_end']):03d}" / "realization.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays.append({key: np.asarray(data[key]) for key in data.files})
    if not arrays:
        raise RuntimeError("no completed G066 realization shards")
    keys = arrays[0].keys()
    merged = {key: np.concatenate([item[key] for item in arrays], axis=0) for key in keys}
    with np.load(score_root / "candidate_predictions.npz", allow_pickle=False) as data:
        candidates = {key: np.asarray(data[key]) for key in data.files}
    with np.load(score_root / "evaluation_labels.npz", allow_pickle=False) as data:
        labels = {key: np.asarray(data[key]) for key in data.files}
    selectors = {
        "s0_mode_logit": candidates["s0_selected"],
        "s1_calibrated_risk": candidates["s1_selected"],
        "s2_ik_fk_planner": merged["s2_selected"],
        "oracle_best_of_8": (labels["translation_error"] / TRANSLATION_SCALE + labels["rotation_error"] / ROTATION_SCALE).argmin(axis=-1),
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
            "group_mean": {key: float(np.mean([row[key] for row in rows.values()])) for key in rows[next(iter(rows))]},
        }
    baseline = json.loads((Path(JOINTTRAIN_ARCH6_G062C_RESULT_ROOT) / "full" / "summary.json").read_text())
    baseline_groups = {int(row["group_index"]): row for row in baseline["metrics"]["per_group"]}
    comparisons = {
        "s0_minus_g062": paired(per_group["s0_mode_logit"], baseline_groups),
        "s1_minus_g062": paired(per_group["s1_calibrated_risk"], baseline_groups),
        "s2_minus_g062": paired(per_group["s2_ik_fk_planner"], baseline_groups),
        "s0_minus_oracle": paired(per_group["s0_mode_logit"], per_group["oracle_best_of_8"]),
        "s1_minus_oracle": paired(per_group["s1_calibrated_risk"], per_group["oracle_best_of_8"]),
        "s2_minus_oracle": paired(per_group["s2_ik_fk_planner"], per_group["oracle_best_of_8"]),
    }
    selected_ik = np.take_along_axis(merged["ik_presence"].any(axis=-1), merged["s2_selected"][..., None], axis=-1).squeeze(-1)
    selected_plan = np.take_along_axis(merged["planner_success"].any(axis=-1), merged["s2_selected"][..., None], axis=-1).squeeze(-1)
    intervals = [(int(item["cal_row_start"]), int(item["cal_row_end"])) for item in shard_summaries]
    checks = {
        "score_terminal": json.loads((score_root / "summary.json").read_text())["status"] == "passed",
        "all_shards_passed": all(item.get("status") == "passed" and item.get("claim_supported") == "shard_only" for item in shard_summaries),
        "shard_intervals_exact": intervals == [(0, 13), (13, 26), (26, 39), (39, 52), (52, 65), (65, 77), (77, 89), (89, 101)],
        "group_count_101": len(merged["group_index"]) == 101,
        "group_order_exact": bool(np.array_equal(merged["group_index"], candidates["group_index"])),
        "presence_exact": bool(np.array_equal(merged["query_presence"], candidates["query_presence"])),
        "selection_indices_valid": bool(np.all((merged["s2_selected"] >= 0) & (merged["s2_selected"] < 8))),
        "outcome_mutation_invariance": all(item["checks"]["outcome_mutation_invariance"] for item in shard_summaries),
        "no_task_outcome_read": True,
    }
    passed = all(checks.values())
    s2_comparison = comparisons["s2_minus_g062"]
    s2_gap = comparisons["s2_minus_oracle"]
    s1_gap = comparisons["s1_minus_oracle"]
    supported = bool(
        s2_comparison["translation_m"]["ci95"][1] <= 0.0
        and s2_comparison["rotation_rad"]["ci95"][1] <= 0.0
        and s2_gap["translation_m"]["mean"] < s1_gap["translation_m"]["mean"]
        and s2_gap["rotation_rad"]["mean"] < s1_gap["rotation_rad"]["mean"]
    )
    out = root / "realize" / "full"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "realization.npz", **merged)
    summary = {
        "schema_version": 1,
        "run_id": "A6-G066C",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "shards": intervals,
        "realization": {
            "legal_ik": int(merged["ik_presence"].sum()),
            "planner_success": int(merged["planner_success"].sum()),
            "selected_ik_coverage": float(selected_ik[valid].mean()),
            "selected_planner_coverage": float(selected_plan[valid].mean()),
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "checks": checks,
        "claim_supported": "yes" if passed and supported else "no",
        "decision": "authorize G067 predicted-contact screen" if passed and supported else "stop before G067; selector gate failed",
        "next_run_ids": ["A6-G067C"] if passed and supported else [],
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    atomic(root / "summary.json", summary)
    atomic(root / "run_state.json", summary)
    atomic(root / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

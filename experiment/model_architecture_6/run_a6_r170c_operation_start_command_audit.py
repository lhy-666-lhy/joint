#!/usr/bin/env python3
"""Audit the operation-start last-command source used by A6 live rollouts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_operation_start_contract import load_operation_start
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D160C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_R170C_RESULT_ROOT,
)
from run_a6_o010r_mlp_fixed64 import atomic_json


RUN_ID = "a6_r170c_operation_start_command_audit_v1"
LIVE_TARGETS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def live_selection() -> list[dict]:
    manifest = json.loads(
        (
            Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT) / "full" / "input_manifest.json"
        ).read_text(encoding="utf-8")
    )
    selected: list[dict] = []
    seen: set[str] = set()
    for row in manifest["rows"]:
        if row["anchor_rank"] != 0 or row["target"] in seen:
            continue
        seen.add(row["target"])
        selected.append(
            {
                "split": "A5_CAL",
                "scope": "corrected_live8",
                "target": row["target"],
                "trajectory_relative_path": row["trajectory_relative_path"],
                "source_sha256": row["source_sha256"],
            }
        )
        if len(selected) == LIVE_TARGETS:
            break
    return selected


def recovery_selection() -> list[dict]:
    selection = json.loads(
        (Path(JOINTTRAIN_ARCH6_D160C_RESULT_ROOT) / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        {**row, "split": "A5_TRAIN", "scope": "recovery16"}
        for row in selection["targets"]
    ]


def max_abs(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    return float(np.max(np.abs(left - right)))


def main() -> int:
    selected = live_selection() + recovery_selection()
    rows: list[dict] = []
    for item in selected:
        trajectory = Path(ARTICU_COLLECTION_ROOT) / item["trajectory_relative_path"]
        record = load_operation_start(trajectory)
        command = record["command_qpos"]
        robot = record["robot_qpos"].astype(np.float32)
        deprecated = record["deprecated_command_qpos"]
        rows.append(
            {
                **item,
                "source_hash_matches": sha256_file(trajectory) == item["source_sha256"],
                "operation_start_index": record["operation_index"],
                "selected_command_source": record["command_source"],
                "selected_finger_command": command[7:9].tolist(),
                "selected_vs_logged_max_abs": max_abs(
                    command, record["logged_command_qpos"]
                ),
                "selected_vs_raw_max_abs": max_abs(command, record["raw_command_qpos"]),
                "selected_vs_repaired_max_abs": max_abs(
                    command, record["repaired_command_qpos"]
                ),
                "deprecated_vs_robot_max_abs": max_abs(deprecated, robot),
                "deprecated_vs_selected_max_abs": max_abs(deprecated, command),
                "deprecated_finger_value": None
                if deprecated is None
                else deprecated[7:9].tolist(),
                "consumed_fields": record["consumed_fields"],
            }
        )

    tolerance = 1e-7
    checks = {
        "live_targets_8": sum(row["scope"] == "corrected_live8" for row in rows)
        == 8,
        "recovery_targets_16": sum(row["scope"] == "recovery16" for row in rows)
        == 16,
        "source_hashes_match": all(row["source_hash_matches"] for row in rows),
        "selected_logged_source": all(
            row["selected_command_source"]
            == "logged_operation_start_joint_command_qpos"
            for row in rows
        ),
        "selected_matches_logged": all(
            row["selected_vs_logged_max_abs"] is not None
            and row["selected_vs_logged_max_abs"] <= tolerance
            for row in rows
        ),
        "selected_matches_raw": all(
            row["selected_vs_raw_max_abs"] is not None
            and row["selected_vs_raw_max_abs"] <= tolerance
            for row in rows
        ),
        "selected_matches_repaired": all(
            row["selected_vs_repaired_max_abs"] is not None
            and row["selected_vs_repaired_max_abs"] <= tolerance
            for row in rows
        ),
        "selected_finger_zero": all(
            max(abs(value) for value in row["selected_finger_command"]) <= tolerance
            for row in rows
        ),
        "deprecated_field_is_robot_qpos": all(
            row["deprecated_vs_robot_max_abs"] is not None
            and row["deprecated_vs_robot_max_abs"] <= tolerance
            for row in rows
        ),
        "deprecated_field_differs_from_command": all(
            row["deprecated_vs_selected_max_abs"] is not None
            and row["deprecated_vs_selected_max_abs"] > tolerance
            for row in rows
        ),
        "no_outcome_or_progress_consumed": all(
            not any(
                token in field
                for token in ("result", "outcome", "progress")
            )
            for row in rows
            for field in row["consumed_fields"]
        ),
    }
    passed = all(checks.values())
    out = Path(JOINTTRAIN_ARCH6_R170C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "rows": rows,
        "checks": checks,
        "invalidated_live_runs": [
            "A6-O131C",
            "A6-O135C/A6-O136C",
            "A6-O140C/A6-O141C-live",
            "A6-O146C/A6-O147C-live",
            "A6-O148C",
            "A6-O149C",
            "A6-O153C-live",
            "A6-O156C-live",
            "A6-D160C/A6-O161C/A6-O162C/A6-O163C",
        ],
    }
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "complete": True,
        "terminal": True,
        "status": "passed" if passed else "failed",
        "failure_class": None if passed else "operation_start_contract_failure",
        "scientific_scope": "A6 operation-start command-source audit",
        "checks": checks,
        "counts": {"rows": len(rows), "live_cal": 8, "recovery_train": 16},
        "decision": "authorize corrected O171C live8"
        if passed
        else "repair operation-start command selection before any live rollout",
        "next_run_ids": ["A6-O171C"] if passed else [],
    }
    atomic_json(out / "audit.json", audit)
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run_state.json", summary)
    atomic_json(
        out / "queue_state.json",
        {**summary, "jobs": [{"id": "A6-R170C", "status": summary["status"]}]},
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

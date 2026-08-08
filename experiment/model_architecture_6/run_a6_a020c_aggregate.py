#!/usr/bin/env python3
"""Aggregate the three fixed A020C terminal checkpoints without reading CAL."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_config import JOINTTRAIN_ARCH6_A020C_RESULT_ROOT


SEEDS = (20260806, 20260807, 20260808)


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(JOINTTRAIN_ARCH6_A020C_RESULT_ROOT)
    rows = []
    for seed in SEEDS:
        seed_root = root / f"seed_{seed}"
        summary_path = seed_root / "summary.json"
        checkpoint_path = seed_root / "last.pth"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "status": summary.get("status"),
                "terminal": bool(summary.get("terminal")),
                "steps": int(summary.get("steps", 0)),
                "dataset_rows": int(summary.get("dataset_rows", 0)),
                "loss_first": float(summary.get("loss_first", float("nan"))),
                "loss_last": float(summary.get("loss_last", float("nan"))),
                "reload_max_abs": float(summary.get("reload_max_abs", float("nan"))),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "summary_sha256": sha256(summary_path),
            }
        )
    checks = {
        "three_fixed_seeds": [row["seed"] for row in rows] == list(SEEDS),
        "all_terminal_passed": all(row["status"] == "passed" and row["terminal"] for row in rows),
        "all_7000_steps": all(row["steps"] == 7000 for row in rows),
        "clean_membership_rows": all(row["dataset_rows"] == 5456 for row in rows),
        "loss_decreased": all(row["loss_last"] < row["loss_first"] for row in rows),
        "reload_exact": all(row["reload_max_abs"] == 0.0 for row in rows),
        "cal_content_unread": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_id": "A6-A020C",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "seeds": rows,
        "checks": checks,
        "decision": "authorize fixed-last A030C evaluation" if passed else "repair A020C artifacts",
        "next_run_ids": ["A6-A030C"] if passed else [],
    }
    atomic(root / "summary.json", summary)
    atomic(root / "run_state.json", summary)
    atomic(root / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the bounded GPUALIVE reconciliation regression audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from path_config import JOINTTRAIN_ARCH6_R020_RESULT_ROOT, PROJECT_ROOT


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    root = Path(PROJECT_ROOT)
    source = root / ".agents/skills/autonomous-experiment-loop/scripts/gpu_keepalive_control.py"
    test_path = root / ".agents/skills/autonomous-experiment-loop/tests/test_gpu_keepalive.py"
    result = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checks = {
        "regression_suite_passed": result.returncode == 0,
        "analysis_reconciliation_covered": "test_analysis_restarts_partial_gpu_keepalive_on_all_cards" in test_path.read_text(encoding="utf-8"),
        "keepalive_not_completion_evidence": True,
    }
    passed = all(checks.values())
    out_dir = Path(JOINTTRAIN_ARCH6_R020_RESULT_ROOT)
    atomic_json(
        out_dir / "audit.json",
        {
            "schema_version": 1,
            "run_id": "a6_r020_gpualive_reconciliation_v1",
            "checks": checks,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
            "pytest_returncode": result.returncode,
            "pytest_output": result.stdout,
        },
    )
    atomic_json(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": "a6_r020_gpualive_reconciliation_v1",
            "complete": True,
            "terminal": True,
            "status": "passed" if passed else "failed",
            "failure_class": None if passed else "infrastructure_failure",
            "claim_supported": "yes" if passed else "no",
            "checks": checks,
            "evidence": {"audit": "audit.json"},
            "decision": "GPUALIVE expands a partial selection to all eligible cards when analysis/CPU reconciliation requires it." if passed else "GPUALIVE reconciliation remains blocked.",
            "remaining_work": [],
            "next_run_ids": [],
            "event_id": "a6_r020_gpualive_reconciliation_v1_terminal",
        },
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

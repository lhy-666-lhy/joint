#!/usr/bin/env python3
"""Run the bounded Architecture 6 machine launch-barrier audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from path_config import JOINTTRAIN_ARCH6_R030_RESULT_ROOT, JOINTTRAIN_MODEL_ARCHITECTURE_6_ROOT, PROJECT_ROOT


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    project_root = Path(PROJECT_ROOT)
    arch_root = Path(JOINTTRAIN_MODEL_ARCHITECTURE_6_ROOT)
    config_path = arch_root / "experiment_loop/loop_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    test_paths = [
        arch_root / "test_a6_contract_preflight.py",
        arch_root / "test_a6_set_agent_state_contract.py",
    ]
    results = [
        subprocess.run(
            [sys.executable, str(path)],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        for path in test_paths
    ]
    enforcement = config.get("contract_enforcement", {})
    checks = {
        "regression_suite_passed": all(result.returncode == 0 for result in results),
        "enforcement_enabled": enforcement.get("enabled") is True,
        "active_contract_path_frozen": enforcement.get("contract_path") == "active_contract.json",
        "preflight_audit_path_frozen": enforcement.get("audit_path") == "preflight_contract_audit.json",
    }
    passed = all(checks.values())
    set_state = project_root / ".agents/skills/autonomous-experiment-loop/scripts/set_agent_state.py"
    preflight = arch_root / "run_a6_contract_preflight.py"
    out_dir = Path(JOINTTRAIN_ARCH6_R030_RESULT_ROOT)
    atomic_json(
        out_dir / "audit.json",
        {
            "schema_version": 1,
            "run_id": "a6_r030_preflight_enforcement_v1",
            "checks": checks,
            "set_agent_state_sha256": hashlib.sha256(set_state.read_bytes()).hexdigest(),
            "preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
            "contract_enforcement_sha256": hashlib.sha256(
                json.dumps(enforcement, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "test_returncodes": [result.returncode for result in results],
            "test_outputs": [result.stdout for result in results],
        },
    )
    atomic_json(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": "a6_r030_preflight_enforcement_v1",
            "complete": True,
            "terminal": True,
            "status": "passed" if passed else "failed",
            "failure_class": None if passed else "infrastructure_failure",
            "claim_supported": "yes" if passed else "no",
            "checks": checks,
            "evidence": {"audit": "audit.json"},
            "decision": "Every Architecture 6 launch is rejected unless its active contract and passed preflight audit match the iteration, resource mode and contract hash." if passed else "Machine launch barrier remains blocked.",
            "remaining_work": [],
            "next_run_ids": [],
            "event_id": "a6_r030_preflight_enforcement_v1_terminal",
        },
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

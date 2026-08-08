from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SET_AGENT_STATE = ROOT.parents[2] / ".agents/skills/autonomous-experiment-loop/scripts/set_agent_state.py"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def setup_launch(tmp_path: Path, *, iteration: str = "r1") -> Path:
    contract_path = tmp_path / "active_contract.json"
    write_json(
        contract_path,
        {"status": "active", "run_id": iteration, "resource_mode": "cpu"},
    )
    write_json(
        tmp_path / "preflight_contract_audit.json",
        {
            "status": "passed",
            "run_id": iteration,
            "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        },
    )
    write_json(
        tmp_path / "loop_config.json",
        {
            "iteration_id": iteration,
            "resource_mode": "cpu",
            "agent_state_path": "agent_state.json",
            "contract_enforcement": {
                "enabled": True,
                "contract_path": "active_contract.json",
                "audit_path": "preflight_contract_audit.json",
            },
            "gpu_keepalive": {"enabled": False},
        },
    )
    return tmp_path / "loop_config.json"


def run_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SET_AGENT_STATE),
            "--config",
            str(config_path),
            "--state",
            "launching",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class SetAgentStateContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_launch_accepts_matching_passed_contract(self) -> None:
        config_path = setup_launch(self.root)
        result = run_launch(config_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((self.root / "agent_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["state"], "launching")
        self.assertEqual(state["iteration_id"], "r1")

    def test_launch_rejects_stale_contract_hash_without_state_change(self) -> None:
        config_path = setup_launch(self.root)
        contract_path = self.root / "active_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["dependencies"] = ["changed-after-preflight"]
        write_json(contract_path, contract)
        result = run_launch(config_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("audit_contract_hash", result.stderr)
        self.assertFalse((self.root / "agent_state.json").exists())

    def test_launch_rejects_wrong_iteration_without_state_change(self) -> None:
        config_path = setup_launch(self.root, iteration="r1")
        contract_path = self.root / "active_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["run_id"] = "stale-run"
        write_json(contract_path, contract)
        audit_path = self.root / "preflight_contract_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["contract_sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        write_json(audit_path, audit)
        result = run_launch(config_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contract_iteration", result.stderr)
        self.assertFalse((self.root / "agent_state.json").exists())


if __name__ == "__main__":
    unittest.main()

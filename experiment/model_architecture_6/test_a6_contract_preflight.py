from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "run_a6_contract_preflight.py"


class ContractPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_preflight(self, contract: dict, actual: dict) -> tuple[subprocess.CompletedProcess[str], dict]:
        contract_path = self.root / "contract.json"
        actual_path = self.root / "actual.json"
        output_path = self.root / "audit.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        actual_path.write_text(json.dumps(actual), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--contract", str(contract_path), "--actual", str(actual_path), "--output", str(output_path)],
            check=False,
        )
        return result, json.loads(output_path.read_text(encoding="utf-8"))

    def test_preflight_accepts_exact_contract(self) -> None:
        contract = {
            "run_id": "r1",
            "resource_mode": "gpu",
            "required_config": {"config.learning_rate": 0.0001},
            "allowed_splits": ["A5_TRAIN"],
            "forbidden_splits": ["A5_MECH_DEV"],
            "input_artifacts": {"fixed_input": "abc"},
            "dependencies": ["d1"],
        }
        actual = {
            "run_id": "r1",
            "resource_mode": "gpu",
            "config": {"learning_rate": 0.0001},
            "splits_read": ["A5_TRAIN"],
            "input_artifacts": {"fixed_input": "abc"},
            "dependencies": ["d1"],
        }
        result, audit = self.run_preflight(contract, actual)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(audit["status"], "passed")

    def test_preflight_rejects_config_or_split_drift(self) -> None:
        contract = {
            "run_id": "r1",
            "resource_mode": "gpu",
            "required_config": {"config.learning_rate": 0.0001},
            "allowed_splits": ["A5_TRAIN"],
            "forbidden_splits": ["A5_MECH_DEV"],
            "input_artifacts": {},
            "dependencies": [],
        }
        actual = {
            "run_id": "r1",
            "resource_mode": "gpu",
            "config": {"learning_rate": 0.001},
            "splits_read": ["A5_MECH_DEV"],
            "input_artifacts": {},
            "dependencies": [],
        }
        result, audit = self.run_preflight(contract, actual)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(audit["checks"]["required_config_exact"])
        self.assertFalse(audit["checks"]["forbidden_splits_absent"])


if __name__ == "__main__":
    unittest.main()

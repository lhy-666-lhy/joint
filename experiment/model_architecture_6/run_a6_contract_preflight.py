#!/usr/bin/env python3
"""Validate an Architecture 6 launch against its machine-readable contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_key)
        value = value[key]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    required = contract.get("required_config", {})
    config_checks: dict[str, bool] = {}
    for key, expected in required.items():
        try:
            config_checks[key] = nested_value(actual, key) == expected
        except KeyError:
            config_checks[key] = False
    splits_read = set(actual.get("splits_read", []))
    allowed_splits = set(contract.get("allowed_splits", []))
    forbidden_splits = set(contract.get("forbidden_splits", []))
    checks = {
        "run_id_exact": actual.get("run_id") == contract.get("run_id"),
        "resource_mode_exact": actual.get("resource_mode") == contract.get("resource_mode"),
        "required_config_exact": all(config_checks.values()),
        "splits_allowed": splits_read <= allowed_splits,
        "forbidden_splits_absent": not bool(splits_read & forbidden_splits),
        "input_artifacts_exact": actual.get("input_artifacts", {}) == contract.get("input_artifacts", {}),
        "dependencies_exact": actual.get("dependencies", []) == contract.get("dependencies", []),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "run_id": contract.get("run_id"),
        "status": "passed" if passed else "failed",
        "contract_sha256": sha256_file(args.contract),
        "actual_sha256": sha256_file(args.actual),
        "checks": checks,
        "config_checks": config_checks,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

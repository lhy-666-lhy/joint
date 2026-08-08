#!/usr/bin/env python3
"""Artifact-only completion checker for Architecture 6 iterations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    payload = read_json(summary_path)
    complete = bool(payload.get("complete") or payload.get("terminal"))
    status = str(payload.get("status") or ("success" if complete else "pending"))
    result_paths = [str(summary_path)] if summary_path.exists() else []
    print(json.dumps({
        "complete": complete,
        "event_id": str(payload.get("event_id") or args.iteration),
        "status": status,
        "result_paths": result_paths,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

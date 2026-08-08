from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def now() -> datetime:
    return datetime.now().astimezone()


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    tmp.replace(path)


def resolve(base: Path, path: str | None) -> Path | None:
    if not path:
        return None
    item = Path(path)
    return item if item.is_absolute() else base / item


def mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def run_command(command: dict[str, Any], root: Path, *, dry_run: bool) -> dict[str, Any]:
    cmd = [str(item) for item in command.get("cmd", [])]
    cwd = resolve(root, command.get("cwd")) or root
    log_path = resolve(root, command.get("log_path"))
    if dry_run:
        return {"dry_run": True, "cmd": cmd, "cwd": str(cwd), "log_path": str(log_path) if log_path else None}
    started = now()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n# supervisor command started: {iso(started)}\n")
            log.write("# command: " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
        output_tail = ""
    else:
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        output_tail = (proc.stdout or "")[-4000:]
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": int(proc.returncode),
        "started_at": iso(started),
        "finished_at": iso(now()),
        "output_tail": output_tail,
    }


def process_lines(patterns: list[str]) -> list[str]:
    if not patterns:
        return []
    try:
        proc = subprocess.run(["ps", "-ef"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        if "experiment_supervisor.py" in line:
            continue
        if all(pattern in line for pattern in patterns):
            rows.append(line)
    return rows[:8]


def parse_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = read_json(path, {})
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def check_job(job: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = resolve(root, job.get("summary_path"))
    log_path = resolve(root, job.get("log_path"))
    artifact_paths = [resolve(root, item) for item in job.get("artifact_paths", [])]
    timestamps = [mtime(summary_path), mtime(log_path)] + [mtime(item) for item in artifact_paths]
    timestamps = [item for item in timestamps if item is not None]
    latest = max(timestamps) if timestamps else None
    summary = parse_summary(summary_path)
    status = "complete" if summary is not None else "pending"
    metrics: dict[str, Any] = {}
    if summary is not None:
        error = str(summary.get("status", "")).lower() == "error" or int(summary.get("error_count", 0) or 0) > 0
        status = "error" if error else "complete"
        metrics = {
            "summary_status": summary.get("status"),
            "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0) or 0.0),
            "success_count": int(summary.get("success_count", 0) or 0),
            "finished_count": int(summary.get("finished_count", 0) or 0),
            "failure_reason_counts": summary.get("failure_reason_counts", {}),
        }
        if "cluster_affordance_count" in summary:
            metrics["cluster_success_count"] = int(summary.get("cluster_affordance_success_count", 0) or 0)
            metrics["cluster_count"] = int(summary.get("cluster_affordance_count", 0) or 0)
    return {
        "id": str(job.get("id", "")),
        "status": status,
        "summary_path": str(summary_path) if summary_path else "",
        "log_path": str(log_path) if log_path else "",
        "latest_mtime": iso(datetime.fromtimestamp(latest).astimezone()) if latest else None,
        "metrics": metrics,
    }


def check_round(round_cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    jobs = [check_job(job, root) for job in round_cfg.get("jobs", [])]
    complete = [job for job in jobs if job["status"] == "complete"]
    errors = [job for job in jobs if job["status"] == "error"]
    process = process_lines([str(item) for item in round_cfg.get("process_patterns", [])])
    elapsed = [float(job.get("metrics", {}).get("elapsed_seconds", 0.0) or 0.0) for job in complete]
    unit_seconds = sum(elapsed) / len(elapsed) if elapsed else float(round_cfg.get("expected_unit_seconds", 3600) or 3600)
    incomplete_count = max(0, len(jobs) - len(complete) - len(errors))
    status = "complete" if len(complete) + len(errors) == len(jobs) and not errors else "error" if errors else "running_or_pending" if process or complete else "pending"
    return {
        "id": str(round_cfg.get("id", "")),
        "status": status,
        "complete_count": len(complete),
        "error_count": len(errors),
        "total_count": len(jobs),
        "estimated_unit_seconds": unit_seconds,
        "estimated_remaining_seconds": incomplete_count * unit_seconds,
        "process_lines": process,
        "jobs": jobs,
    }


def choose_next_check(config: dict[str, Any], rounds: list[dict[str, Any]]) -> datetime:
    policy = config.get("policy", {})
    active = next((item for item in rounds if item["status"] not in {"complete", "error"}), None)
    if active is None:
        delay = int(policy.get("eta_lt_1h_seconds", 1200) or 1200)
    elif active["complete_count"] == 0:
        delay = int(policy.get("pre_first_unit_seconds", 3600) or 3600)
    else:
        eta = float(active.get("estimated_remaining_seconds", 0.0) or 0.0)
        if eta > 4 * 3600:
            delay = int(policy.get("eta_gt_4h_seconds", 7200) or 7200)
        elif eta > 3600:
            delay = int(policy.get("eta_1h_to_4h_seconds", 3600) or 3600)
        else:
            delay = int(policy.get("eta_lt_1h_seconds", 1200) or 1200)
    return now() + timedelta(seconds=max(60, delay))


def dependencies_complete(round_cfg: dict[str, Any], states: dict[str, dict[str, Any]]) -> bool:
    return all(states.get(str(dep), {}).get("status") == "complete" for dep in round_cfg.get("depends_on", []))


def maybe_launch(config: dict[str, Any], root: Path, round_states: list[dict[str, Any]], *, allow_launch: bool, dry_run: bool) -> dict[str, Any] | None:
    if not allow_launch:
        return None
    states = {item["id"]: item for item in round_states}
    for round_cfg in config.get("rounds", []):
        state = states.get(str(round_cfg.get("id")), {})
        if state.get("status") == "complete" or state.get("process_lines") or state.get("complete_count", 0) > 0:
            continue
        if not dependencies_complete(round_cfg, states):
            continue
        if not bool(round_cfg.get("auto_launch", False)):
            continue
        launch = round_cfg.get("launch")
        if launch:
            return {"round": round_cfg.get("id"), "action": run_command(launch, root, dry_run=dry_run)}
    return None


def run_commands(config: dict[str, Any], root: Path, round_states: list[dict[str, Any]], key: str, *, dry_run: bool) -> list[dict[str, Any]]:
    states = {item["id"]: item for item in round_states}
    actions = []
    for round_cfg in config.get("rounds", []):
        state = states.get(str(round_cfg.get("id")), {})
        if state.get("complete_count", 0) <= 0:
            continue
        for command in round_cfg.get(key, []):
            actions.append({"round": round_cfg.get("id"), "kind": key, "action": run_command(command, root, dry_run=dry_run)})
    return actions


def run_final_analyze(config: dict[str, Any], root: Path, round_states: list[dict[str, Any]], *, dry_run: bool) -> list[dict[str, Any]]:
    if not round_states or not all(item["status"] in {"complete", "error"} for item in round_states):
        return []
    return [{"kind": "final_analyze", "action": run_command(command, root, dry_run=dry_run)} for command in config.get("final_analyze", [])]


def write_next_check(path: Path, state: dict[str, Any]) -> None:
    lines = ["# Next Experiment Check", "", f"Updated: {state['updated_at']}", f"Next check: {state['next_check_at']}", "", "## Rounds"]
    for item in state.get("rounds", []):
        lines.append(f"- `{item['id']}`: {item['status']} {item['complete_count']}/{item['total_count']} complete, ETA {float(item.get('estimated_remaining_seconds', 0.0) or 0.0) / 3600.0:.2f}h")
    lines.extend(["", "## Read-Only Check", "", "```bash", f"python experiment_supervisor.py --config {state['config_path']} --once --read-only", "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_once(config_path: Path, *, read_only: bool, allow_launch: bool, dry_run: bool) -> dict[str, Any]:
    config = read_json(config_path, {})
    config_dir = config_path.parent.resolve()
    root = resolve(config_dir, config.get("root")) or config_dir
    rounds = [check_round(round_cfg, root) for round_cfg in config.get("rounds", [])]
    actions: list[dict[str, Any]] = []
    launch_action = None
    if not read_only:
        actions.extend(run_commands(config, root, rounds, "summarize", dry_run=dry_run))
        actions.extend(run_commands(config, root, rounds, "analyze", dry_run=dry_run))
        actions.extend(run_final_analyze(config, root, rounds, dry_run=dry_run))
        launch_action = maybe_launch(config, root, rounds, allow_launch=allow_launch, dry_run=dry_run)
    next_check = choose_next_check(config, rounds)
    state = {
        "schema_version": 1,
        "updated_at": iso(now()),
        "config_path": str(config_path),
        "root": str(root),
        "read_only": bool(read_only),
        "allow_launch": bool(allow_launch),
        "next_check_at": iso(next_check),
        "rounds": rounds,
        "actions": actions,
        "launch_action": launch_action,
    }
    state_path = resolve(config_dir, config.get("state_path")) or config_dir / "monitor_state.json"
    next_path = resolve(config_dir, config.get("next_check_path")) or config_dir / "NEXT_CHECK.md"
    write_json(state_path, state)
    write_next_check(next_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic long-running experiment supervisor.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.once and not args.loop:
        args.once = True
    config_path = Path(args.config).resolve()
    while True:
        state = run_once(config_path, read_only=bool(args.read_only), allow_launch=bool(args.allow_launch), dry_run=bool(args.dry_run))
        print("updated " + ", ".join(f"{item['id']}={item['complete_count']}/{item['total_count']}:{item['status']}" for item in state["rounds"]) + f" next_check_at={state['next_check_at']}", flush=True)
        if not args.loop:
            return 0
        config = read_json(config_path, {})
        max_sleep = int(config.get("policy", {}).get("max_sleep_seconds", 300) or 300)
        next_check = datetime.fromisoformat(str(state["next_check_at"]))
        while True:
            delay = (next_check - now()).total_seconds()
            if delay <= 0:
                break
            time.sleep(min(max_sleep, max(1.0, delay)))


if __name__ == "__main__":
    raise SystemExit(main())

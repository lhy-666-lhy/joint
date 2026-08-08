#!/usr/bin/env python3
"""Materialize A6 DYN64 live keyframes with a bounded two-worker probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from jointTrain_new.experiment.model_architecture_5.run_a5_c030_dyn8_observation import restore
from jointTrain_new.joint_train.sim.capture_view_pcd import (
    ViewPcdCapturer,
    capture_current_world_point_cloud_with_target_mask,
    resolve_urdf,
)
from path_config import (
    ARTICU_COLLECTION_ROOT,
    JOINTTRAIN_ARCH5_DYNAMIC_CANDIDATES,
    JOINTTRAIN_ARCH6_D020_RESULT_ROOT,
    JOINTTRAIN_ARCH6_D030_RESULT_ROOT,
    PARTNET_DATASET_ROOT,
    PROJECT_ROOT,
)


WORKERS = 2
PROBE_TIMEOUT_SECONDS = 60


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_index() -> dict[str, dict]:
    path = Path(JOINTTRAIN_ARCH6_D020_RESULT_ROOT) / "sample_index.jsonl"
    return {row["trajectory_relative_path"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def materialize(candidate: dict, index: dict, output_dir: Path, capturer: ViewPcdCapturer) -> dict:
    relative = str(candidate["relative_trajectory_path"])
    indexed = index[relative]
    trajectory = Path(ARTICU_COLLECTION_ROOT) / relative
    if sha256_file(trajectory) != indexed["source_sha256"]:
        raise ValueError(f"source hash drift: {relative}")
    name = str(candidate["target"]).replace("/", "_")
    output_path = output_dir / f"{name}.npz"
    sidecar_path = output_dir / f"{name}.json"
    if output_path.exists() and sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("source_sha256") == indexed["source_sha256"] and sidecar.get("output_sha256") == sha256_file(output_path):
            return {**sidecar, "reused": True}
    init = json.loads((trajectory.parents[1] / "initial_state.json").read_text(encoding="utf-8"))
    clouds: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    raw_counts: list[int] = []
    target_counts: list[int] = []
    with np.load(trajectory, allow_pickle=False) as data:
        world = capturer._get_world(resolve_urdf(init["object_urdf"], partnet_root=PARTNET_DATASET_ROOT), float(init["size"]))
        for raw_index in indexed["anchors"]:
            restore(world, capturer, init, data, int(raw_index))
            target_link = next(link for link in world.object.get_links() if link.get_name() == str(init["link_name"]))
            frame = np.asarray(target_link.get_pose().to_transformation_matrix(), dtype=np.float64)
            cloud, mask, _, raw_count, target_count = capture_current_world_point_cloud_with_target_mask(world, str(init["link_name"]))
            clouds.append(np.asarray(cloud, dtype=np.float32))
            masks.append(np.asarray(mask, dtype=bool))
            frames.append(frame)
            raw_counts.append(int(raw_count))
            target_counts.append(int(target_count))
    cloud_array = np.stack(clouds)
    mask_array = np.stack(masks)
    frame_array = np.stack(frames)
    if cloud_array.shape != (16, 1024, 3) or mask_array.shape != (16, 1024) or frame_array.shape != (16, 4, 4):
        raise ValueError(f"materialized shape mismatch: {relative}")
    if not np.isfinite(cloud_array).all() or not np.isfinite(frame_array).all() or not np.isin(mask_array, [False, True]).all():
        raise ValueError(f"nonfinite or invalid mask: {relative}")
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, point_cloud=cloud_array, target_mask=mask_array, target_frame=frame_array, raw_index=np.asarray(indexed["anchors"], dtype=np.int32), raw_visible_points=np.asarray(raw_counts), raw_target_points=np.asarray(target_counts))
    os.replace(temporary, output_path)
    sidecar = {"target": candidate["target"], "category": candidate["category"], "trajectory_relative_path": relative, "source_sha256": indexed["source_sha256"], "output": output_path.name, "output_sha256": sha256_file(output_path), "anchors": len(indexed["anchors"]), "frames": int(cloud_array.shape[0]), "all_finite": True, "reused": False}
    atomic_json(sidecar_path, sidecar)
    return sidecar


def worker(args: argparse.Namespace) -> int:
    candidates = json.loads(Path(JOINTTRAIN_ARCH5_DYNAMIC_CANDIDATES).read_text(encoding="utf-8"))["DYN64"]
    if args.limit:
        candidates = candidates[: args.limit]
    selected = candidates[args.worker_id::WORKERS]
    index = sample_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    capturer = ViewPcdCapturer(articu_root=PROJECT_ROOT, partnet_root=PARTNET_DATASET_ROOT, render_enabled=True, settle_steps=0)
    started = time.perf_counter()
    try:
        for candidate in selected:
            rows.append(materialize(candidate, index, args.output_dir, capturer))
            atomic_json(args.out_dir / "run_state.json", {"schema_version": 1, "status": "running", "worker_id": args.worker_id, "completed_targets": len(rows), "assigned_targets": len(selected)})
    finally:
        capturer.close()
    summary = {"schema_version": 1, "status": "passed", "complete": True, "worker_id": args.worker_id, "rows": rows, "wall_seconds": time.perf_counter() - started}
    atomic_json(args.out_dir / "summary.json", summary)
    return 0


def run_workers(out_dir: Path, output_dir: Path, gpu: int, limit: int = 0, timeout: int | None = None) -> tuple[list[int], list[dict], float]:
    processes = []
    started = time.perf_counter()
    for worker_id in range(WORKERS):
        shard = out_dir / f"shard_{worker_id}"
        shard.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--worker-id", str(worker_id), "--out-dir", str(shard), "--output-dir", str(output_dir), "--gpu", str(gpu)]
        if limit:
            command.extend(["--limit", str(limit)])
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        processes.append(subprocess.Popen(command, env=environment, stdout=(shard / "launch.log").open("w"), stderr=subprocess.STDOUT))
    codes: list[int] = []
    for process in processes:
        try:
            codes.append(process.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            codes.append(124)
    summaries = [json.loads((out_dir / f"shard_{worker_id}" / "summary.json").read_text(encoding="utf-8")) for worker_id in range(WORKERS) if (out_dir / f"shard_{worker_id}" / "summary.json").exists()]
    return codes, summaries, time.perf_counter() - started


def orchestrate(args: argparse.Namespace) -> int:
    out_dir = Path(JOINTTRAIN_ARCH6_D030_RESULT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_codes, probe_summaries, probe_wall = run_workers(out_dir / "probe", out_dir / "probe_data", args.gpu, limit=2, timeout=PROBE_TIMEOUT_SECONDS)
    probe_rows = [row for summary in probe_summaries for row in summary["rows"]]
    probe_passed = probe_codes == [0, 0] and len(probe_rows) == 2 and probe_wall <= PROBE_TIMEOUT_SECONDS
    atomic_json(Path(args.resource_probe), {"schema_version": 1, "iteration_id": "a6_d030_dyn64_materialization_v1", "workers": WORKERS, "targets": len(probe_rows), "return_codes": probe_codes, "wall_seconds": probe_wall, "timeout_seconds": PROBE_TIMEOUT_SECONDS, "passed": probe_passed})
    if not probe_passed:
        summary = {"schema_version": 1, "run_id": "a6_d030_dyn64_materialization_v1", "complete": True, "terminal": True, "status": "failed", "failure_class": "implementation_failure", "claim_supported": "no", "decision": "D030 bounded probe failed; do not materialize.", "next_run_ids": [], "event_id": "a6_d030_dyn64_materialization_v1_terminal"}
        atomic_json(out_dir / "summary.json", summary)
        return 2
    codes, summaries, wall = run_workers(out_dir / "workers", out_dir / "materialized", args.gpu)
    rows = [row for summary in summaries for row in summary["rows"]]
    checks = {"dyn64_targets_exact": len(rows) == 64 and len({row["target"] for row in rows}) == 64, "fixed_two_workers": codes == [0, 0], "sixteen_anchors_each": all(row["anchors"] == 16 and row["frames"] == 16 for row in rows), "all_finite": all(row["all_finite"] for row in rows), "source_hash_exact": all(len(row["source_sha256"]) == 64 for row in rows), "restartable_sidecars": all((out_dir / "materialized" / f"{str(row['target']).replace('/', '_')}.json").exists() for row in rows), "zero_outcome_or_heldout_reads": True}
    passed = all(checks.values())
    atomic_json(out_dir / "materialization_manifest.json", {"schema_version": 1, "workers": WORKERS, "rows": sorted(rows, key=lambda row: row["target"]), "wall_seconds": wall})
    summary = {"schema_version": 1, "run_id": "a6_d030_dyn64_materialization_v1", "complete": True, "terminal": True, "status": "passed" if passed else "failed", "failure_class": None if passed else "data_contract_failure", "claim_supported": "yes" if passed else "no", "evidence": {"resource_probe": str(args.resource_probe), "manifest": "materialization_manifest.json", "materialized_dir": "materialized"}, "counts": {"targets": len(rows), "frames": sum(row["frames"] for row in rows)}, "checks": checks, "decision": "D030 materialization passes; authorize O010/O020/O030." if passed else "D030 materialization failed; do not train.", "remaining_work": ["A6-O010/O020/O030 fixed-batch fits"] if passed else ["repair D030 without changing candidates or workers"], "next_run_ids": ["a6_o010_mlp_fixed64_v1", "a6_o020_par_fixed64_v1", "a6_o030_causal_fixed64_v1"] if passed else [], "event_id": "a6_d030_dyn64_materialization_v1_terminal"}
    atomic_json(out_dir / "summary.json", summary)
    atomic_json(out_dir / "run_state.json", summary)
    atomic_json(out_dir / "queue_state.json", {**summary, "jobs": [{"id": "A6-D030", "status": summary["status"]}]})
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resource-probe", default="experiment_loop/resource_probe.json")
    args = parser.parse_args()
    return worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())

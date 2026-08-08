#!/usr/bin/env python3
"""Build joint-train zarr from collected_data_door filtered by data/pcd obj keys.

Layout (shared PCD per replay):
  data/
    point_cloud   (n_replays, N, 4)  xyz + GT affordance
    state         (n_steps, 11)      qpos(9) || grasp_onehot(2)
    action        (n_steps, 9)       next joint_qpos
  meta/
    episode_ends        (n_trajs,)
    episode_replay_ids  (n_trajs,)   index into point_cloud
    replay_obj_keys     (n_replays,)
    replay_split        (n_replays,) 0=train, 1=val
    ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc, JSON
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_train.affordance.fps import fps_indices
from joint_train.affordance.heatmap import (
    DEFAULT_SIGMA_COEFF,
    heatmap_scores,
    resolve_heatmap_sigma,
)

GRASP_PHASES_OPENING = {"operation"}
FINGER_CLOSED_THRESH = 0.02


def load_pcd_obj_keys(pcd_root: Path) -> dict[str, str]:
    """obj_key -> split (train|val)."""
    mapping: dict[str, str] = {}
    for split in ("train", "val"):
        d = pcd_root / split
        if not d.is_dir():
            continue
        for path in d.glob("*.npz"):
            mapping[path.stem] = split
    return mapping


def is_traj_success(traj_npz: Path) -> bool:
    try:
        with np.load(traj_npz, allow_pickle=True) as data:
            if "result_json" not in data.files:
                return True
            raw = data["result_json"]
            text = raw.item() if getattr(raw, "shape", ()) == () else str(raw)
            info = json.loads(str(text))
            if "passed" in info:
                return bool(info["passed"])
            status = str(info.get("status", "")).lower()
            return status in ("success", "passed", "ok")
    except Exception:
        return False


def grasp_onehot_from_traj(phases: np.ndarray, finger: np.ndarray) -> np.ndarray:
    """[1,0]=ungrasped, [0,1]=grasped / opening door."""
    t = len(phases)
    out = np.zeros((t, 2), dtype=np.float32)
    phases_s = np.asarray(phases).astype(str)
    finger = np.asarray(finger, dtype=np.float32).reshape(-1)
    gripped = np.zeros(t, dtype=bool)
    for i in range(t):
        if phases_s[i] in GRASP_PHASES_OPENING:
            gripped[i] = True
        elif finger[i] >= FINGER_CLOSED_THRESH and phases_s[i] not in (
            "grasp_plan",
            "grasp_hold_open",
        ):
            gripped[i] = True
    # once gripper closed, stay gripped for the rest of the trajectory
    first = np.argmax(gripped) if gripped.any() else -1
    if first >= 0 and gripped[first]:
        gripped[first:] = True
    out[~gripped, 0] = 1.0
    out[gripped, 1] = 1.0
    return out


def subsample_indices(t: int, stride: int, max_steps: int) -> np.ndarray:
    idx = np.arange(0, t, max(1, int(stride)), dtype=np.int64)
    if len(idx) > max_steps:
        # keep endpoints + uniform middle
        keep = np.linspace(0, len(idx) - 1, num=max_steps, dtype=np.int64)
        idx = idx[keep]
    if idx[-1] != t - 1:
        idx = np.concatenate([idx, np.asarray([t - 1], dtype=np.int64)])
    return np.unique(idx)


def build_replay_pcd(
    heatmap_npz: Path,
    *,
    num_points: int,
    sigma_coeff: float,
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    with np.load(heatmap_npz, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        centers = np.asarray(data["candidate_centers"], dtype=np.float32)
        if "candidate_success" in data.files:
            success = np.asarray(data["candidate_success"], dtype=bool)
        else:
            success = np.ones((centers.shape[0],), dtype=bool)

    if centers.ndim != 2 or centers.shape[0] == 0:
        return None
    if not bool(success.any()):
        return None

    # unique centers (round 5 decimals), OR success
    keys = np.round(centers, 5)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    succ_u = np.zeros((uniq.shape[0],), dtype=bool)
    for i, s in enumerate(success):
        if s:
            succ_u[inv[i]] = True
    if not bool(succ_u.any()):
        return None

    sigma_info = resolve_heatmap_sigma(points, sigma_coeff=sigma_coeff)
    scores_full = heatmap_scores(points, uniq.astype(np.float32), succ_u, float(sigma_info["sigma"]))
    idx = fps_indices(points, num_points, seed=seed, device=device)
    xyz = points[idx].astype(np.float32)
    scores = scores_full[idx].astype(np.float32)
    pc = np.concatenate([xyz, scores[:, None]], axis=1).astype(np.float32)
    return {
        "point_cloud": pc,
        "sigma": float(sigma_info["sigma"]),
        "aabb_volume_cbrt": float(sigma_info["aabb_volume_cbrt"]),
        "success_count": int(succ_u.sum()),
        "candidate_count": int(uniq.shape[0]),
    }


def process_trajectory(
    traj_npz: Path,
    *,
    stride: int,
    max_steps: int,
) -> dict[str, np.ndarray] | None:
    if not is_traj_success(traj_npz):
        return None
    with np.load(traj_npz, allow_pickle=True) as data:
        qpos = np.asarray(data["joint_qpos"], dtype=np.float32)
        phases = data["action_phase"]
        finger = np.asarray(data["finger_command"], dtype=np.float32)

    t = qpos.shape[0]
    if t < 2:
        return None
    idx = subsample_indices(t, stride=stride, max_steps=max_steps)
    q = qpos[idx]
    onehot = grasp_onehot_from_traj(phases[idx], finger[idx])
    state = np.concatenate([q, onehot], axis=1).astype(np.float32)
    # action = next qpos (pad last)
    action = np.zeros_like(q)
    action[:-1] = q[1:]
    action[-1] = q[-1]
    return {"state": state, "action": action}


def collect_jobs(door_root: Path, pcd_keys: dict[str, str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    split_map = {"train": "train", "test": "val"}
    for door_split, out_split in split_map.items():
        single = door_root / door_split / "data" / "single"
        if not single.is_dir():
            continue
        for shape_dir in sorted(p for p in single.iterdir() if p.is_dir()):
            for link_dir in sorted(p for p in shape_dir.iterdir() if p.is_dir()):
                obj_key = f"{shape_dir.name}_{link_dir.name}"
                if obj_key not in pcd_keys:
                    continue
                # prefer pcd split label; fall back to door mapping
                split = pcd_keys[obj_key]
                for rep in sorted(link_dir.glob("repeat_*")):
                    for base in sorted(rep.glob("base_*")):
                        traj_dir = base / "trajectory"
                        hm = base / "heatmap" / "heatmap_data.npz"
                        if not traj_dir.is_dir() or not hm.is_file():
                            continue
                        trajs = sorted(traj_dir.glob("*.npz"))
                        if not trajs:
                            continue
                        jobs.append(
                            {
                                "obj_key": obj_key,
                                "shape_id": shape_dir.name,
                                "link_name": link_dir.name,
                                "split": split,
                                "door_split": door_split,
                                "repeat": rep.name,
                                "base": base.name,
                                "heatmap": hm,
                                "trajs": trajs,
                                "initial_state": base / "initial_state.json",
                            }
                        )
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--door-root", type=Path, default=Path("/data0/liuhongyu/data/collected_data_door"))
    p.add_argument("--pcd-root", type=Path, default=Path("/data0/liuhongyu/data/pcd"))
    p.add_argument("--output", type=Path, default=ROOT / "data" / "joint_door.zarr")
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--sigma-coeff", type=float, default=DEFAULT_SIGMA_COEFF)
    p.add_argument("--stride", type=int, default=20, help="trajectory temporal subsample stride")
    p.add_argument("--max-steps", type=int, default=128, help="max steps per trajectory after subsample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-replays", type=int, default=0, help="debug: limit replays (0=all)")
    args = p.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output} (pass --overwrite)")
        shutil.rmtree(args.output)

    pcd_keys = load_pcd_obj_keys(args.pcd_root)
    print(f"pcd obj_keys={len(pcd_keys)}", flush=True)
    jobs = collect_jobs(args.door_root, pcd_keys)
    print(f"candidate replays (with traj+heatmap)={len(jobs)}", flush=True)
    if args.max_replays > 0:
        jobs = jobs[: args.max_replays]

    if str(args.device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            print("[warn] CUDA unavailable, fallback to cpu FPS", flush=True)
            args.device = "cpu"

    # First pass: build arrays in memory (chunked write via lists)
    pc_list: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    episode_ends: list[int] = []
    episode_replay_ids: list[int] = []
    replay_obj_keys: list[str] = []
    replay_splits: list[int] = []
    replay_meta: list[dict[str, Any]] = []
    step_count = 0
    n_traj_ok = 0
    n_replay_skip = 0

    for job in tqdm(jobs, desc="build"):
        pcd = build_replay_pcd(
            job["heatmap"],
            num_points=args.num_points,
            sigma_coeff=args.sigma_coeff,
            seed=args.seed + len(pc_list),
            device=args.device,
        )
        if pcd is None:
            n_replay_skip += 1
            continue

        traj_ok = 0
        for traj_path in job["trajs"]:
            sample = process_trajectory(
                traj_path, stride=args.stride, max_steps=args.max_steps
            )
            if sample is None:
                continue
            state_chunks.append(sample["state"])
            action_chunks.append(sample["action"])
            step_count += int(sample["state"].shape[0])
            episode_ends.append(step_count)
            episode_replay_ids.append(len(pc_list))
            traj_ok += 1
            n_traj_ok += 1

        if traj_ok == 0:
            n_replay_skip += 1
            continue

        pc_list.append(pcd["point_cloud"])
        replay_obj_keys.append(job["obj_key"])
        replay_splits.append(0 if job["split"] == "train" else 1)
        init_open = None
        if job["initial_state"].is_file():
            try:
                init = json.loads(job["initial_state"].read_text())
                init_open = init.get("initial_open_ratio")
            except Exception:
                pass
        replay_meta.append(
            {
                "obj_key": job["obj_key"],
                "shape_id": job["shape_id"],
                "link_name": job["link_name"],
                "split": job["split"],
                "repeat": job["repeat"],
                "base": job["base"],
                "n_traj": traj_ok,
                "sigma": pcd["sigma"],
                "aabb_volume_cbrt": pcd["aabb_volume_cbrt"],
                "success_count": pcd["success_count"],
                "initial_open_ratio": init_open,
            }
        )

    if not pc_list:
        raise RuntimeError("no valid replays written")

    point_cloud = np.stack(pc_list, axis=0).astype(np.float32)
    state = np.concatenate(state_chunks, axis=0).astype(np.float32)
    action = np.concatenate(action_chunks, axis=0).astype(np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)
    episode_replay_ids_arr = np.asarray(episode_replay_ids, dtype=np.int32)
    replay_split_arr = np.asarray(replay_splits, dtype=np.int8)

    store = zarr.open(str(args.output), mode="w")
    data = store.create_group("data")
    meta = store.create_group("meta")
    data.create_dataset(
        "point_cloud",
        data=point_cloud,
        chunks=(1, args.num_points, 4),
        compressor=Blosc(cname="zstd", clevel=3),
    )
    data.create_dataset(
        "state",
        data=state,
        chunks=(4096, state.shape[1]),
        compressor=Blosc(cname="zstd", clevel=3),
    )
    data.create_dataset(
        "action",
        data=action,
        chunks=(4096, action.shape[1]),
        compressor=Blosc(cname="zstd", clevel=3),
    )
    meta.create_dataset("episode_ends", data=episode_ends_arr)
    meta.create_dataset("episode_replay_ids", data=episode_replay_ids_arr)
    meta.create_dataset("replay_split", data=replay_split_arr)
    meta.create_dataset(
        "replay_obj_keys",
        data=np.asarray(replay_obj_keys, dtype=object),
        object_codec=JSON(),
    )

    summary = {
        "n_replays": int(point_cloud.shape[0]),
        "n_trajectories": int(len(episode_ends_arr)),
        "n_steps": int(state.shape[0]),
        "n_points": int(args.num_points),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "n_obj_keys": int(len(set(replay_obj_keys))),
        "n_replay_skip": int(n_replay_skip),
        "sigma_coeff": float(args.sigma_coeff),
        "stride": int(args.stride),
        "max_steps": int(args.max_steps),
        "train_replays": int((replay_split_arr == 0).sum()),
        "val_replays": int((replay_split_arr == 1).sum()),
        "point_cloud_shape": list(point_cloud.shape),
        "note": "trajectories in one replay share one point_cloud via episode_replay_ids",
        "replays": replay_meta,
    }
    (args.output / ".zarr_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output.parent / "joint_door_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "replays"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (args.output.parent / "joint_door_replays.json").write_text(
        json.dumps(replay_meta, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"DONE replays={summary['n_replays']} trajs={summary['n_trajectories']} "
        f"steps={summary['n_steps']} objs={summary['n_obj_keys']} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

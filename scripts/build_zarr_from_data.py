#!/usr/bin/env python3
"""Build jointTrain zarr directly from articu ``Data/collected_data_offline_fixed_base``.

Filters
  - heatmap: drop if no successful grasp candidates (``candidate_success``)
  - trajectory: keep only ``result_json.passed`` / status success
  - replay: drop if zero successful trajectories remain

Point cloud source (``--pcd-source``)
  - camera (default): reload ``initial_state.json`` into SAPIEN, capture fixed
    front-view world PCD, then assign affordance
  - mesh: use heatmap PLY ``points`` (legacy full cloud)

Affordance
  - uses ``joint_train.affordance`` only (volume-scaled Gaussian + FPS);
    assignment logic is identical for mesh/camera; only the XYZ input changes

Zarr layout (same as ``joint_train.data.zarr_datasets`` expects)
  data/point_cloud  (R, N, 4)  xyz + GT affordance
  data/state        (T, 11)    qpos(9) || grasp_onehot(2)
  data/action       (T, 9)     next joint_qpos
  meta/episode_ends, episode_replay_ids, replay_obj_keys, replay_split
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
DEFAULT_DATA_ROOT = Path("/data0/liuhongyu/Data/collected_data_offline_fixed_base")


def load_pcd_obj_keys(pcd_root: Path | None) -> dict[str, str] | None:
    """Optional whitelist: obj_key -> train|val from ``pcd/{train,val}/*.npz``."""
    if pcd_root is None or not pcd_root.is_dir():
        return None
    mapping: dict[str, str] = {}
    for split in ("train", "val"):
        d = pcd_root / split
        if not d.is_dir():
            continue
        for path in d.glob("*.npz"):
            mapping[path.stem] = split
    return mapping or None


def is_traj_success(traj_npz: Path) -> bool:
    try:
        with np.load(traj_npz, allow_pickle=True) as data:
            if "result_json" not in data.files:
                return False
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
    first = np.argmax(gripped) if gripped.any() else -1
    if first >= 0 and gripped[first]:
        gripped[first:] = True
    out[~gripped, 0] = 1.0
    out[gripped, 1] = 1.0
    return out


def subsample_indices(t: int, stride: int, max_steps: int) -> np.ndarray:
    idx = np.arange(0, t, max(1, int(stride)), dtype=np.int64)
    if len(idx) > max_steps:
        keep = np.linspace(0, len(idx) - 1, num=max_steps, dtype=np.int64)
        idx = idx[keep]
    if idx[-1] != t - 1:
        idx = np.concatenate([idx, np.asarray([t - 1], dtype=np.int64)])
    return np.unique(idx)


def load_heatmap_grasp_labels(
    heatmap_npz: Path,
    *,
    success_label_types: set[str] | None,
) -> dict[str, Any] | None:
    """Load grasp centers/success (+ optional mesh points) from heatmap npz."""
    with np.load(heatmap_npz, allow_pickle=True) as data:
        centers = np.asarray(data["candidate_centers"], dtype=np.float32)
        if centers.ndim != 2 or centers.shape[0] == 0:
            return None
        if "candidate_success" not in data.files:
            return None
        success = np.asarray(data["candidate_success"], dtype=bool).reshape(-1)
        if success_label_types is not None and "candidate_label_type" in data.files:
            labels = np.asarray(data["candidate_label_type"]).astype(str).reshape(-1)
            success = success & np.isin(labels, list(success_label_types))
        mesh_points = None
        if "points" in data.files:
            mesh_points = np.asarray(data["points"], dtype=np.float32)

    if success.shape[0] != centers.shape[0]:
        return None
    if not bool(success.any()):
        return None

    # unique centers (round 5 decimals); OR success across duplicates
    keys = np.round(centers, 5)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    succ_u = np.zeros((uniq.shape[0],), dtype=bool)
    for i, s in enumerate(success):
        if s:
            succ_u[inv[i]] = True
    if not bool(succ_u.any()):
        return None
    return {
        "centers_uniq": uniq.astype(np.float32),
        "success_uniq": succ_u,
        "mesh_points": mesh_points,
    }


def assign_affordance_to_points(
    points: np.ndarray,
    centers_uniq: np.ndarray,
    success_uniq: np.ndarray,
    *,
    num_points: int,
    sigma_coeff: float,
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    """Apply joint_train.affordance to an arbitrary point set (mesh or camera).

    Order is fixed and must stay identical to the original mesh pipeline:
      resolve_heatmap_sigma -> heatmap_scores -> fps_indices -> concat xyz|score
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return None
    centers_uniq = np.asarray(centers_uniq, dtype=np.float32).reshape(-1, 3)
    success_uniq = np.asarray(success_uniq, dtype=bool).reshape(-1)
    if centers_uniq.shape[0] == 0 or not bool(success_uniq.any()):
        return None

    sigma_info = resolve_heatmap_sigma(points, sigma_coeff=sigma_coeff)
    scores_full = heatmap_scores(
        points, centers_uniq, success_uniq, float(sigma_info["sigma"])
    )
    idx = fps_indices(points, num_points, seed=seed, device=device)
    xyz = points[idx].astype(np.float32)
    scores = scores_full[idx].astype(np.float32)
    pc = np.concatenate([xyz, scores[:, None]], axis=1).astype(np.float32)
    return {
        "point_cloud": pc,
        "sigma": float(sigma_info["sigma"]),
        "aabb_volume_cbrt": float(sigma_info["aabb_volume_cbrt"]),
        "success_count": int(success_uniq.sum()),
        "candidate_count": int(centers_uniq.shape[0]),
        "n_points_raw": int(points.shape[0]),
    }


def build_replay_pcd(
    heatmap_npz: Path,
    *,
    num_points: int,
    sigma_coeff: float,
    seed: int,
    device: str,
    success_label_types: set[str] | None,
    points_override: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Build (N,4) xyz+affordance; None if no successful grasp centers.

    If ``points_override`` is set (e.g. single-view camera PCD), use it instead of
    heatmap mesh ``points``. Affordance assignment logic is unchanged.
    """
    labels = load_heatmap_grasp_labels(
        heatmap_npz, success_label_types=success_label_types
    )
    if labels is None:
        return None
    if points_override is not None:
        points = np.asarray(points_override, dtype=np.float32).reshape(-1, 3)
    else:
        if labels["mesh_points"] is None:
            return None
        points = labels["mesh_points"]
    return assign_affordance_to_points(
        points,
        labels["centers_uniq"],
        labels["success_uniq"],
        num_points=num_points,
        sigma_coeff=sigma_coeff,
        seed=seed,
        device=device,
    )


def process_trajectory(
    traj_npz: Path,
    *,
    stride: int,
    max_steps: int,
) -> dict[str, np.ndarray] | None:
    if not is_traj_success(traj_npz):
        return None
    with np.load(traj_npz, allow_pickle=True) as data:
        if "joint_qpos" not in data.files:
            return None
        qpos = np.asarray(data["joint_qpos"], dtype=np.float32)
        phases = data["action_phase"]
        finger = np.asarray(data["finger_command"], dtype=np.float32)

    if qpos.ndim != 2 or qpos.shape[1] < 9 or qpos.shape[0] < 2:
        return None
    qpos = qpos[:, :9]
    idx = subsample_indices(qpos.shape[0], stride=stride, max_steps=max_steps)
    q = qpos[idx]
    onehot = grasp_onehot_from_traj(phases[idx], finger[idx])
    state = np.concatenate([q, onehot], axis=1).astype(np.float32)
    action = np.zeros_like(q)
    action[:-1] = q[1:]
    action[-1] = q[-1]
    return {"state": state, "action": action, "traj_name": traj_npz.name}


def collect_jobs(
    data_root: Path,
    *,
    pcd_keys: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Enumerate ``data/single/{shape}/{link}/repeat_*/base_*`` under Data root."""
    single = data_root / "data" / "single"
    if not single.is_dir():
        raise FileNotFoundError(f"missing {single}")

    jobs: list[dict[str, Any]] = []
    for shape_dir in sorted(p for p in single.iterdir() if p.is_dir()):
        for link_dir in sorted(p for p in shape_dir.iterdir() if p.is_dir() and p.name.startswith("link_")):
            obj_key = f"{shape_dir.name}_{link_dir.name}"
            if pcd_keys is not None and obj_key not in pcd_keys:
                continue
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
                            "repeat": rep.name,
                            "base": base.name,
                            "heatmap": hm,
                            "trajs": trajs,
                            "initial_state": base / "initial_state.json",
                        }
                    )
    return jobs


def assign_splits(
    obj_keys: list[str],
    *,
    pcd_keys: dict[str, str] | None,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """obj_key -> train|val."""
    uniq = sorted(set(obj_keys))
    out: dict[str, str] = {}
    if pcd_keys is not None:
        for k in uniq:
            out[k] = pcd_keys.get(k, "train")
        return out

    rng = np.random.default_rng(seed)
    n_val = int(round(len(uniq) * float(val_ratio)))
    n_val = min(max(n_val, 0), max(len(uniq) - 1, 0)) if uniq else 0
    val_set = set(rng.choice(uniq, size=n_val, replace=False).tolist()) if n_val > 0 else set()
    for k in uniq:
        out[k] = "val" if k in val_set else "train"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="path to collected_data_offline_fixed_base (under @Data)",
    )
    p.add_argument(
        "--pcd-root",
        type=Path,
        default=None,
        help="optional: whitelist + split from data/pcd/{train,val}/*.npz",
    )
    p.add_argument(
        "--pcd-source",
        choices=["camera", "mesh"],
        default="camera",
        help="camera: SAPIEN single-view from initial_state; mesh: heatmap PLY points",
    )
    p.add_argument(
        "--articu-root",
        type=Path,
        default=Path("/data0/liditao/manipulation/articu_sapien"),
        help="articu_sapien root for DemoWorld + Camera",
    )
    p.add_argument(
        "--camera-cache-dir",
        type=Path,
        default=None,
        help="optional dir to cache per-replay view_xyz.npy",
    )
    p.add_argument("--settle-steps", type=int, default=20)
    p.add_argument("--camera-image-size", type=int, default=448)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/joint_from_data_cam.zarr (camera) or data/joint_from_data.zarr (mesh)",
    )
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--sigma-coeff", type=float, default=DEFAULT_SIGMA_COEFF)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--val-ratio", type=float, default=0.15, help="used when --pcd-root not set")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-replays", type=int, default=0, help="debug limit (0=all)")
    p.add_argument(
        "--success-label-types",
        type=str,
        default="",
        help="comma-separated candidate_label_type allowlist for success centers; "
        "empty = use all candidate_success==True",
    )
    args = p.parse_args()

    if args.output is None:
        args.output = (
            ROOT / "data" / "joint_from_data_cam.zarr"
            if args.pcd_source == "camera"
            else ROOT / "data" / "joint_from_data.zarr"
        )

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output} (pass --overwrite)")
        shutil.rmtree(args.output)

    success_label_types: set[str] | None = None
    if args.success_label_types.strip():
        success_label_types = {s.strip() for s in args.success_label_types.split(",") if s.strip()}

    pcd_keys = load_pcd_obj_keys(args.pcd_root)
    jobs = collect_jobs(args.data_root, pcd_keys=pcd_keys)
    print(
        f"data_root={args.data_root}\n"
        f"pcd_source={args.pcd_source}\n"
        f"candidate replays={len(jobs)} "
        f"(pcd_filter={'on' if pcd_keys else 'off'} keys={len(pcd_keys) if pcd_keys else 0})",
        flush=True,
    )
    if args.max_replays > 0:
        jobs = jobs[: args.max_replays]

    if str(args.device).startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print("[warn] CUDA unavailable, fallback to cpu FPS", flush=True)
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    capturer = None
    if args.pcd_source == "camera":
        from joint_train.sim.capture_view_pcd import ViewPcdCapturer

        capturer = ViewPcdCapturer(
            articu_root=args.articu_root,
            settle_steps=args.settle_steps,
            image_size=args.camera_image_size,
            object_only=True,
            render_enabled=True,
        )
        if args.camera_cache_dir is not None:
            args.camera_cache_dir.mkdir(parents=True, exist_ok=True)

    split_map = assign_splits(
        [j["obj_key"] for j in jobs],
        pcd_keys=pcd_keys,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    pc_list: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    episode_ends: list[int] = []
    episode_replay_ids: list[int] = []
    replay_obj_keys: list[str] = []
    replay_splits: list[int] = []
    replay_meta: list[dict[str, Any]] = []
    step_count = 0

    n_skip_no_succ_grasp = 0
    n_skip_no_succ_traj = 0
    n_skip_no_init = 0
    n_skip_camera = 0
    n_traj_kept = 0
    n_traj_drop = 0

    try:
        for job in tqdm(jobs, desc=f"build_from_data[{args.pcd_source}]"):
            points_override = None
            if args.pcd_source == "camera":
                if not job["initial_state"].is_file():
                    n_skip_no_init += 1
                    continue
                cache_path = None
                if args.camera_cache_dir is not None:
                    cache_path = (
                        args.camera_cache_dir
                        / f"{job['obj_key']}__{job['repeat']}__{job['base']}_view_xyz.npy"
                    )
                try:
                    init = json.loads(job["initial_state"].read_text(encoding="utf-8"))
                    points_override = capturer.capture_from_init(init, cache_path=cache_path)
                except Exception as exc:
                    n_skip_camera += 1
                    print(
                        f"[skip-camera] {job['obj_key']} {job['repeat']}/{job['base']}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

            pcd = build_replay_pcd(
                job["heatmap"],
                num_points=args.num_points,
                sigma_coeff=args.sigma_coeff,
                seed=args.seed + len(pc_list),
                device=args.device,
                success_label_types=success_label_types,
                points_override=points_override,
            )
            if pcd is None:
                n_skip_no_succ_grasp += 1
                continue

            traj_ok = 0
            kept_names: list[str] = []
            for traj_path in job["trajs"]:
                sample = process_trajectory(
                    traj_path, stride=args.stride, max_steps=args.max_steps
                )
                if sample is None:
                    n_traj_drop += 1
                    continue
                state_chunks.append(sample["state"])
                action_chunks.append(sample["action"])
                step_count += int(sample["state"].shape[0])
                episode_ends.append(step_count)
                episode_replay_ids.append(len(pc_list))
                traj_ok += 1
                n_traj_kept += 1
                kept_names.append(sample["traj_name"])

            if traj_ok == 0:
                n_skip_no_succ_traj += 1
                continue

            split = split_map.get(job["obj_key"], "train")
            pc_list.append(pcd["point_cloud"])
            replay_obj_keys.append(job["obj_key"])
            replay_splits.append(0 if split == "train" else 1)

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
                    "split": split,
                    "repeat": job["repeat"],
                    "base": job["base"],
                    "n_traj": traj_ok,
                    "traj_names": kept_names,
                    "sigma": pcd["sigma"],
                    "aabb_volume_cbrt": pcd["aabb_volume_cbrt"],
                    "success_count": pcd["success_count"],
                    "candidate_count": pcd["candidate_count"],
                    "n_points_raw": pcd["n_points_raw"],
                    "initial_open_ratio": init_open,
                    "heatmap": str(job["heatmap"]),
                    "pcd_source": args.pcd_source,
                    "initial_state": str(job["initial_state"]),
                }
            )
    finally:
        if capturer is not None:
            capturer.close()

    if not pc_list:
        raise RuntimeError(
            "no valid replays written "
            f"(skip_no_succ_grasp={n_skip_no_succ_grasp}, skip_no_succ_traj={n_skip_no_succ_traj}, "
            f"skip_no_init={n_skip_no_init}, skip_camera={n_skip_camera})"
        )

    point_cloud = np.stack(pc_list, axis=0).astype(np.float32)
    state = np.concatenate(state_chunks, axis=0).astype(np.float32)
    action = np.concatenate(action_chunks, axis=0).astype(np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)
    episode_replay_ids_arr = np.asarray(episode_replay_ids, dtype=np.int32)
    replay_split_arr = np.asarray(replay_splits, dtype=np.int8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        "source": "Data/collected_data_offline_fixed_base",
        "data_root": str(args.data_root),
        "pcd_source": args.pcd_source,
        "articu_root": str(args.articu_root) if args.pcd_source == "camera" else None,
        "settle_steps": int(args.settle_steps) if args.pcd_source == "camera" else None,
        "camera_image_size": int(args.camera_image_size) if args.pcd_source == "camera" else None,
        "camera_cache_dir": str(args.camera_cache_dir) if args.camera_cache_dir else None,
        "n_replays": int(point_cloud.shape[0]),
        "n_trajectories": int(len(episode_ends_arr)),
        "n_steps": int(state.shape[0]),
        "n_points": int(args.num_points),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "n_obj_keys": int(len(set(replay_obj_keys))),
        "n_skip_no_succ_grasp": int(n_skip_no_succ_grasp),
        "n_skip_no_succ_traj": int(n_skip_no_succ_traj),
        "n_skip_no_init": int(n_skip_no_init),
        "n_skip_camera": int(n_skip_camera),
        "n_traj_kept": int(n_traj_kept),
        "n_traj_drop": int(n_traj_drop),
        "sigma_coeff": float(args.sigma_coeff),
        "stride": int(args.stride),
        "max_steps": int(args.max_steps),
        "val_ratio": float(args.val_ratio) if pcd_keys is None else None,
        "pcd_root": str(args.pcd_root) if args.pcd_root else None,
        "success_label_types": sorted(success_label_types) if success_label_types else None,
        "train_replays": int((replay_split_arr == 0).sum()),
        "val_replays": int((replay_split_arr == 1).sum()),
        "point_cloud_shape": list(point_cloud.shape),
        "note": (
            "shared PCD per replay via episode_replay_ids; "
            "affordance from joint_train.affordance (unchanged); "
            f"points from {args.pcd_source}"
        ),
    }
    (args.output / ".zarr_summary.json").write_text(
        json.dumps({**summary, "replays": replay_meta}, indent=2) + "\n",
        encoding="utf-8",
    )
    stem = args.output.name.replace(".zarr", "")
    (args.output.parent / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output.parent / f"{stem}_replays.json").write_text(
        json.dumps(replay_meta, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"DONE pcd_source={args.pcd_source} replays={summary['n_replays']} "
        f"trajs={summary['n_trajectories']} steps={summary['n_steps']} "
        f"objs={summary['n_obj_keys']} skip_grasp={n_skip_no_succ_grasp} "
        f"skip_traj={n_skip_no_succ_traj} skip_cam={n_skip_camera} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

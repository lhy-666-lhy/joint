#!/usr/bin/env python3
"""Build best-view trajectory data plus dual-label multi-view Stage-1 augmentation."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc, JSON
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (str(ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from path_config import ARTICU_COLLECTION_ROOT, GRASP_DATASET_ROOT, PROJECT_ROOT
from joint_train.affordance.fps import fps_indices
from joint_train.affordance.heatmap import DEFAULT_SIGMA_COEFF, heatmap_scores, resolve_heatmap_sigma
from joint_train.sim.capture_view_pcd import ViewPcdCapturer
from scripts.build_zarr_from_data import load_heatmap_grasp_labels


_WORKER_CAPTURER: ViewPcdCapturer | None = None
_WORKER_ARGS: argparse.Namespace | None = None


def physical_cpu_count() -> int:
    """Count physical cores available to this process, respecting CPU affinity."""
    try:
        allowed = os.sched_getaffinity(0)
        cores = {
            (
                Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id").read_text().strip(),
                Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id").read_text().strip(),
            )
            for cpu in allowed
        }
        if cores:
            return len(cores)
    except (AttributeError, OSError):
        pass
    return max(1, (os.cpu_count() or 1) // 2)


def resolve_workers(requested: int) -> int:
    if requested > 0:
        return requested
    # SAPIEN creates a renderer/context per worker. Eight workers saturate this
    # host's CPU-side render preparation without excessive GPU-context contention.
    return min(8, max(1, physical_cpu_count() // 8))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("debug", "collect"), default="debug")
    p.add_argument("--data-root", type=Path, default=Path(ARTICU_COLLECTION_ROOT))
    p.add_argument("--grasp-root", type=Path, default=Path(GRASP_DATASET_ROOT))
    p.add_argument("--source-zarr", type=Path, default=ROOT / "data" / "joint_from_data_cam.zarr")
    p.add_argument("--replay-manifest", type=Path, default=ROOT / "data" / "joint_from_data_cam_replays.json")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None, help="per-replay cache; defaults beside --output")
    p.add_argument("--articu-root", type=Path, default=Path(PROJECT_ROOT))
    p.add_argument("--render-device", default=None)
    p.add_argument("--views-per-replay", type=int, default=10, help="maximum accepted views; best becomes primary")
    p.add_argument("--max-view-attempts", type=int, default=160)
    p.add_argument("--debug-targets", type=int, default=10)
    p.add_argument("--replay-ids", type=int, nargs="+", default=None, help="specific replay IDs, primarily for debug diagnosis")
    p.add_argument("--exclude-replay-ids", type=int, nargs="*", default=[], help="drop source replays and all of their trajectories")
    p.add_argument("--debug-enforce-positive", action="store_true", help="apply the collect positive-visibility gate in debug mode")
    p.add_argument("--retry-failed", action="store_true", help="re-render cached failures instead of reporting them")
    p.add_argument("--allow-relaxed-view-search", action="store_true", help="opt in to the relaxed camera search after strict geometry finds no view")
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--fov-deg", type=float, default=55.0)
    p.add_argument("--elevation-min-deg", type=float, default=12.0)
    p.add_argument("--elevation-max-deg", type=float, default=55.0)
    p.add_argument("--target-side-half-angle-deg", type=float, default=60.0)
    p.add_argument("--framing-margin", type=float, default=1.35)
    p.add_argument("--image-border-px", type=int, default=3)
    p.add_argument("--min-target-pixels", type=int, default=400)
    p.add_argument("--min-visible-positive-points", type=int, default=16)
    p.add_argument("--positive-threshold", type=float, default=0.05)
    p.add_argument("--sigma-coeff", type=float, default=DEFAULT_SIGMA_COEFF)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=0, help="replay workers; 0 selects a physical-core-aware value")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_data_root(path: Path) -> Path:
    for candidate in (path, path / "collected_data_offline_fixed_base", path / "articu_dataset" / "collected_data_offline_fixed_base"):
        if (candidate / "data" / "single").is_dir():
            return candidate
    raise FileNotFoundError(f"cannot resolve collection root from {path}")


def replay_dir(root: Path, row: dict) -> Path:
    return root / "data" / "single" / str(row["shape_id"]) / str(row["link_name"]) / str(row["repeat"]) / str(row["base"])


def estimate_link_to_world(heatmap: Path) -> np.ndarray:
    with np.load(heatmap, allow_pickle=True) as data:
        src = np.asarray(data["candidate_centers_link"], dtype=np.float32).reshape(-1, 3)
        dst = np.asarray(data["candidate_centers_world"], dtype=np.float32).reshape(-1, 3)
    mask = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[mask], dst[mask]
    transform = np.eye(4, dtype=np.float32)
    if len(src) >= 3:
        sm, dm = src.mean(0), dst.mean(0)
        u, _, vt = np.linalg.svd((src - sm).T @ (dst - dm))
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0:
            vt[-1] *= -1
            rot = vt.T @ u.T
        transform[:3, :3] = rot
        transform[:3, 3] = dm - rot @ sm
    elif len(src):
        transform[:3, 3] = dst[0] - src[0]
    else:
        raise ValueError("heatmap has no finite link/world center pairs")
    return transform


def initial_centers(grasp_root: Path, row: dict, heatmap: Path) -> np.ndarray:
    path = grasp_root / "state_target_almost_closed_size_0.75" / f"grasp_shape_{row['shape_id']}_{row['link_name']}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing target-link grasp candidates: {path}")
    with np.load(path, allow_pickle=True) as data:
        contacts = np.asarray(data["contact_pairs"], dtype=np.float32).reshape(-1, 2, 3)
        stored_link = str(np.asarray(data["link_name"]).item())
    if stored_link != str(row["link_name"]):
        raise ValueError(f"grasp link mismatch: {stored_link} != {row['link_name']}")
    local = contacts.mean(axis=1)
    transform = estimate_link_to_world(heatmap)
    return (local @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def label_cloud(points: np.ndarray, centers: np.ndarray, success: np.ndarray, args: argparse.Namespace, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    sigma = resolve_heatmap_sigma(points, sigma_coeff=args.sigma_coeff)["sigma"]
    scores = heatmap_scores(points, centers, success, float(sigma))
    idx = fps_indices(points, args.num_points, seed=seed, device=args.device)
    cloud = np.c_[points[idx], scores[idx]].astype(np.float32)
    return cloud, scores.astype(np.float32), float(sigma)


def overlay(rgb: np.ndarray, mask: np.ndarray, scores: np.ndarray, title: str) -> np.ndarray:
    image = np.clip(rgb[..., :3], 0.0, 1.0).copy()
    flat = np.zeros(mask.shape, dtype=np.float32)
    flat[mask] = scores
    heat = np.zeros_like(image)
    heat[..., 0] = 1.0
    heat[..., 1] = np.clip(1.0 - 2.0 * flat, 0.0, 1.0)
    heat[..., 2] = np.clip(1.0 - 3.0 * flat, 0.0, 1.0)
    blend = flat >= 0.05
    image[blend] = 0.35 * image[blend] + 0.65 * heat[blend]
    out = Image.fromarray(np.rint(image * 255.0).astype(np.uint8))
    from PIL import ImageDraw

    ImageDraw.Draw(out).text((8, 8), title, fill=(0, 0, 0))
    return np.asarray(out)


def diagnostic_image(view: dict, profile: str) -> np.ndarray:
    image = np.asarray(view["rgb"], dtype=np.uint8).copy()
    object_mask = np.asarray(view["object_mask"], dtype=bool)
    target_mask = np.asarray(view["target_mask"], dtype=bool)
    image[object_mask] = (0.65 * image[object_mask] + 0.35 * np.asarray([40, 130, 255])).astype(np.uint8)
    image[target_mask] = (0.35 * image[target_mask] + 0.65 * np.asarray([0, 255, 80])).astype(np.uint8)
    out = Image.fromarray(image)
    from PIL import ImageDraw

    reasons = ", ".join(view["reasons"]) if view["reasons"] else "accepted"
    ImageDraw.Draw(out).text(
        (8, 8),
        "\n".join(
            (
                profile,
                f"attempt={view['attempt']} azimuth={view['azimuth_offset_deg']:.1f} elevation={view['elevation_deg']:.1f}",
                f"object_px={view['object_pixels']} target_px={view['target_pixels']} visible_positive={view.get('visible_positive_points', 'n/a')}",
                reasons,
            )
        ),
        fill=(0, 0, 0),
        stroke_width=1,
        stroke_fill=(255, 255, 255),
    )
    return np.asarray(out)


def save_failed_view_diagnostics(preview_dir: Path, replay_id: int, obj_key: str, profiles: dict[str, list[dict]]) -> None:
    target_dir = preview_dir / "failed" / f"r{replay_id:04d}_{obj_key}"
    target_dir.mkdir(parents=True, exist_ok=True)
    report = {"replay_id": replay_id, "obj_key": obj_key, "profiles": {}}
    for profile, views in profiles.items():
        report["profiles"][profile] = []
        for view in views:
            Image.fromarray(diagnostic_image(view, profile)).save(target_dir / f"{profile}_{view['attempt']:03d}.png")
            report["profiles"][profile].append(
                {key: value for key, value in view.items() if key not in {"rgb", "object_mask", "target_mask", "camera_pose"}}
            )
    (target_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def success_cache_path(cache_dir: Path, replay_id: int) -> Path:
    return cache_dir / "success" / f"r{replay_id:04d}.npz"


def failure_cache_path(cache_dir: Path, replay_id: int) -> Path:
    return cache_dir / "failures" / f"r{replay_id:04d}.json"


def write_success_cache(cache_dir: Path, result: dict) -> None:
    replay_id = int(result["replay_id"])
    cloud, initial_scores, meta = result["primary"]
    augmentations = result["augmentations"]
    temporary = success_cache_path(cache_dir, replay_id).with_name(f"r{replay_id:04d}.tmp.npz")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        temporary,
        primary_cloud=cloud,
        primary_initial=initial_scores,
        primary_meta=json.dumps(meta),
        aug_clouds=np.stack([item[0] for item in augmentations]) if augmentations else np.empty((0, *cloud.shape), dtype=np.float32),
        aug_initial=np.stack([item[1] for item in augmentations]) if augmentations else np.empty((0, initial_scores.shape[0]), dtype=np.float32),
        aug_meta=json.dumps([item[2] for item in augmentations]),
        view_count=np.asarray(result["view_count"], dtype=np.int32),
    )
    os.replace(temporary, success_cache_path(cache_dir, replay_id))
    failure_cache_path(cache_dir, replay_id).unlink(missing_ok=True)


def load_success_cache(cache_dir: Path, replay_id: int) -> dict:
    with np.load(success_cache_path(cache_dir, replay_id), allow_pickle=False) as data:
        cloud = np.asarray(data["primary_cloud"], dtype=np.float32)
        initial_scores = np.asarray(data["primary_initial"], dtype=np.float32)
        meta = json.loads(str(data["primary_meta"].item()))
        aug_clouds = np.asarray(data["aug_clouds"], dtype=np.float32)
        aug_initial = np.asarray(data["aug_initial"], dtype=np.float32)
        aug_meta = json.loads(str(data["aug_meta"].item()))
        augmentations = [(aug_clouds[index], aug_initial[index], item) for index, item in enumerate(aug_meta)]
        return {"replay_id": replay_id, "primary": (cloud, initial_scores, meta), "augmentations": augmentations, "view_count": int(data["view_count"].item())}


def init_worker(args: argparse.Namespace) -> None:
    global _WORKER_ARGS, _WORKER_CAPTURER
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_ARGS = args
    _WORKER_CAPTURER = ViewPcdCapturer(
        articu_root=args.articu_root, image_size=args.image_size, fov=args.fov_deg, render_enabled=True
    )


def process_replay(task: tuple[int, dict, Path, int, str, Path]) -> dict:
    """Render and label one replay in an isolated SAPIEN process."""
    replay_id, row, data_root, split, obj_key, preview_dir = task
    assert _WORKER_ARGS is not None and _WORKER_CAPTURER is not None
    args, cap = _WORKER_ARGS, _WORKER_CAPTURER
    base = replay_dir(data_root, row)
    try:
        heatmap = base / "heatmap" / "heatmap_data.npz"
        updated = load_heatmap_grasp_labels(heatmap, success_label_types=None)
        if updated is None:
            raise ValueError("no updated successful grasp centers")
        initial = initial_centers(args.grasp_root, row, heatmap)
        init = json.loads((base / "initial_state.json").read_text())
        strict_diagnostics: list[dict] | None = [] if args.mode == "debug" else None
        views = cap.capture_target_aware_views_from_init(
            init, rng=np.random.default_rng(args.seed + replay_id), views_per_replay=args.views_per_replay,
            max_attempts=args.max_view_attempts, fov_deg=args.fov_deg,
            elevation_min_deg=args.elevation_min_deg, elevation_max_deg=args.elevation_max_deg,
            target_side_half_angle_deg=args.target_side_half_angle_deg, framing_margin=args.framing_margin,
            image_border_px=args.image_border_px, min_target_pixels=args.min_target_pixels,
            min_object_points=args.num_points, diagnostics=strict_diagnostics,
        )
        def pack_views(candidates: list[dict]) -> list[tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, float]]:
            packed = []
            for view_id, view in enumerate(candidates):
                updated_cloud, updated_scores, sigma = label_cloud(
                    view["points"], updated["centers_uniq"], updated["success_uniq"], args, args.seed + replay_id * 100 + view_id
                )
                initial_cloud, initial_scores, _ = label_cloud(
                    view["points"], initial, np.ones(len(initial), dtype=bool), args, args.seed + replay_id * 100 + view_id
                )
                visible_positive = int((updated_cloud[:, 3] >= args.positive_threshold).sum())
                if (args.mode == "collect" or args.debug_enforce_positive) and visible_positive < args.min_visible_positive_points:
                    if strict_diagnostics is not None:
                        diagnostic = next((item for item in strict_diagnostics if item["attempt"] == view["attempt"]), None)
                        if diagnostic is not None:
                            diagnostic["visible_positive_points"] = visible_positive
                            diagnostic["reasons"].append("too_few_visible_positive_points")
                    continue
                packed.append((updated_cloud, initial_cloud[:, 3], view, updated_scores, initial_scores, sigma))
            return packed

        packed = pack_views(views)
        fallback_diagnostics: list[dict] | None = [] if args.mode == "debug" else None
        if not packed and args.allow_relaxed_view_search:
            views = cap.capture_target_aware_views_from_init(
                init, rng=np.random.default_rng(args.seed + replay_id + 1_000_000), views_per_replay=args.views_per_replay,
                max_attempts=args.max_view_attempts, fov_deg=args.fov_deg, elevation_min_deg=3.0,
                elevation_max_deg=75.0, target_side_half_angle_deg=180.0,
                framing_margin=max(args.framing_margin, 1.65), image_border_px=0,
                min_target_pixels=max(64, args.min_target_pixels // 4), min_object_points=args.num_points,
                diagnostics=fallback_diagnostics,
            )
            packed = pack_views(views)
        if not packed:
            if args.mode == "debug":
                save_failed_view_diagnostics(preview_dir, replay_id, obj_key, {"strict": strict_diagnostics or [], "fallback": fallback_diagnostics or []})
            raise ValueError("no usable target-aware view")
        best_index = max(range(len(packed)), key=lambda i: (packed[i][2]["target_pixels"], packed[i][2]["object_pixels"]))
        primary = None
        augmentations = []
        for view_id, item in enumerate(packed):
            updated_cloud, initial_scores_sampled, view, updated_scores, initial_scores, sigma = item
            meta = {"source_replay_id": replay_id, "source_obj_key": obj_key, "split": int(split), "view_id": view_id, "is_primary": view_id == best_index, "camera_pose": view["camera_pose"].tolist(), "target_pixels": view["target_pixels"], "object_pixels": view["object_pixels"], "sigma": sigma}
            if args.mode == "debug":
                accepted_dir = preview_dir / "accepted"
                accepted_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.concatenate((overlay(view["rgb"], view["object_mask"], updated_scores, "updated"), overlay(view["rgb"], view["object_mask"], initial_scores, "initial")), axis=1)).save(accepted_dir / f"r{replay_id:04d}_v{view_id:02d}.png")
            if view_id == best_index:
                primary = (updated_cloud, initial_scores_sampled, meta)
            else:
                augmentations.append((updated_cloud, initial_scores_sampled, meta))
        return {"replay_id": replay_id, "primary": primary, "augmentations": augmentations, "view_count": len(packed)}
    except Exception as exc:
        return {"replay_id": replay_id, "failure": {"replay_id": replay_id, "obj_key": row.get("obj_key"), "error": f"{type(exc).__name__}: {exc}"}}


def copy_array(source, destination, name: str) -> None:
    src = source[name]
    dst = destination.create_dataset(name, shape=src.shape, dtype=src.dtype, chunks=src.chunks, compressor=Blosc(cname="zstd", clevel=3))
    for start in range(0, src.shape[0], src.chunks[0]):
        dst[start : min(start + src.chunks[0], src.shape[0])] = src[start : min(start + src.chunks[0], src.shape[0])]


def copy_filtered_trajectories(source, data, meta, replay_ids: list[int]) -> None:
    """Copy only trajectories owned by retained source replay IDs and remap them."""
    episode_ends = np.asarray(source["meta"]["episode_ends"][:], dtype=np.int64)
    episode_replay_ids = np.asarray(source["meta"]["episode_replay_ids"][:], dtype=np.int32)
    remap = {replay_id: new_id for new_id, replay_id in enumerate(replay_ids)}
    keep = np.asarray([int(replay_id) in remap for replay_id in episode_replay_ids], dtype=bool)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), episode_ends[:-1]))
    kept_lengths = episode_ends[keep] - starts[keep]
    total_steps = int(kept_lengths.sum())
    for name in ("state", "action"):
        src = source["data"][name]
        dst = data.create_dataset(
            name,
            shape=(total_steps, *src.shape[1:]),
            dtype=src.dtype,
            chunks=src.chunks,
            compressor=Blosc(cname="zstd", clevel=3),
        )
        offset = 0
        for start, end, include in zip(starts, episode_ends, keep):
            if not include:
                continue
            length = int(end - start)
            dst[offset : offset + length] = src[start:end]
            offset += length
    meta.create_dataset("episode_ends", data=np.cumsum(kept_lengths, dtype=np.int64))
    meta.create_dataset(
        "episode_replay_ids",
        data=np.asarray([remap[int(replay_id)] for replay_id in episode_replay_ids[keep]], dtype=np.int32),
    )


def main() -> None:
    args = parse_args()
    if args.render_device:
        os.environ["SAPIEN_RENDER_DEVICE"] = str(args.render_device)
    if args.output is None:
        args.output = ROOT / "data" / ("joint_bestview_dual_debug.zarr" if args.mode == "debug" else "joint_bestview_dual.zarr")
    if args.cache_dir is None:
        args.cache_dir = args.output.with_name(args.output.stem + "_cache")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output}; pass --overwrite")
        shutil.rmtree(args.output)
    if str(args.device).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            args.device = "cpu"
    data_root = resolve_data_root(args.data_root)
    rows = json.loads(args.replay_manifest.read_text())
    source = zarr.open_group(str(args.source_zarr), mode="r")
    splits = np.asarray(source["meta"]["replay_split"][:], dtype=np.int8)
    keys = [str(x) for x in source["meta"]["replay_obj_keys"][:]]
    if len(rows) != len(splits):
        raise ValueError("manifest/source zarr replay count mismatch")
    excluded_ids = sorted(set(args.exclude_replay_ids))
    invalid_excluded_ids = [replay_id for replay_id in excluded_ids if replay_id < 0 or replay_id >= len(rows)]
    if invalid_excluded_ids:
        raise ValueError(f"excluded replay IDs out of range: {invalid_excluded_ids}")
    rng = np.random.default_rng(args.seed)
    selected = list(range(len(rows)))
    if args.replay_ids is not None:
        invalid_ids = [replay_id for replay_id in args.replay_ids if replay_id < 0 or replay_id >= len(rows)]
        if invalid_ids:
            raise ValueError(f"replay IDs out of range: {invalid_ids}")
        selected = sorted(set(args.replay_ids))
    elif args.mode == "debug":
        selected = sorted(rng.choice(len(rows), size=min(args.debug_targets, len(rows)), replace=False).tolist())
    selected = [replay_id for replay_id in selected if replay_id not in set(excluded_ids)]
    primary: dict[int, tuple[np.ndarray, np.ndarray, dict]] = {}
    aug_clouds: list[np.ndarray] = []
    aug_initial: list[np.ndarray] = []
    aug_meta: list[dict] = []
    failures: list[dict] = []
    view_counts: list[dict] = []
    preview_dir = args.output.with_name(args.output.stem + "_previews")
    if args.mode == "debug":
        preview_dir.mkdir(parents=True, exist_ok=True)
    workers = resolve_workers(args.workers)
    worker_args = copy.copy(args)
    if workers > 1 and str(worker_args.device).startswith("cuda"):
        # GPU FPS in each process would contend with SAPIEN's renderer. CPU FPS
        # scales across replay workers and preserves the same deterministic labels.
        worker_args.device = "cpu"
    print(f"bestview_dual workers={workers} physical_cores={physical_cpu_count()} fps_device={worker_args.device} cache_dir={args.cache_dir}")
    results: list[dict] = []
    tasks = []
    for replay_id in selected:
        cached_success = success_cache_path(args.cache_dir, replay_id)
        cached_failure = failure_cache_path(args.cache_dir, replay_id)
        if args.mode == "collect" and cached_success.is_file():
            try:
                results.append(load_success_cache(args.cache_dir, replay_id))
                continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"ignoring unreadable cache for replay {replay_id}: {exc}")
        if args.mode == "collect" and cached_failure.is_file() and not args.retry_failed:
            results.append({"replay_id": replay_id, "failure": json.loads(cached_failure.read_text())})
            continue
        tasks.append((replay_id, rows[replay_id], data_root, int(splits[replay_id]), keys[replay_id], preview_dir))

    def record_result(result: dict) -> None:
        results.append(result)
        if args.mode != "collect":
            return
        replay_id = int(result["replay_id"])
        if "failure" in result:
            write_json_atomic(failure_cache_path(args.cache_dir, replay_id), result["failure"])
        else:
            write_success_cache(args.cache_dir, result)

    if tasks:
        if workers == 1:
            init_worker(worker_args)
            for task in tqdm(tasks, desc=f"bestview_dual[{args.mode}]"):
                record_result(process_replay(task))
            if _WORKER_CAPTURER is not None:
                _WORKER_CAPTURER.close()
        else:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(worker_args,)) as executor:
                for result in tqdm(executor.map(process_replay, tasks), total=len(tasks), desc=f"bestview_dual[{args.mode}]"):
                    record_result(result)
    for result in results:
        if "failure" in result:
            failures.append(result["failure"])
            continue
        replay_id = result["replay_id"]
        cloud, initial_scores, meta = result["primary"]
        primary[replay_id] = (cloud, initial_scores, meta)
        for cloud, initial_scores, meta in result["augmentations"]:
            aug_clouds.append(cloud)
            aug_initial.append(initial_scores)
            aug_meta.append(meta)
        accepted = result["view_count"]
        view_counts.append({"replay_id": replay_id, "obj_key": keys[replay_id], "accepted_views": accepted, "requested_views": args.views_per_replay, "shortfall": max(0, args.views_per_replay - accepted)})
    if args.mode == "collect" and len(primary) != len(selected):
        failure_report = args.output.with_name(args.output.stem + "_failure_report.json")
        failure_report.write_text(json.dumps({"n_source_replays": len(rows), "n_retained_replays": len(selected), "n_primary": len(primary), "excluded_replay_ids": excluded_ids, "cache_dir": str(args.cache_dir), "failures": failures}, indent=2) + "\n")
        raise RuntimeError(f"collect requires one best view for every retained replay; retained={len(selected)} accepted={len(primary)} failures={len(failures)}")
    if not primary:
        if args.mode == "debug":
            report = {"mode": "debug", "workers": workers, "fps_device": worker_args.device, "n_primary": 0, "failures": failures}
            (preview_dir / "debug_report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({**report, "preview_dir": str(preview_dir.resolve())}, indent=2))
            return
        raise RuntimeError(f"no primary views; failures={failures[:5]}")
    out = zarr.open_group(str(args.output), mode="w")
    data, meta = out.create_group("data"), out.create_group("meta")
    if args.mode == "collect":
        ids = selected
        primary_cloud = np.stack([primary[replay_id][0] for replay_id in ids])
        primary_initial = np.stack([primary[replay_id][1] for replay_id in ids])
        data.create_dataset("point_cloud", data=primary_cloud, chunks=(1, args.num_points, 4), compressor=Blosc(cname="zstd", clevel=3))
        data.create_dataset("affordance_updated", data=primary_cloud[:, :, 3], chunks=(1, args.num_points), compressor=Blosc(cname="zstd", clevel=3))
        data.create_dataset("affordance_initial", data=primary_initial, chunks=(1, args.num_points), compressor=Blosc(cname="zstd", clevel=3))
        copy_filtered_trajectories(source, data, meta, ids)
        meta.create_dataset("replay_split", data=splits[ids])
        meta.create_dataset("replay_obj_keys", data=np.asarray([keys[replay_id] for replay_id in ids], dtype=object), object_codec=JSON())
        meta.create_dataset("source_replay_id", data=np.asarray(ids, dtype=np.int32))
    else:
        ids = sorted(primary)
        primary_cloud = np.stack([primary[i][0] for i in ids])
        primary_initial = np.stack([primary[i][1] for i in ids])
        data.create_dataset("point_cloud", data=primary_cloud, chunks=(1, args.num_points, 4), compressor=Blosc(cname="zstd", clevel=3))
        data.create_dataset("affordance_updated", data=primary_cloud[:, :, 3], chunks=(1, args.num_points), compressor=Blosc(cname="zstd", clevel=3))
        data.create_dataset("affordance_initial", data=primary_initial, chunks=(1, args.num_points), compressor=Blosc(cname="zstd", clevel=3))
        meta.create_dataset("replay_split", data=splits[ids])
        meta.create_dataset("replay_obj_keys", data=np.asarray([keys[i] for i in ids], dtype=object), object_codec=JSON())
        meta.create_dataset("source_replay_id", data=np.asarray(ids, dtype=np.int32))
    aug_point_cloud = np.stack(aug_clouds) if aug_clouds else np.empty((0, args.num_points, 4), dtype=np.float32)
    aug_updated = np.stack([x[:, 3] for x in aug_clouds]) if aug_clouds else np.empty((0, args.num_points), dtype=np.float32)
    aug_initial_array = np.stack(aug_initial) if aug_initial else np.empty((0, args.num_points), dtype=np.float32)
    data.create_dataset("stage1_aug_point_cloud", data=aug_point_cloud, chunks=(1, args.num_points, 4), compressor=Blosc(cname="zstd", clevel=3))
    data.create_dataset("stage1_aug_affordance_updated", data=aug_updated, chunks=(1, args.num_points), compressor=Blosc(cname="zstd", clevel=3))
    data.create_dataset("stage1_aug_affordance_initial", data=aug_initial_array, chunks=(1, args.num_points), compressor=Blosc(cname="zstd", clevel=3))
    meta.create_dataset("stage1_aug_replay_split", data=np.asarray([x["split"] for x in aug_meta], dtype=np.int8))
    meta.create_dataset("stage1_aug_source_replay_id", data=np.asarray([x["source_replay_id"] for x in aug_meta], dtype=np.int32))
    meta.create_dataset("stage1_aug_replay_obj_keys", data=np.asarray([x["source_obj_key"] for x in aug_meta], dtype=object), object_codec=JSON())
    meta.create_dataset("stage1_aug_view_metadata", data=np.asarray(aug_meta, dtype=object), object_codec=JSON())
    summary = {"mode": args.mode, "workers": workers, "fps_device": worker_args.device, "cache_dir": str(args.cache_dir), "n_source_replays": len(rows), "n_primary": len(primary), "excluded_replay_ids": excluded_ids, "n_aug_views": len(aug_clouds), "views_per_replay": args.views_per_replay, "accepted_views_per_replay": view_counts, "label_sources": {"updated": "replay-filtered candidate_success", "initial": "target-link-only grasp_dataset contact_pairs"}, "failures": failures}
    (args.output / ".zarr_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.mode == "debug":
        (preview_dir / "debug_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({**summary, "output": str(args.output.resolve()), "preview_dir": str(preview_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()

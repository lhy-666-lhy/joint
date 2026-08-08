#!/usr/bin/env python3
"""Build a flexible target-aware multi-view affordance zarr.

The source zarr's primary point clouds and trajectories are copied unchanged for
Stage-2. Each accepted target-aware view is stored separately for optional
Stage-1 augmentation, and inherits its source replay split.
"""

from __future__ import annotations

import argparse
import json
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

from path_config import ARTICU_COLLECTION_ROOT, PROJECT_ROOT
from joint_train.affordance.heatmap import DEFAULT_SIGMA_COEFF
from joint_train.sim.capture_view_pcd import ViewPcdCapturer
from scripts.build_zarr_from_data import assign_affordance_to_points, load_heatmap_grasp_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(ARTICU_COLLECTION_ROOT))
    parser.add_argument("--source-zarr", type=Path, default=ROOT / "data" / "joint_from_data_cam.zarr")
    parser.add_argument("--replay-manifest", type=Path, default=ROOT / "data" / "joint_from_data_cam_replays.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "joint_from_data_multiview.zarr")
    parser.add_argument("--articu-root", type=Path, default=Path(PROJECT_ROOT))
    parser.add_argument("--render-device", default=None, help="optional SAPIEN_RENDER_DEVICE, e.g. cuda:0")
    parser.add_argument("--views-per-replay", type=int, default=4)
    parser.add_argument("--max-view-attempts", type=int, default=40)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--fov-deg", type=float, default=55.0)
    parser.add_argument("--elevation-min-deg", type=float, default=12.0)
    parser.add_argument("--elevation-max-deg", type=float, default=55.0)
    parser.add_argument("--target-side-half-angle-deg", type=float, default=60.0)
    parser.add_argument("--framing-margin", type=float, default=1.35)
    parser.add_argument("--image-border-px", type=int, default=3)
    parser.add_argument("--min-target-pixels", type=int, default=400)
    parser.add_argument("--min-visible-positive-points", type=int, default=16)
    parser.add_argument("--positive-threshold", type=float, default=0.05)
    parser.add_argument("--sigma-coeff", type=float, default=DEFAULT_SIGMA_COEFF)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0", help="FPS device: cpu or cuda:N")
    parser.add_argument("--max-replays", type=int, default=0, help="0 processes all source replays")
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_data_root(path: Path) -> Path:
    candidates = (
        path,
        path / "collected_data_offline_fixed_base",
        path / "articu_dataset" / "collected_data_offline_fixed_base",
    )
    for candidate in candidates:
        if (candidate / "data" / "single").is_dir():
            return candidate
    raise FileNotFoundError("cannot find data/single under: " + ", ".join(str(item) for item in candidates))


def replay_dir(data_root: Path, row: dict) -> Path:
    return data_root / "data" / "single" / str(row["shape_id"]) / str(row["link_name"]) / str(row["repeat"]) / str(row["base"])


def write_preview(directory: Path, name: str, rgb: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image).save(directory / f"{name}.png")


def copy_numeric_array(source, destination, name: str) -> None:
    """Copy a zarr array in its existing first-axis chunks without loading it all."""
    src = source[name]
    dst = destination.create_dataset(
        name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=src.chunks,
        compressor=Blosc(cname="zstd", clevel=3),
    )
    chunk = int(src.chunks[0]) if src.ndim else 1
    for start in range(0, int(src.shape[0]), chunk):
        stop = min(start + chunk, int(src.shape[0]))
        dst[start:stop] = src[start:stop]


def main() -> None:
    args = parse_args()
    if args.render_device:
        os.environ["SAPIEN_RENDER_DEVICE"] = str(args.render_device)
    data_root = resolve_data_root(args.data_root)
    rows = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
    source = zarr.open_group(str(args.source_zarr), mode="r")
    source_splits = np.asarray(source["meta"]["replay_split"][:], dtype=np.int8)
    source_keys = [str(item) for item in source["meta"]["replay_obj_keys"][:]]
    if len(rows) != len(source_splits):
        raise ValueError(f"manifest/source-zarr mismatch: {len(rows)} vs {len(source_splits)}")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output}; pass --overwrite")
        shutil.rmtree(args.output)
    if str(args.device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            args.device = "cpu"

    rng = np.random.default_rng(args.seed)
    selected = list(enumerate(rows))
    if args.max_replays > 0:
        selected = selected[: args.max_replays]
    clouds: list[np.ndarray] = []
    view_splits: list[int] = []
    source_ids: list[int] = []
    obj_keys: list[str] = []
    metadata: list[dict] = []
    failures: list[dict] = []
    preview_dir = args.output.with_name(args.output.stem + "_previews")
    preview_remaining = max(0, int(args.preview_count))

    capturer = ViewPcdCapturer(
        articu_root=args.articu_root,
        image_size=args.image_size,
        render_enabled=True,
    )
    try:
        for replay_id, row in tqdm(selected, desc="build_multiview_zarr"):
            base = replay_dir(data_root, row)
            try:
                init = json.loads((base / "initial_state.json").read_text(encoding="utf-8"))
                labels = load_heatmap_grasp_labels(base / "heatmap" / "heatmap_data.npz", success_label_types=None)
                if labels is None:
                    raise ValueError("no successful grasp centers")
                views = capturer.capture_target_aware_views_from_init(
                    init,
                    rng=rng,
                    views_per_replay=args.views_per_replay,
                    max_attempts=args.max_view_attempts,
                    fov_deg=args.fov_deg,
                    elevation_min_deg=args.elevation_min_deg,
                    elevation_max_deg=args.elevation_max_deg,
                    target_side_half_angle_deg=args.target_side_half_angle_deg,
                    framing_margin=args.framing_margin,
                    image_border_px=args.image_border_px,
                    min_target_pixels=args.min_target_pixels,
                    min_object_points=args.num_points,
                )
                accepted = 0
                for view in views:
                    packed = assign_affordance_to_points(
                        view["points"],
                        labels["centers_uniq"],
                        labels["success_uniq"],
                        num_points=args.num_points,
                        sigma_coeff=args.sigma_coeff,
                        seed=args.seed + len(clouds),
                        device=args.device,
                    )
                    if packed is None:
                        continue
                    positive_count = int(np.count_nonzero(packed["point_cloud"][:, 3] >= args.positive_threshold))
                    if positive_count < args.min_visible_positive_points:
                        continue
                    view_index = accepted
                    clouds.append(packed["point_cloud"])
                    view_splits.append(int(source_splits[replay_id]))
                    source_ids.append(int(replay_id))
                    obj_keys.append(source_keys[replay_id])
                    metadata.append(
                        {
                            "source_replay_id": int(replay_id),
                            "source_obj_key": source_keys[replay_id],
                            "split": int(source_splits[replay_id]),
                            "view_index": view_index,
                            "shape_id": row["shape_id"],
                            "link_name": row["link_name"],
                            "repeat": row["repeat"],
                            "base": row["base"],
                            "camera_pose": view["camera_pose"].tolist(),
                            "attempt": view["attempt"],
                            "object_pixels": view["object_pixels"],
                            "target_pixels": view["target_pixels"],
                            "distance": view["distance"],
                            "elevation_deg": view["elevation_deg"],
                            "azimuth_offset_deg": view["azimuth_offset_deg"],
                            "n_points_raw": packed["n_points_raw"],
                            "visible_positive_points": positive_count,
                            "sigma": packed["sigma"],
                        }
                    )
                    if preview_remaining > 0:
                        write_preview(preview_dir, f"r{replay_id:04d}_v{view_index:02d}", view["rgb"])
                        preview_remaining -= 1
                    accepted += 1
                if accepted == 0:
                    raise ValueError("no view passed affordance visibility checks")
            except Exception as exc:
                failures.append({"source_replay_id": replay_id, "obj_key": row.get("obj_key"), "error": f"{type(exc).__name__}: {exc}"})
    finally:
        capturer.close()

    if not clouds:
        raise RuntimeError(f"no accepted views; failures={json.dumps(failures[:10], indent=2)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open_group(str(args.output), mode="w")
    data = out.create_group("data")
    meta = out.create_group("meta")
    # Preserve the original replay/trajectory schema for Stage-2 exactly.
    for name in ("point_cloud", "state", "action"):
        copy_numeric_array(source["data"], data, name)
    for name in ("episode_ends", "episode_replay_ids", "replay_split"):
        copy_numeric_array(source["meta"], meta, name)
    meta.create_dataset(
        "replay_obj_keys",
        data=np.asarray(source["meta"]["replay_obj_keys"][:], dtype=object),
        object_codec=JSON(),
    )

    point_cloud = np.stack(clouds, axis=0).astype(np.float32)
    splits = np.asarray(view_splits, dtype=np.int8)
    data.create_dataset(
        "stage1_aug_point_cloud",
        data=point_cloud,
        chunks=(1, args.num_points, 4),
        compressor=Blosc(cname="zstd", clevel=3),
    )
    meta.create_dataset("stage1_aug_replay_split", data=splits)
    meta.create_dataset("stage1_aug_source_replay_id", data=np.asarray(source_ids, dtype=np.int32))
    meta.create_dataset("stage1_aug_replay_obj_keys", data=np.asarray(obj_keys, dtype=object), object_codec=JSON())
    meta.create_dataset("stage1_aug_camera_pose", data=np.asarray([item["camera_pose"] for item in metadata], dtype=np.float32))
    meta.create_dataset("stage1_aug_view_metadata", data=np.asarray(metadata, dtype=object), object_codec=JSON())
    summary = {
        "schema_version": 1,
        "purpose": "stage2_primary_plus_optional_stage1_multiview_augmentation",
        "source_zarr": str(args.source_zarr),
        "data_root": str(data_root),
        "n_source_replays_requested": len(selected),
        "n_source_replays_accepted": len(set(source_ids)),
        "n_primary_replays": int(source["data"]["point_cloud"].shape[0]),
        "n_stage1_aug_views": len(clouds),
        "n_points": args.num_points,
        "stage1_aug_train_views": int((splits == 0).sum()),
        "stage1_aug_val_views": int((splits == 1).sum()),
        "views_per_replay_requested": args.views_per_replay,
        "camera": {
            "fov_deg": args.fov_deg,
            "elevation_deg": [args.elevation_min_deg, args.elevation_max_deg],
            "target_side_half_angle_deg": args.target_side_half_angle_deg,
            "framing_margin": args.framing_margin,
            "min_target_pixels": args.min_target_pixels,
            "image_border_px": args.image_border_px,
        },
        "failures": failures,
    }
    (args.output / ".zarr_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "output": str(args.output), "preview_dir": str(preview_dir)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export fixed-camera RGBs and flag potentially invalid single-view point clouds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (str(ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from path_config import ARTICU_COLLECTION_ROOT, PROJECT_ROOT
from joint_train.sim.capture_view_pcd import ViewPcdCapturer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(ARTICU_COLLECTION_ROOT))
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "joint_from_data_cam_replays.json"
    )
    parser.add_argument("--zarr", type=Path, default=ROOT / "data" / "joint_from_data_cam.zarr")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "camera_rgb")
    parser.add_argument("--articu-root", type=Path, default=Path(PROJECT_ROOT))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--max-replays", type=int, default=0, help="0 exports every selected replay")
    parser.add_argument("--normal-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-object-pixels", type=int, default=1024)
    parser.add_argument("--min-target-pixels", type=int, default=64)
    parser.add_argument("--border-pixels", type=int, default=2)
    parser.add_argument(
        "--treat-border-as-abnormal",
        action="store_true",
        help="treat object/link border contact as an abnormal condition instead of a framing warning",
    )
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
    choices = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"cannot find data/single under any data-root candidate:\n  {choices}")


def replay_dir(data_root: Path, row: dict) -> Path:
    return data_root / "data" / "single" / str(row["shape_id"]) / str(row["link_name"]) / str(row["repeat"]) / str(row["base"])


def stem(index: int, row: dict) -> str:
    return "{index:04d}_{obj}_{repeat}_{base}".format(
        index=index, obj=row["obj_key"], repeat=row["repeat"], base=row["base"]
    )


def touches_border(mask: np.ndarray, border: int) -> bool:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return False
    height, width = mask.shape
    return bool(
        rows.min() < border
        or rows.max() >= height - border
        or cols.min() < border
        or cols.max() >= width - border
    )


def save_sample(directory: Path, name: str, rgb: np.ndarray, point_cloud: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(directory / f"{name}.png")
    np.save(directory / f"{name}.npy", point_cloud)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"manifest must be a JSON list: {args.manifest}")
    selected = [(index, row) for index, row in enumerate(rows) if args.split == "all" or row.get("split") == args.split]
    if args.max_replays > 0:
        selected = selected[: args.max_replays]
    zarr_root = zarr.open_group(str(args.zarr), mode="r")
    clouds = zarr_root["data"]["point_cloud"]
    if len(rows) != len(clouds):
        raise ValueError(f"manifest replays ({len(rows)}) do not match zarr clouds ({len(clouds)})")

    rng = np.random.default_rng(args.seed)
    normal_count = min(int(args.normal_samples), len(selected))
    normal_indices = set(rng.choice(len(selected), size=normal_count, replace=False).tolist()) if normal_count else set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    abnormal_dir = args.output_dir / "abnormal"
    normal_dir = args.output_dir / "normal_sample"
    checks: list[dict] = []
    capturer = ViewPcdCapturer(articu_root=args.articu_root, render_enabled=True)
    try:
        for selected_index, (replay_index, row) in enumerate(tqdm(selected, desc="export_camera_rgb")):
            state_path = replay_dir(data_root, row) / "initial_state.json"
            result = {**row, "replay_id": replay_index, "initial_state": str(state_path)}
            try:
                init = json.loads(state_path.read_text(encoding="utf-8"))
                view = capturer.capture_view_masks_from_init(init)
                object_pixels = int(view["object_mask"].sum())
                target_pixels = int(view["target_mask"].sum())
                reasons = []
                warnings = []
                if object_pixels < args.min_object_pixels:
                    reasons.append("too_few_object_pixels")
                if target_pixels < args.min_target_pixels:
                    reasons.append("too_few_target_link_pixels")
                if touches_border(view["object_mask"], args.border_pixels):
                    warnings.append("object_touches_image_border")
                if touches_border(view["target_mask"], args.border_pixels):
                    warnings.append("target_link_touches_image_border")
                raw_points = int(view["object_mask"].sum())
                if raw_points < int(clouds.shape[1]):
                    reasons.append("raw_object_points_below_zarr_point_count")
                if args.treat_border_as_abnormal:
                    reasons.extend(warnings)
                result.update(
                    object_pixels=object_pixels,
                    target_pixels=target_pixels,
                    raw_object_points=raw_points,
                    reasons=reasons,
                    warnings=warnings,
                )
                if reasons:
                    save_sample(abnormal_dir, stem(replay_index, row), view["rgb"], np.asarray(clouds[replay_index]))
                elif selected_index in normal_indices:
                    save_sample(normal_dir, stem(replay_index, row), view["rgb"], np.asarray(clouds[replay_index]))
            except Exception as exc:
                result["reasons"] = ["render_or_state_error"]
                result["error"] = f"{type(exc).__name__}: {exc}"
                abnormal_dir.mkdir(parents=True, exist_ok=True)
                np.save(abnormal_dir / f"{stem(replay_index, row)}.npy", np.asarray(clouds[replay_index]))
                print(f"[abnormal] {row.get('obj_key')} {row.get('repeat')}/{row.get('base')}: {exc}", flush=True)
            checks.append(result)
    finally:
        capturer.close()

    abnormal = [row for row in checks if row.get("reasons")]
    (args.output_dir / "quality_report.json").write_text(
        json.dumps({"data_root": str(data_root), "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE checked={len(checks)} abnormal={len(abnormal)} normal_saved={len(list(normal_dir.glob('*.png')))} "
        f"-> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

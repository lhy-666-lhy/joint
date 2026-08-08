"""Capture single-view world-frame point clouds via SAPIEN (articu DemoWorld + Camera).

Depends on articu_sapien at ``ARTICU_ROOT`` (default: /data0/liditao/manipulation/articu_sapien).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ARTICU_ROOT = Path("/data0/liditao/manipulation/articu_sapien")
DEFAULT_PARTNET_ROOT = Path("/data0/dataset/partnet-mobility/where2act_original_sapien_dataset")
INSPIRE_PARTNET_PREFIXES = (
    "/inspire/hdd/global_user/liditao-253108110078/datasets/partnet-mobility/where2act_original_sapien_dataset",
    "/inspire/hdd/global_user/liditao-253108110078/datasets/partnet-mobility/where2act_original_sapien_dataset/",
)

_MIN_POINTS = 256


class InsufficientPointCloudError(ValueError):
    def __init__(self, required: int, available: int) -> None:
        self.required = int(required)
        self.available = int(available)
        super().__init__(
            f"point cloud requires at least {self.required} finite points, got {self.available}"
        )


def deterministic_resample(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if count <= 0 or points.shape[0] < count or not np.isfinite(points).all():
        raise InsufficientPointCloudError(count, points.shape[0])
    indices = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
    return points[indices].astype(np.float32, copy=True)


def capture_current_world_point_cloud(
    world,
    *,
    camera=None,
    num_points: int = 1024,
    image_size: int = 448,
    fov: float = 35.0,
):
    """Capture a causal world-frame object cloud from the current SAPIEN world."""
    from sapien_utils.camera import Camera

    if not bool(getattr(world, "render_enabled", False)):
        raise RuntimeError("live point cloud capture requires the SAPIEN renderer")
    if not hasattr(world, "object_position_offset"):
        world.object_position_offset = 0.0
    if camera is None:
        camera = Camera(
            world, near=0.05, far=100.0, image_size=int(image_size), dist=2.0,
            phi=float(np.pi / 5.0), theta=float(np.pi), fov=float(fov),
            fixed_position=False, random_position=False,
        )
    if hasattr(world.scene, "update_render"):
        world.scene.update_render()
    camera.get_observation(vis_rgbd=False, vis_pcd=False)
    raw = np.asarray(camera.get_world_pointcloud(object_only=True), dtype=np.float32)
    return deterministic_resample(raw, int(num_points)), camera, int(raw.shape[0])


def capture_current_world_point_cloud_with_target_mask(
    world,
    target_part: str,
    *,
    camera=None,
    num_points: int = 1024,
    image_size: int = 448,
    fov: float = 35.0,
):
    """Capture aligned whole-object world XYZ and target-link membership."""
    cloud, camera, raw_count = capture_current_world_point_cloud(
        world,
        camera=camera,
        num_points=num_points,
        image_size=image_size,
        fov=fov,
    )
    del cloud
    target_link = next(
        (link for link in world.object.get_links() if link.get_name() == str(target_part)),
        None,
    )
    if target_link is None:
        raise ValueError(f"target part not found in current scene: {target_part}")
    position = np.asarray(camera.last_position, dtype=np.float32)
    valid_object = (position[..., 3] < 1.0) & np.asarray(camera.get_object_mask(), dtype=bool)
    points_opengl = position[..., :3][valid_object]
    model_matrix = camera.camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    segmentation = np.asarray(camera.camera.get_picture("Segmentation"))
    target_id = int(target_link.entity.get_per_scene_id())
    target_pixels = segmentation[..., 1] == target_id
    raw_target_mask = np.asarray(target_pixels[valid_object], dtype=bool)
    if points_world.shape[0] != raw_count or raw_target_mask.shape != (raw_count,):
        raise RuntimeError("object point and segmentation buffers are not aligned")
    indices = np.linspace(0, raw_count - 1, int(num_points), dtype=np.int64)
    return (
        np.asarray(points_world[indices], dtype=np.float32),
        raw_target_mask[indices].copy(),
        camera,
        int(raw_count),
        int(raw_target_mask.sum()),
    )


def capture_current_target_point_cloud(
    world,
    target_part: str,
    *,
    camera=None,
    num_points: int = 1024,
):
    """Capture the explicitly selected target link from the current camera buffers."""
    points_world, camera = capture_current_target_points(
        world, target_part, camera=camera, num_points=num_points
    )
    return (
        deterministic_resample(points_world, int(num_points)),
        camera,
        int(points_world.shape[0]),
    )


def capture_current_target_points(
    world,
    target_part: str,
    *,
    camera=None,
    num_points: int = 1024,
):
    """Capture all visible target-link points in the world frame."""
    cloud, camera, _ = capture_current_world_point_cloud(
        world, camera=camera, num_points=num_points
    )
    del cloud
    target_link = next(
        (link for link in world.object.get_links() if link.get_name() == str(target_part)),
        None,
    )
    if target_link is None:
        raise ValueError(f"target part not found in current scene: {target_part}")
    position = np.asarray(camera.last_position, dtype=np.float32)
    valid = position[..., 3] < 1.0
    target_id = int(target_link.entity.get_per_scene_id())
    segmentation = np.asarray(camera.camera.get_picture("Segmentation"))
    target_mask = valid & (segmentation[..., 1] == target_id)
    points_opengl = position[..., :3][target_mask]
    model_matrix = camera.camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    return points_world.astype(np.float32, copy=False), camera


def _aabb_corners(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (aabb_min[0], aabb_max[0])
            for y in (aabb_min[1], aabb_max[1])
            for z in (aabb_min[2], aabb_max[2])
        ],
        dtype=np.float64,
    )


def _link_aabb(link) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = [link, *list(link.entity.get_components())]
    for candidate in candidates:
        getter = getattr(candidate, "get_global_aabb_fast", None)
        if getter is None:
            continue
        try:
            aabb = np.asarray(getter(), dtype=np.float64)
            if aabb.shape == (2, 3) and np.all(np.isfinite(aabb)):
                return aabb[0], aabb[1]
        except Exception:
            continue
    return None


def _look_at_matrix(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target - position, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    left = np.cross(up_hint, forward)
    left /= max(float(np.linalg.norm(left)), 1e-8)
    up = np.cross(forward, left)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((forward, left, up))
    matrix[:3, 3] = position
    return matrix


def _mask_is_framed(mask: np.ndarray, border: int) -> bool:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return False
    height, width = mask.shape
    return bool(
        rows.min() >= border
        and rows.max() < height - border
        and cols.min() >= border
        and cols.max() < width - border
    )


def ensure_articu_on_path(articu_root: Path | str | None = None) -> Path:
    root = Path(articu_root or os.environ.get("ARTICU_ROOT", DEFAULT_ARTICU_ROOT))
    if not root.is_dir():
        raise FileNotFoundError(f"articu_sapien not found: {root}")
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    os.environ.setdefault("ARTICU_ROOT", s)
    os.environ.setdefault("PROJECT_DIR", s)
    return root


def resolve_urdf(
    urdf_path: str | Path,
    *,
    partnet_root: Path | str = DEFAULT_PARTNET_ROOT,
) -> Path:
    """Map inspire/remote URDF paths onto local PartNet dataset."""
    raw = str(urdf_path)
    p = Path(raw)
    if p.is_file():
        return p.resolve()

    partnet_root = Path(partnet_root)
    for prefix in INSPIRE_PARTNET_PREFIXES:
        if raw.startswith(prefix):
            rel = raw[len(prefix) :].lstrip("/")
            cand = partnet_root / rel
            if cand.is_file():
                return cand.resolve()

    # fallback: .../where2act_original_sapien_dataset/<shape>/mobility_vhacd.urdf
    parts = Path(raw).parts
    if "where2act_original_sapien_dataset" in parts:
        idx = parts.index("where2act_original_sapien_dataset")
        rel = Path(*parts[idx + 1 :])
        cand = partnet_root / rel
        if cand.is_file():
            return cand.resolve()

    # last resort: shape_id folder name
    for i, part in enumerate(parts):
        if part.isdigit() and i + 1 < len(parts) and parts[i + 1].endswith(".urdf"):
            cand = partnet_root / part / parts[i + 1]
            if cand.is_file():
                return cand.resolve()

    raise FileNotFoundError(f"cannot resolve URDF locally: {raw}")


def base_pose_from_init(init: dict) -> Any:
    from force_admittance_collect.data_types import BasePose

    raw = init.get("base_pose") or init.get("planned_base_pose")
    vals = [float(x) for x in raw]
    # with frame_transform: stored as [x, y, yaw, z]
    if init.get("frame_transform") is not None and len(vals) >= 4:
        return BasePose(vals[0], vals[1], vals[2], vals[3])
    if len(vals) >= 4:
        return BasePose(vals[0], vals[1], vals[3], vals[2])
    return BasePose(vals[0], vals[1], vals[2], 0.0)


def load_initial_state(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ViewPcdCapturer:
    """Reuse DemoWorld across replays that share the same URDF+size."""

    def __init__(
        self,
        *,
        articu_root: Path | str | None = None,
        partnet_root: Path | str = DEFAULT_PARTNET_ROOT,
        settle_steps: int = 20,
        image_size: int = 448,
        fov: float = 35.0,
        min_points: int = _MIN_POINTS,
        object_only: bool = True,
        render_enabled: bool = True,
    ):
        self.articu_root = ensure_articu_on_path(articu_root)
        self.partnet_root = Path(partnet_root)
        self.settle_steps = int(settle_steps)
        self.image_size = int(image_size)
        self.fov = float(fov)
        self.min_points = int(min_points)
        self.object_only = bool(object_only)
        self.render_enabled = bool(render_enabled)

        self._world = None
        self._world_key: tuple[str, float] | None = None
        self._camera = None

    def close(self) -> None:
        self._camera = None
        self._world = None
        self._world_key = None

    def _get_world(self, urdf: Path, size: float):
        from force_admittance_collect.world import DemoWorld

        key = (str(urdf), float(size))
        if self._world is not None and self._world_key == key:
            return self._world
        # drop previous
        self._camera = None
        self._world = DemoWorld(str(urdf), size=float(size), render_enabled=self.render_enabled)
        # Camera API expects this attribute (legacy ContactEnv)
        if not hasattr(self._world, "object_position_offset"):
            self._world.object_position_offset = 0.0
        self._world_key = key
        return self._world

    def _ensure_camera(self, world) -> Any:
        from sapien_utils.camera import Camera

        if self._camera is not None and getattr(self._camera, "env", None) is world:
            return self._camera
        # Closer than Camera's default dist=4 so the object fills the frame.
        self._camera = Camera(
            world,
            near=0.05,
            far=100.0,
            image_size=self.image_size,
            dist=2.0,
            phi=float(np.pi / 5.0),
            theta=float(np.pi),
            fov=self.fov,
            fixed_position=False,
            random_position=False,
        )
        return self._camera

    def apply_initial_state(self, world, init: dict) -> None:
        from sapien_utils.env import set_articulation_joint_state

        link_name = str(init["link_name"])
        base_pose = base_pose_from_init(init)
        obj_qpos = np.asarray(init["initial_object_qpos"], dtype=np.float64)
        robot_q = init.get("robot_default_full_qpos")
        if robot_q is None:
            robot_q = world.default_full_qpos
        else:
            robot_q = np.asarray(robot_q, dtype=np.float64)

        world.set_object_origin()
        world.set_base_pose(base_pose)
        set_articulation_joint_state(
            world.object,
            init.get("state", "target_almost_closed"),
            target_link_name=link_name,
            zero_qvel=True,
        )
        world.object.set_qpos(obj_qpos)
        world.object.set_qvel(np.zeros_like(np.asarray(world.object.get_qvel(), dtype=np.float64)))
        world.set_robot_qpos(robot_q)
        for _ in range(self.settle_steps):
            world.scene.step()
        if hasattr(world.scene, "update_render"):
            world.scene.update_render()

    def capture_from_init(
        self,
        init: dict,
        *,
        cache_path: Path | None = None,
    ) -> np.ndarray:
        """Return world-frame XYZ (N,3) float32 for one initial_state."""
        if cache_path is not None and Path(cache_path).is_file():
            pts = np.load(cache_path)
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] >= self.min_points:
                return pts

        urdf = resolve_urdf(init["object_urdf"], partnet_root=self.partnet_root)
        size = float(init["size"])
        world = self._get_world(urdf, size)
        self.apply_initial_state(world, init)
        cam = self._ensure_camera(world)
        # Critical for newer SAPIEN: refresh render buffers before take_picture
        if hasattr(world.scene, "update_render"):
            world.scene.update_render()
        cam.get_observation(vis_rgbd=False, vis_pcd=False)
        pts = np.asarray(
            cam.get_world_pointcloud(object_only=self.object_only), dtype=np.float32
        ).reshape(-1, 3)
        if pts.shape[0] < self.min_points:
            pts_full = np.asarray(
                cam.get_world_pointcloud(object_only=False), dtype=np.float32
            ).reshape(-1, 3)
            if pts_full.shape[0] >= self.min_points:
                pts = pts_full
            else:
                raise RuntimeError(
                    f"too few camera points: object_only={pts.shape[0]} full={pts_full.shape[0]} "
                    f"(min={self.min_points})"
                )

        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, pts)
        return pts

    def capture_rgb_from_init(self, init: dict) -> np.ndarray:
        """Render the same fixed camera view used for the point cloud."""
        urdf = resolve_urdf(init["object_urdf"], partnet_root=self.partnet_root)
        world = self._get_world(urdf, float(init["size"]))
        self.apply_initial_state(world, init)
        cam = self._ensure_camera(world)
        if hasattr(world.scene, "update_render"):
            world.scene.update_render()
        rgb, _ = cam.get_observation(vis_rgbd=False, vis_pcd=False)
        return np.asarray(rgb, dtype=np.float32)

    def capture_view_masks_from_init(self, init: dict) -> dict[str, np.ndarray]:
        """Render one view and return RGB plus visibility masks for quality checks."""
        urdf = resolve_urdf(init["object_urdf"], partnet_root=self.partnet_root)
        world = self._get_world(urdf, float(init["size"]))
        self.apply_initial_state(world, init)
        cam = self._ensure_camera(world)
        if hasattr(world.scene, "update_render"):
            world.scene.update_render()
        rgb, _ = cam.get_observation(vis_rgbd=False, vis_pcd=False)
        position = np.asarray(cam.last_position, dtype=np.float32)
        valid_mask = position[..., 3] < 1.0
        object_mask = valid_mask & np.asarray(cam.get_object_mask(), dtype=bool)
        target_link = next(
            (link for link in world.object.get_links() if link.get_name() == str(init["link_name"])),
            None,
        )
        if target_link is None:
            raise RuntimeError(f"target link not found: {init['link_name']}")
        entity = target_link.entity
        target_id = int(entity.get_per_scene_id())
        segmentation = np.asarray(cam.camera.get_picture("Segmentation"))
        target_mask = np.any(segmentation == target_id, axis=-1)
        return {
            "rgb": np.asarray(rgb, dtype=np.float32),
            "valid_mask": valid_mask,
            "object_mask": object_mask,
            "target_mask": target_mask,
        }

    def capture_target_aware_views_from_init(
        self,
        init: dict,
        *,
        rng: np.random.Generator,
        views_per_replay: int,
        max_attempts: int,
        fov_deg: float = 55.0,
        elevation_min_deg: float = 12.0,
        elevation_max_deg: float = 55.0,
        target_side_half_angle_deg: float = 60.0,
        framing_margin: float = 1.35,
        image_border_px: int = 3,
        min_target_pixels: int = 400,
        min_object_points: int = 4096,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Capture accepted target-aware views as world-frame object point clouds.

        The target and full object AABBs determine a safe camera distance. Candidate
        directions are sampled around the target-facing side and accepted only when
        both object and target masks are in frame.
        """
        from sapien_utils.sapien_compat import Pose

        urdf = resolve_urdf(init["object_urdf"], partnet_root=self.partnet_root)
        world = self._get_world(urdf, float(init["size"]))
        if not bool(getattr(world, "render_enabled", False)):
            raise RuntimeError("SAPIEN offscreen renderer is unavailable; cannot capture target-aware views")
        self.apply_initial_state(world, init)
        world.refresh_object_aabb()
        target_link = next(
            (link for link in world.object.get_links() if link.get_name() == str(init["link_name"])),
            None,
        )
        if target_link is None:
            raise RuntimeError(f"target link not found: {init['link_name']}")
        target_aabb = _link_aabb(target_link)
        if target_aabb is None:
            raise RuntimeError(f"target link has no usable AABB: {init['link_name']}")

        object_min = np.asarray(world.object_aabb_min, dtype=np.float64)
        object_max = np.asarray(world.object_aabb_max, dtype=np.float64)
        target_min, target_max = target_aabb
        focus = 0.25 * (object_min + object_max + target_min + target_max)
        framing = np.concatenate((_aabb_corners(object_min, object_max), _aabb_corners(target_min, target_max)))
        radius = float(np.max(np.linalg.norm(framing - focus[None], axis=1)))
        distance = max(0.25, radius / math.tan(math.radians(float(fov_deg)) * 0.5) * float(framing_margin))

        object_center = 0.5 * (object_min + object_max)
        target_center = 0.5 * (target_min + target_max)
        side_xy = target_center[:2] - object_center[:2]
        side_norm = float(np.linalg.norm(side_xy))
        if side_norm < 1e-8:
            side_xy = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            side_xy /= side_norm

        object_ids = {
            int(link.entity.get_per_scene_id())
            for link in world.object.get_links()
            if hasattr(link.entity, "get_per_scene_id")
        }
        target_id = int(target_link.entity.get_per_scene_id())
        cam = self._ensure_camera(world)
        accepted: list[dict[str, Any]] = []
        attempts = 0
        while len(accepted) < int(views_per_replay) and attempts < int(max_attempts):
            attempts += 1
            # Prefer target-facing views, then broaden the search if occlusion prevents acceptance.
            half_angle = float(target_side_half_angle_deg)
            if attempts > max(1, int(max_attempts) // 2):
                half_angle = 180.0
            azimuth_offset = math.radians(rng.uniform(-half_angle, half_angle))
            elevation = math.radians(rng.uniform(float(elevation_min_deg), float(elevation_max_deg)))
            direction_xy = np.asarray(
                [
                    math.cos(azimuth_offset) * side_xy[0] - math.sin(azimuth_offset) * side_xy[1],
                    math.sin(azimuth_offset) * side_xy[0] + math.cos(azimuth_offset) * side_xy[1],
                ],
                dtype=np.float64,
            )
            direction = np.asarray(
                [math.cos(elevation) * direction_xy[0], math.cos(elevation) * direction_xy[1], math.sin(elevation)],
                dtype=np.float64,
            )
            matrix = _look_at_matrix(focus + distance * direction, focus)
            if hasattr(cam.camera, "entity"):
                cam.camera.entity.set_pose(Pose(matrix))
            else:
                cam.camera.set_pose(Pose(matrix))
            world.scene.update_render()
            rgb, _ = cam.get_observation(vis_rgbd=False, vis_pcd=False)
            position = np.asarray(cam.last_position, dtype=np.float32)
            segmentation = np.asarray(cam.camera.get_picture("Segmentation"))
            valid_mask = position[..., 3] < 1.0
            object_mask = valid_mask & np.any(np.isin(segmentation, list(object_ids)), axis=-1)
            target_mask = np.any(segmentation == target_id, axis=-1)
            local = position[..., :3][object_mask]
            reasons = []
            if not _mask_is_framed(object_mask, int(image_border_px)):
                reasons.append("object_not_framed")
            if int(target_mask.sum()) < int(min_target_pixels):
                reasons.append("target_too_small")
            elif not _mask_is_framed(target_mask, int(image_border_px)):
                reasons.append("target_not_framed")
            if len(local) < int(min_object_points):
                reasons.append("too_few_object_points")
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "rgb": np.rint(np.clip(rgb[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8),
                        "object_mask": object_mask,
                        "target_mask": target_mask,
                        "camera_pose": matrix.astype(np.float32),
                        "attempt": int(attempts),
                        "object_pixels": int(object_mask.sum()),
                        "target_pixels": int(target_mask.sum()),
                        "distance": float(distance),
                        "elevation_deg": float(math.degrees(elevation)),
                        "azimuth_offset_deg": float(math.degrees(azimuth_offset)),
                        "reasons": reasons,
                    }
                )
            if reasons:
                continue
            model = np.asarray(cam.camera.get_model_matrix(), dtype=np.float32)
            points = local @ model[:3, :3].T + model[:3, 3]
            accepted.append(
                {
                    "points": points.astype(np.float32),
                    "rgb": np.asarray(rgb, dtype=np.float32),
                    "object_mask": object_mask,
                    "camera_pose": matrix.astype(np.float32),
                    "attempt": int(attempts),
                    "object_pixels": int(object_mask.sum()),
                    "target_pixels": int(target_mask.sum()),
                    "distance": float(distance),
                    "elevation_deg": float(math.degrees(elevation)),
                    "azimuth_offset_deg": float(math.degrees(azimuth_offset)),
                }
            )
        return accepted


def capture_single_view_xyz_from_initial_state_json(
    initial_state_json: Path | str,
    *,
    articu_root: Path | str | None = None,
    settle_steps: int = 20,
    cache_path: Path | None = None,
) -> np.ndarray:
    capturer = ViewPcdCapturer(articu_root=articu_root, settle_steps=settle_steps)
    try:
        init = load_initial_state(initial_state_json)
        return capturer.capture_from_init(init, cache_path=cache_path)
    finally:
        capturer.close()

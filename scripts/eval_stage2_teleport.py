#!/usr/bin/env python3
"""Stage-2 sim eval: load ckpt, teleport absolute qpos, physics on, save GIF.

Protocol:
  - Pick N random val objects
  - Static point cloud obs (one cloud per object/replay)
  - Each replan: predict 8 actions, execute the first 4 (no temporal ensemble)
  - Physics + light interpolation between teleports for smoother GIF
  - When gripper finger contacts target link: latch grasp (lock fingers closed)
  - Save GIF per object for qualitative check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for p in (str(ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from joint_train.models.joint_policy import JointDiffusionPolicy  # noqa: E402
from joint_train.utils.affordance_interventions import intervene_affordance  # noqa: E402

from force_admittance_collect.data_types import BasePose  # noqa: E402
from force_admittance_collect.controller import find_target_joint, target_joint_index  # noqa: E402
from force_admittance_collect.feedback import read_target_contact_feedback  # noqa: E402
from force_admittance_collect.gif import SapienGifRecorder  # noqa: E402
from force_admittance_collect.world import DemoWorld  # noqa: E402
from path_config import ARTICU_COLLECTION_ROOT, PARTNET_DATASET_ROOT  # noqa: E402
from sapien_utils.env import set_articulation_joint_state  # noqa: E402


COLLECTED_ROOT = Path(ARTICU_COLLECTION_ROOT)
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0


def parse_obj_key(obj_key: str) -> tuple[str, str]:
    if "_link_" not in obj_key:
        raise ValueError(f"bad obj_key={obj_key}")
    shape_id, link_suffix = obj_key.rsplit("_link_", 1)
    return shape_id, f"link_{link_suffix}"


def resolve_path(path: str | Path) -> Path:
    item = Path(path)
    return item if item.is_absolute() else Path(PARTNET_DATASET_ROOT) / item


def find_demo_dir(obj_key: str, rng: np.random.Generator) -> Path:
    shape_id, link_name = parse_obj_key(obj_key)
    cands = sorted(
        COLLECTED_ROOT.glob(f"**/single/{shape_id}/{link_name}/repeat_*/base_0000/initial_state.json")
    )
    if not cands:
        raise FileNotFoundError(f"no collected demo for {obj_key}")
    pick = Path(rng.choice(cands))
    return pick.parent


def demo_dir_from_replay(row: dict) -> Path:
    return (
        COLLECTED_ROOT
        / "data"
        / "single"
        / str(row["shape_id"])
        / str(row["link_name"])
        / str(row["repeat"])
        / str(row["base"])
    )


def load_init(demo_dir: Path, trajectory_name: str | None = None) -> dict:
    init = json.loads((demo_dir / "initial_state.json").read_text())
    traj_files = (
        [demo_dir / "trajectory" / trajectory_name]
        if trajectory_name is not None
        else sorted((demo_dir / "trajectory").glob("*.npz"))
    )
    if not traj_files:
        raise FileNotFoundError(f"no trajectory under {demo_dir}")
    traj = np.load(traj_files[0], allow_pickle=True)
    result = {}
    if "result_json" in traj.files:
        try:
            result = json.loads(str(np.asarray(traj["result_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError):
            result = {}
    return {"init": init, "traj": traj, "traj_path": traj_files[0], "result": result}


def stable_target_seed(seed: int, obj_key: str) -> int:
    digest = hashlib.sha256(str(obj_key).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return int((int(seed) + offset) % (2**31 - 1))


def base_pose_from_init(init: dict) -> BasePose:
    raw = init.get("base_pose") or init.get("planned_base_pose")
    vals = [float(x) for x in raw]
    # with frame_transform: stored as [x, y, yaw, z]
    if init.get("frame_transform") is not None and len(vals) >= 4:
        return BasePose(vals[0], vals[1], vals[2], vals[3])
    if len(vals) >= 4:
        return BasePose(vals[0], vals[1], vals[3], vals[2])
    return BasePose(vals[0], vals[1], vals[2], 0.0)


def list_split_objects(zarr_path: Path, split: str = "val") -> list[tuple[str, int]]:
    """Unique obj_key with one representative replay_id for train(0)/val(1)."""
    root = zarr.open(str(zarr_path), mode="r")
    splits = np.asarray(root["meta"]["replay_split"][:])
    keys = [str(k) for k in root["meta"]["replay_obj_keys"][:]]
    want = 0 if split == "train" else 1
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for rid in np.nonzero(splits == want)[0].tolist():
        k = keys[rid]
        if k in seen:
            continue
        seen.add(k)
        out.append((k, int(rid)))
    return out


def qpos9_from_robot(world: DemoWorld) -> np.ndarray:
    q = np.asarray(world.robot.get_qpos(), dtype=np.float64).reshape(-1)
    return q[:9].astype(np.float32)


def set_qpos9_teleport(world: DemoWorld, q9: np.ndarray) -> None:
    q9 = np.asarray(q9, dtype=np.float64).reshape(-1)
    full = np.asarray(world.robot.get_qpos(), dtype=np.float64).copy()
    full[:7] = q9[:7]
    if full.shape[0] >= 9:
        full[7] = float(np.clip(q9[7], 0.0, FINGER_OPEN))
        full[8] = float(np.clip(q9[8] if q9.shape[0] > 8 else q9[7], 0.0, FINGER_OPEN))
    world.set_robot_qpos(full)
    world.robot.set_qvel(np.zeros_like(world.robot.get_qvel()))


def interpolate_execute(
    world: DemoWorld,
    q_tgt: np.ndarray,
    recorder: SapienGifRecorder,
    *,
    interp_steps: int,
    physics_per_substep: int,
    capture_every: int = 1,
) -> None:
    """Smoothly move from current qpos to target; capture sparsely for speed."""
    q0 = qpos9_from_robot(world).astype(np.float64)
    q1 = np.asarray(q_tgt, dtype=np.float64).reshape(-1).copy()
    q1[7:9] = np.clip(q1[7:9], 0.0, FINGER_OPEN)
    n = max(1, int(interp_steps))
    cap_every = max(1, int(capture_every))
    for i in range(1, n + 1):
        a = float(i) / float(n)
        q = (1.0 - a) * q0 + a * q1
        set_qpos9_teleport(world, q.astype(np.float32))
        for _ in range(max(1, int(physics_per_substep))):
            world.step(render=False)
        # capture at most once per interp substep (and only every cap_every)
        if i % cap_every == 0 or i == n:
            recorder.capture(force=True)


def build_state(q9: np.ndarray, grasped: bool) -> np.ndarray:
    onehot = np.asarray([0.0, 1.0] if grasped else [1.0, 0.0], dtype=np.float32)
    return np.concatenate([np.asarray(q9, dtype=np.float32).reshape(9), onehot], axis=0)


def load_policy(ckpt_path: Path, device: torch.device) -> JointDiffusionPolicy:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    dual = bool(ckpt.get("dual_head", False)) if isinstance(ckpt, dict) else False
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not dual and isinstance(state, dict):
        dual = any(k.startswith("affordance_net.convs4_cls") or k.startswith("convs4_cls") for k in state)

    # affordance keys may be nested under affordance_net.
    if not dual and isinstance(state, dict):
        dual = any("convs4_cls" in k for k in state)

    policy = JointDiffusionPolicy(
        action_dim=9,
        state_dim=11,
        horizon=int(args.get("horizon", 16)),
        n_obs_steps=int(args.get("n_obs_steps", 2)),
        n_action_steps=int(args.get("n_action_steps", 8)),
        down_dims=tuple(args.get("down_dims", (512, 1024, 2048))),
        reuse_static_point_feature=bool(args.get("reuse_static_point_feature", False)),
        condition_mode=("no_map" if args.get("condition_variant") == "no_map" else "affordance"),
        obs_encoder_variant=str(args.get("obs_encoder_variant", "pointnet")),
        affordance_adapter=str(args.get("affordance_adapter", "none")),
        affordance_aux_weight=float(args.get("affordance_aux_weight", 0.0)),
        contact_condition=str(args.get("contact_condition", "none")),
        dual_head=dual,
    )
    policy.condition_variant = str(args.get("condition_variant", "updated"))
    policy.contact_sidecar_path = args.get("contact_sidecar")
    # Prefer EMA weights for eval if present
    if isinstance(ckpt, dict) and ckpt.get("ema") is not None:
        missing, unexpected = policy.load_state_dict(ckpt["ema"], strict=False)
        print(f"loaded EMA weights missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    else:
        missing, unexpected = policy.load_state_dict(state, strict=False)
        print(f"loaded model weights missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if isinstance(ckpt, dict) and ckpt.get("normalizer") is not None:
        policy.normalizer.load_state_dict(ckpt["normalizer"])
    policy.to(device)
    policy.eval()
    return policy


@torch.no_grad()
def policy_act(
    policy: JointDiffusionPolicy,
    xyz: np.ndarray,
    state_hist: deque,
    device: torch.device,
    use_gt_affordance: bool,
    aff_gt: np.ndarray | None,
    contact_info: dict | None = None,
) -> np.ndarray:
    """Return (n_action_steps, 9) absolute qpos actions."""
    To = policy.n_obs_steps
    states = list(state_hist)[-To:]
    while len(states) < To:
        states = [states[0]] + states
    state = np.stack(states, axis=0)  # To,11
    # pad to horizon for normalizer API used in training (full horizon state)
    H = policy.horizon
    state_full = np.zeros((H, 11), dtype=np.float32)
    state_full[:To] = state
    state_full[To:] = state[-1]

    batch = {
        "point_cloud_xyz": torch.from_numpy(xyz.astype(np.float32)).unsqueeze(0).to(device),
        "state": torch.from_numpy(state_full).unsqueeze(0).to(device),
    }
    if contact_info is not None:
        batch["contact_xyz_world"] = torch.from_numpy(
            np.asarray(contact_info["contact_xyz_world"], dtype=np.float32)
        ).unsqueeze(0).to(device)
        batch["contact_visible_5cm"] = torch.tensor(
            [float(contact_info["contact_visible_5cm"])], device=device
        )
        batch["contact_valid"] = torch.tensor(
            [float(contact_info["contact_valid"])], device=device
        )
    if use_gt_affordance and aff_gt is not None:
        batch["affordance_gt"] = torch.from_numpy(aff_gt.astype(np.float32)).unsqueeze(0).to(device)
        out = policy.predict_action(batch, use_gt_affordance=True)
    else:
        out = policy.predict_action(batch, use_gt_affordance=False)
    act = out["action"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return act


def select_contact_info(
    sidecar: dict[str, np.ndarray],
    *,
    replay_id: int,
    trajectory_name: str,
    intervention: str,
) -> dict:
    replay_ids = sidecar["replay_id"]
    names = sidecar["trajectory_name"]
    valid_rows = np.flatnonzero(
        (replay_ids == int(replay_id))
        & np.isfinite(sidecar["contact_xyz_world"]).all(axis=1)
    )
    exact = [int(index) for index in valid_rows if str(names[index]) == str(trajectory_name)]
    if len(exact) != 1:
        raise RuntimeError(
            f"expected one contact for replay={replay_id} trajectory={trajectory_name}, got {exact}"
        )
    correct_id = exact[0]
    kind = str(intervention).lower()
    if kind == "correct":
        selected_id = correct_id
        valid = True
    elif kind == "wrong":
        wrong = [int(index) for index in valid_rows if int(index) != correct_id]
        if not wrong:
            raise RuntimeError(f"no wrong contact candidate for replay={replay_id}")
        correct_xyz = np.asarray(sidecar["contact_xyz_world"][correct_id], dtype=np.float32)
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(sidecar["contact_xyz_world"][index], dtype=np.float32)
                    - correct_xyz
                )
            )
            for index in wrong
        ]
        selected_id = wrong[int(np.argmax(distances))]
        valid = True
    elif kind == "zero":
        selected_id = correct_id
        valid = False
    else:
        raise ValueError(f"unknown contact_intervention={intervention}")
    return {
        "contact_xyz_world": (
            np.asarray(sidecar["contact_xyz_world"][selected_id], dtype=np.float32)
            if valid
            else np.zeros(3, dtype=np.float32)
        ).tolist(),
        "contact_visible_5cm": bool(sidecar["visible_5cm"][selected_id]) if valid else False,
        "contact_valid": bool(valid),
        "contact_intervention": kind,
        "contact_episode_id": int(selected_id),
        "correct_contact_episode_id": int(correct_id),
        "distance_from_correct": float(
            np.linalg.norm(
                np.asarray(sidecar["contact_xyz_world"][selected_id], dtype=np.float32)
                - np.asarray(sidecar["contact_xyz_world"][correct_id], dtype=np.float32)
            )
        ) if valid else None,
    }


def run_one_object(
    *,
    policy: JointDiffusionPolicy,
    zarr_root,
    obj_key: str,
    replay_id: int,
    out_gif: Path,
    device: torch.device,
    args,
    rng: np.random.Generator,
    replay_row: dict | None,
    contact_sidecar: dict[str, np.ndarray] | None,
) -> dict:
    if replay_row is None:
        demo_dir = find_demo_dir(obj_key, rng)
        trajectory_name = None
    else:
        demo_dir = demo_dir_from_replay(replay_row)
        names = [str(item) for item in replay_row.get("traj_names", [])]
        if not names:
            raise RuntimeError(f"source replay for {obj_key} has no trajectories")
        trajectory_name = names[0]
    packed = load_init(demo_dir, trajectory_name=trajectory_name)
    init = packed["init"]
    traj = packed["traj"]
    traj_result = packed["result"]
    link_name = str(init["link_name"])
    urdf = str(resolve_path(init["object_urdf"]))
    size = float(init["size"])
    base_pose = base_pose_from_init(init)
    obj_qpos = np.asarray(init["initial_object_qpos"], dtype=np.float64)
    q0 = np.asarray(traj["joint_qpos"][0], dtype=np.float32)

    pc = np.asarray(zarr_root["data"]["point_cloud"][replay_id], dtype=np.float32)
    xyz = pc[:, :3]
    condition_variant = str(getattr(policy, "condition_variant", "updated"))
    if condition_variant == "initial":
        aff_gt = np.clip(
            np.asarray(zarr_root["data"]["affordance_initial"][replay_id], dtype=np.float32),
            0.0,
            1.0,
        )
    else:
        aff_gt = np.clip(pc[:, 3], 0.0, 1.0)
    aff_gt, intervention_metadata = intervene_affordance(
        zarr_root,
        xyz=xyz,
        correct=aff_gt,
        obj_key=obj_key,
        replay_id=replay_id,
        label_source=("initial" if condition_variant == "initial" else "updated"),
        intervention=args.affordance_intervention,
    )
    contact_info = None
    if policy.contact_condition == "coordinate":
        if contact_sidecar is None:
            raise RuntimeError("coordinate contact policy requires contact sidecar")
        contact_info = select_contact_info(
            contact_sidecar,
            replay_id=replay_id,
            trajectory_name=packed["traj_path"].name,
            intervention=args.contact_intervention,
        )

    print(f"\n=== {obj_key} replay={replay_id} demo={demo_dir} urdf={urdf}", flush=True)

    world = DemoWorld(urdf, size=size, render_enabled=True)
    world.set_object_origin()
    world.set_base_pose(base_pose)
    set_articulation_joint_state(
        world.object, init["state"], target_link_name=link_name, zero_qvel=True
    )
    world.object.set_qpos(obj_qpos)
    world.object.set_qvel(np.zeros_like(world.object.get_qvel()))
    set_qpos9_teleport(world, q0)
    for _ in range(int(args.settle_steps)):
        world.step(render=False)

    recorder = SapienGifRecorder(
        world,
        out_gif,
        width=args.gif_width,
        height=args.gif_height,
        fps=args.gif_fps,
        max_frames=args.gif_max_frames,
        render_every=1,  # we control sparsity via interpolate capture_every
        name=f"eval_{obj_key}",
    )
    # camera toward object origin
    recorder.set_view(np.zeros(3, dtype=np.float64), radius=1.5)

    grasped = False
    grasp_step = None
    state_hist: deque = deque(maxlen=max(8, policy.n_obs_steps + 2))
    state_hist.append(build_state(qpos9_from_robot(world), grasped))

    target_joint = find_target_joint(world.object, link_name)
    if target_joint is None:
        raise RuntimeError(f"target joint not found for {link_name}")
    target_joint_idx = int(target_joint_index(world.object, target_joint))
    qlimits = np.asarray(world.object.get_qlimits(), dtype=np.float64)
    if not 0 <= target_joint_idx < qlimits.shape[0]:
        raise RuntimeError(f"target joint index {target_joint_idx} outside qlimits {qlimits.shape}")
    lower, upper = (float(value) for value in qlimits[target_joint_idx])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise RuntimeError(f"invalid target joint limits: [{lower}, {upper}]")
    door0 = float(np.asarray(world.object.get_qpos()).reshape(-1)[target_joint_idx])
    success_open_ratio = float(
        traj_result.get("success_open_ratio")
        if traj_result.get("success_open_ratio") is not None
        else args.success_open_ratio
    )
    success_open_ratio = float(np.clip(success_open_ratio, 0.0, 1.0))
    success_qpos = lower + (upper - lower) * success_open_ratio
    required_delta = success_qpos - door0
    if abs(required_delta) <= 1e-8:
        required_delta = upper - lower

    def target_progress() -> tuple[float, float, float]:
        current = float(np.asarray(world.object.get_qpos()).reshape(-1)[target_joint_idx])
        signed = float(np.sign(required_delta) * (current - door0))
        normalized = float((current - door0) / required_delta)
        return current, signed, normalized
    n_policy = 0
    policy_inference_seconds = 0.0
    n_teleport = 0
    exec_n = max(1, int(args.execute_steps))

    for step_i in range(int(args.max_policy_steps)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        act = policy_act(
            policy,
            xyz,
            state_hist,
            device,
            use_gt_affordance=(args.affordance_source == "gt"),
            aff_gt=aff_gt,
            contact_info=contact_info,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        policy_inference_seconds += time.perf_counter() - inference_start
        n_policy += 1
        # predict 8, execute first exec_n (default 4); no ensemble
        chunk = np.asarray(act, dtype=np.float32)[:exec_n]
        for q_raw in chunk:
            q_cmd = np.asarray(q_raw, dtype=np.float32).copy()
            q_cmd[7:9] = np.clip(q_cmd[7:9], 0.0, FINGER_OPEN)
            if grasped:
                q_cmd[7:9] = FINGER_CLOSED

            interpolate_execute(
                world,
                q_cmd,
                recorder,
                interp_steps=int(args.interp_steps),
                physics_per_substep=int(args.physics_per_substep),
                capture_every=int(args.gif_render_every),
            )
            n_teleport += 1

            fb = read_target_contact_feedback(world, link_name)
            if (not grasped) and bool(fb.gripper_target_contact):
                grasped = True
                grasp_step = n_teleport
                q_now = qpos9_from_robot(world)
                q_now[7:9] = FINGER_CLOSED
                interpolate_execute(
                    world,
                    q_now,
                    recorder,
                    interp_steps=max(2, int(args.interp_steps) // 2),
                    physics_per_substep=int(args.physics_per_substep),
                    capture_every=int(args.gif_render_every),
                )
                print(f"  [grasp latch] step={grasp_step} links={fb.target_robot_links}", flush=True)

            state_hist.append(build_state(qpos9_from_robot(world), grasped))

        door_q, _, normalized_progress = target_progress()
        if normalized_progress >= float(args.success_progress_threshold):
            print(
                f"  target progress reached: {door0:.3f} -> {door_q:.3f} "
                f"normalized={normalized_progress:.3f}",
                flush=True,
            )
            break

    # force a few last frames
    for _ in range(8):
        recorder.capture(force=True)
    gif_path = recorder.save()
    door_q, signed_progress, normalized_progress = target_progress()
    world.close()

    summary = {
        "obj_key": obj_key,
        "replay_id": replay_id,
        "source_replay_id": int(replay_row["source_replay_id"]) if replay_row is not None else None,
        "repeat": str(replay_row["repeat"]) if replay_row is not None else demo_dir.parent.name,
        "base": str(replay_row["base"]) if replay_row is not None else demo_dir.name,
        "trajectory": packed["traj_path"].name,
        "condition_variant": condition_variant,
        "affordance_adapter": str(policy.affordance_adapter_mode),
        "affordance_aux_weight": float(policy.affordance_aux_weight),
        "affordance_intervention": str(args.affordance_intervention),
        "affordance_intervention_metadata": intervention_metadata,
        "contact_condition": str(policy.contact_condition),
        "contact_intervention": (
            str(args.contact_intervention) if policy.contact_condition == "coordinate" else None
        ),
        "contact_metadata": contact_info,
        "demo_dir": str(demo_dir),
        "gif": str(gif_path) if gif_path else None,
        "grasped": grasped,
        "grasp_step": grasp_step,
        "door_q0": door0,
        "door_qf": door_q,
        "door_delta": abs(door_q - door0),
        "target_joint_index": target_joint_idx,
        "target_joint_lower": lower,
        "target_joint_upper": upper,
        "initial_open_ratio": float((door0 - lower) / (upper - lower)),
        "success_open_ratio": success_open_ratio,
        "success_target_qpos": success_qpos,
        "signed_progress": signed_progress,
        "normalized_progress": normalized_progress,
        "success": bool(normalized_progress >= float(args.success_progress_threshold)),
        "n_policy": n_policy,
        "policy_inference_seconds": policy_inference_seconds,
        "mean_policy_inference_ms": 1000.0 * policy_inference_seconds / max(n_policy, 1),
        "n_teleport": n_teleport,
        "execute_steps": int(args.execute_steps),
    }
    print(
        f"  done grasped={grasped} normalized_progress={normalized_progress:.4f} "
        f"success={summary['success']} gif={gif_path}",
        flush=True,
    )
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", type=Path, default=ROOT / "data" / "joint_door.zarr")
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "stage2_tiny100" / "last.pth")
    p.add_argument("--out_dir", type=Path, default=ROOT / "runs" / "stage2_tiny100" / "eval_teleport")
    p.add_argument("--num_objects", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=str, default="2")
    p.add_argument("--affordance_source", choices=["infer", "gt"], default="infer")
    p.add_argument(
        "--affordance_intervention",
        choices=("correct", "zero", "swap", "shift"),
        default="correct",
        help="evaluation-only intervention applied to the selected GT affordance map",
    )
    p.add_argument(
        "--contact_sidecar",
        type=Path,
        default=None,
        help="override contact sidecar stored in a contact-conditioned checkpoint",
    )
    p.add_argument(
        "--contact_intervention",
        choices=("correct", "wrong", "zero"),
        default="correct",
        help="evaluation-only trajectory-contact intervention",
    )
    p.add_argument("--max_policy_steps", type=int, default=40, help="number of replan cycles")
    p.add_argument(
        "--execute_steps",
        type=int,
        default=4,
        help="execute first K actions from each 8-step prediction (no ensemble)",
    )
    p.add_argument(
        "--interp_steps",
        type=int,
        default=4,
        help="linear qpos interpolation substeps between commands",
    )
    p.add_argument(
        "--physics_per_substep",
        type=int,
        default=1,
        help="physics steps after each interpolation substep",
    )
    p.add_argument("--physics_steps_per_action", type=int, default=8, help="deprecated; use interp/physics_per_substep")
    p.add_argument("--settle_steps", type=int, default=20)
    p.add_argument("--success_door_delta", type=float, default=0.35)
    p.add_argument("--success_open_ratio", type=float, default=0.4)
    p.add_argument("--success_progress_threshold", type=float, default=1.0)
    p.add_argument("--gif_width", type=int, default=480)
    p.add_argument("--gif_height", type=int, default=360)
    p.add_argument("--gif_fps", type=int, default=12)
    p.add_argument("--gif_max_frames", type=int, default=180)
    p.add_argument("--gif_render_every", type=int, default=2)
    p.add_argument(
        "--obj_keys",
        type=str,
        default="",
        help="comma-separated obj_keys; empty => random from --split pool",
    )
    p.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="object pool for random sampling / replay lookup",
    )
    p.add_argument(
        "--train_subset_json",
        type=Path,
        default=ROOT / "runs" / "stage2_tiny100" / "train_subset.json",
        help="if set and exists with --split train, restrict to these tiny100 objects",
    )
    p.add_argument(
        "--target_manifest",
        type=Path,
        default=None,
        help="JSON manifest; use <split>.target_keys when --obj_keys is empty",
    )
    p.add_argument(
        "--manifest_partition",
        type=str,
        default=None,
        help="manifest partition to evaluate; defaults to --split",
    )
    p.add_argument(
        "--replay_manifest",
        type=Path,
        default=ROOT / "data" / "joint_from_data_cam_replays.json",
        help="source replay metadata used to align point cloud and simulator initial state",
    )
    return p.parse_args()


def main():
    args = parse_args()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool = list_split_objects(args.zarr, split=args.split)
    key_to_rid = {k: rid for k, rid in pool}

    # tiny100: restrict train pool to the 10 training objects if json exists
    subset_keys = None
    if (
        args.split == "train"
        and args.train_subset_json is not None
        and Path(args.train_subset_json).is_file()
    ):
        detail = json.loads(Path(args.train_subset_json).read_text())
        subset_keys = [d["obj"] for d in detail]
        pool = [(k, key_to_rid[k]) for k in subset_keys if k in key_to_rid]
        key_to_rid = {k: rid for k, rid in pool}
        print(f"using train_subset ({len(pool)} objs) from {args.train_subset_json}", flush=True)

    if args.obj_keys.strip():
        chosen_keys = [k.strip() for k in args.obj_keys.split(",") if k.strip()]
    elif args.target_manifest is not None:
        manifest = json.loads(args.target_manifest.read_text())
        partition = args.manifest_partition or args.split
        try:
            chosen_keys = [str(item) for item in manifest[partition]["target_keys"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{args.target_manifest} has no {partition}.target_keys") from exc
    else:
        if len(pool) < args.num_objects:
            raise RuntimeError(f"{args.split} objects={len(pool)} < num_objects={args.num_objects}")
        idxs = rng.choice(len(pool), size=args.num_objects, replace=False)
        chosen_keys = [pool[int(i)][0] for i in idxs]

    print(f"ckpt={args.ckpt} device={device} split={args.split} objects={chosen_keys}", flush=True)
    policy = load_policy(args.ckpt, device)
    contact_sidecar = None
    if policy.contact_condition == "coordinate":
        sidecar_path = args.contact_sidecar or getattr(policy, "contact_sidecar_path", None)
        if sidecar_path is None:
            raise ValueError("contact-conditioned checkpoint has no contact sidecar path")
        packed_sidecar = np.load(str(sidecar_path), allow_pickle=False)
        contact_sidecar = {key: packed_sidecar[key] for key in packed_sidecar.files}
    zarr_root = zarr.open(str(args.zarr), mode="r")
    replay_rows = json.loads(args.replay_manifest.read_text())
    source_replay_ids = (
        np.asarray(zarr_root["meta"]["source_replay_id"][:], dtype=np.int32)
        if "source_replay_id" in zarr_root["meta"]
        else None
    )

    summaries = []
    for obj_key in chosen_keys:
        target_seed = stable_target_seed(args.seed, obj_key)
        target_rng = np.random.default_rng(target_seed)
        torch.manual_seed(target_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(target_seed)
        if obj_key not in key_to_rid:
            # allow explicit obj_keys even if not in restricted pool: resolve from full split
            full = dict(list_split_objects(args.zarr, split=args.split))
            if obj_key not in full:
                # try both splits
                full = {**dict(list_split_objects(args.zarr, "train")), **dict(list_split_objects(args.zarr, "val"))}
            if obj_key not in full:
                print(f"[skip] {obj_key} not found in zarr", flush=True)
                continue
            rid = full[obj_key]
        else:
            rid = key_to_rid[obj_key]
        out_gif = args.out_dir / f"{obj_key}_seed{args.seed}.gif"
        replay_row = None
        if source_replay_ids is not None:
            source_replay_id = int(source_replay_ids[rid])
            replay_row = dict(replay_rows[source_replay_id])
            replay_row["source_replay_id"] = source_replay_id
            if str(replay_row.get("obj_key")) != obj_key:
                raise RuntimeError(
                    f"replay alignment mismatch: {obj_key} vs source row {replay_row.get('obj_key')}"
                )
        try:
            summary = run_one_object(
                policy=policy,
                zarr_root=zarr_root,
                obj_key=obj_key,
                replay_id=rid,
                out_gif=out_gif,
                device=device,
                args=args,
                rng=target_rng,
                replay_row=replay_row,
                contact_sidecar=contact_sidecar,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[error] {obj_key}: {e}", flush=True)
            summaries.append({"obj_key": obj_key, "error": str(e)})

    summary_path = args.out_dir / f"summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    complete = [item for item in summaries if "error" not in item]
    success = [item for item in complete if bool(item["success"])]
    online_metrics = {
        "n_requested": len(chosen_keys),
        "n_completed": len(complete),
        "n_success": len(success),
        "success_rate": float(len(success) / len(complete)) if complete else float("nan"),
        "grasp_rate": float(sum(bool(item["grasped"]) for item in complete) / len(complete)) if complete else float("nan"),
        "mean_door_delta": float(np.mean([item["door_delta"] for item in complete])) if complete else float("nan"),
        "mean_signed_progress": float(np.mean([item["signed_progress"] for item in complete])) if complete else float("nan"),
        "mean_normalized_progress": float(np.mean([item["normalized_progress"] for item in complete])) if complete else float("nan"),
        "mean_policy_inference_ms": float(
            sum(item["policy_inference_seconds"] for item in complete)
            / max(sum(item["n_policy"] for item in complete), 1)
            * 1000.0
        ) if complete else float("nan"),
        "success_door_delta": float(args.success_door_delta),
        "success_progress_threshold": float(args.success_progress_threshold),
        "metric": "signed_normalized_target_joint_progress",
        "objects": summaries,
    }
    (args.out_dir / "online_metrics.json").write_text(json.dumps(online_metrics, indent=2) + "\n")
    print(f"\nDONE -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()

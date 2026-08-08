#!/usr/bin/env python3
"""Print joint_door.zarr statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", type=Path, default=Path("data/joint_door.zarr"))
    args = p.parse_args()
    root = zarr.open(str(args.zarr), mode="r")
    pc = root["data"]["point_cloud"]
    state = root["data"]["state"]
    action = root["data"]["action"]
    ends = np.asarray(root["meta"]["episode_ends"][:])
    rids = np.asarray(root["meta"]["episode_replay_ids"][:])
    splits = np.asarray(root["meta"]["replay_split"][:])
    keys = list(root["meta"]["replay_obj_keys"][:])
    print("point_cloud", pc.shape, pc.dtype)
    print("state", state.shape, "action", action.shape)
    print("n_replays", pc.shape[0], "n_trajs", len(ends), "n_steps", state.shape[0])
    print("n_obj_keys", len(set(keys)))
    print("train_replays", int((splits == 0).sum()), "val_replays", int((splits == 1).sum()))
    print("trajs_per_replay mean", float(np.mean(np.bincount(rids, minlength=pc.shape[0]))))
    print("affordance mean/max", float(np.mean(pc[:, :, 3])), float(np.max(pc[:, :, 3])))
    # shared-pcd check: unique rids count == n_replays used
    print("unique episode_replay_ids", len(np.unique(rids)))
    summary_path = args.zarr.parent / "joint_door_summary.json"
    if summary_path.is_file():
        print("summary:", summary_path.read_text()[:500])


if __name__ == "__main__":
    main()

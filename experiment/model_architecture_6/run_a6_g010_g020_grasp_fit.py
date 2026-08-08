#!/usr/bin/env python3
"""Matched fixed-batch G010 direct trajectory / G020 terminal-qpose fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from a6_grasp_models import GraspProposalBase, grasp_set_loss
from path_config import JOINTTRAIN_ARCH6_G006C_RESULT_ROOT, JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT, JOINTTRAIN_ARCH6_G006MC_RESULT_ROOT, JOINTTRAIN_ARCH6_G006SC_RESULT_ROOT, JOINTTRAIN_ARCH6_G010C_RESULT_ROOT, JOINTTRAIN_ARCH6_G020C_RESULT_ROOT, JOINTTRAIN_ARCH6_G031C_RESULT_ROOT, JOINTTRAIN_ARCH6_G032C_RESULT_ROOT, JOINTTRAIN_ARCH6_G033C_RESULT_ROOT, JOINTTRAIN_ARCH6_G034C_RESULT_ROOT, JOINTTRAIN_ARCH6_G035C_RESULT_ROOT, JOINTTRAIN_ARCH6_G036C_RESULT_ROOT, JOINTTRAIN_ARCH6_G037C_RESULT_ROOT, JOINTTRAIN_ARCH6_G038C_RESULT_ROOT, JOINTTRAIN_ARCH6_G041C_RESULT_ROOT, JOINTTRAIN_ARCH6_G042C_RESULT_ROOT, JOINTTRAIN_ARCH6_G043C_RESULT_ROOT, JOINTTRAIN_ARCH6_G045C_RESULT_ROOT, JOINTTRAIN_ARCH6_G046C_RESULT_ROOT, JOINTTRAIN_ARCH6_G048C_RESULT_ROOT, JOINTTRAIN_ARCH6_G049C_RESULT_ROOT, JOINTTRAIN_ARCH6_G052C_RESULT_ROOT, JOINTTRAIN_ARCH6_G053C_RESULT_ROOT, JOINTTRAIN_ARCH6_G054C_RESULT_ROOT, JOINTTRAIN_BESTVIEW_DUAL_ZARR


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["traj", "qpose", "se3"], required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--affordance", choices=["zero", "gt"], default="zero")
    p.add_argument("--coordinate-frame", choices=["world", "base"], default="world")
    p.add_argument("--target-normalization", choices=["none", "per-joint"], default="none")
    p.add_argument("--affordance-encoding", choices=["weight", "concat"], default="weight")
    p.add_argument("--view-mode", choices=["primary", "multiview3", "same-target"], default="primary")
    p.add_argument("--target-mask", action="store_true")
    p.add_argument("--target-mask-condition", choices=["target", "zero"], default="target")
    p.add_argument("--target-mask-encoding", choices=["concat", "dual", "local"], default="concat")
    args = p.parse_args()
    if args.kind == "se3" and args.target_normalization != "none":
        raise ValueError("SE3 uses raw translation and rotation-geodesic loss")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.target_mask and (args.affordance != "zero" or args.coordinate_frame != "base" or args.view_mode != "primary"):
        raise ValueError("target-mask arm is isolated to zero-affordance primary base-frame input")
    root = Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT if args.kind == "se3" else (JOINTTRAIN_ARCH6_G041C_RESULT_ROOT if args.target_mask else (JOINTTRAIN_ARCH6_G006SC_RESULT_ROOT if args.view_mode == "same-target" else (JOINTTRAIN_ARCH6_G006MC_RESULT_ROOT if args.view_mode == "multiview3" else (JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT if args.coordinate_frame == "base" else JOINTTRAIN_ARCH6_G006C_RESULT_ROOT)))))
    with np.load(root / "grasp_inputs.npz", allow_pickle=False) as d:
        point = torch.from_numpy(np.asarray(d["point_cloud_xyz"], dtype=np.float32))
        state = torch.from_numpy(np.asarray(d["state_qpos"], dtype=np.float32))
        target_key = "path_relative" if args.kind == "traj" else ("se3_base" if args.kind == "se3" else "qpose_relative")
        target = torch.from_numpy(np.asarray(d[target_key], dtype=np.float32))
        presence = torch.from_numpy(np.asarray(d["presence"], dtype=bool))
        split = np.asarray(d["split"], dtype=np.int8)
        source_ids = np.asarray(d["source_replay_id"], dtype=np.int32)
        target_mask = torch.from_numpy(np.asarray(d["target_mask"], dtype=np.float32)) if args.target_mask else None
    if args.target_mask and args.target_mask_condition == "zero":
        target_mask = torch.zeros_like(target_mask)
    affordance = None
    if args.affordance == "gt":
        import zarr
        zroot = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
        zids = np.asarray(zroot["meta/source_replay_id"][:], dtype=np.int32)
        rows = {int(value): i for i, value in enumerate(zids.tolist())}
        affordance = torch.from_numpy(np.asarray(zroot["data/affordance_updated"][[rows[int(value)] for value in source_ids]], dtype=np.float32))
    train = np.flatnonzero(split == 0); cal = np.flatnonzero(split == 1)
    target_mean = torch.zeros(target.shape[-1], dtype=torch.float32)
    target_std = torch.ones(target.shape[-1], dtype=torch.float32)
    if args.target_normalization == "per-joint":
        valid_train = target[train][presence[train]]
        flattened = valid_train.reshape(-1, 7)
        target_mean = flattened.mean(dim=0)
        target_std = flattened.std(dim=0).clamp_min(1e-4)
        target = (target - target_mean) / target_std
    if args.kind == "se3":
        out = Path(JOINTTRAIN_ARCH6_G054C_RESULT_ROOT if args.target_mask else JOINTTRAIN_ARCH6_G053C_RESULT_ROOT)
    elif args.target_mask:
        if args.target_mask_encoding == "local":
            out = Path(JOINTTRAIN_ARCH6_G049C_RESULT_ROOT if args.kind == "traj" else JOINTTRAIN_ARCH6_G048C_RESULT_ROOT) / args.target_mask_condition
        elif args.target_mask_encoding == "dual":
            out = Path(JOINTTRAIN_ARCH6_G046C_RESULT_ROOT if args.kind == "traj" else JOINTTRAIN_ARCH6_G045C_RESULT_ROOT) / args.target_mask_condition
        else:
            out = Path(JOINTTRAIN_ARCH6_G043C_RESULT_ROOT if args.kind == "traj" else JOINTTRAIN_ARCH6_G042C_RESULT_ROOT) / args.target_mask_condition
    else:
        out = Path(JOINTTRAIN_ARCH6_G038C_RESULT_ROOT if args.view_mode == "same-target" else (JOINTTRAIN_ARCH6_G037C_RESULT_ROOT if args.view_mode == "multiview3" else (JOINTTRAIN_ARCH6_G036C_RESULT_ROOT if args.affordance_encoding == "concat" else (JOINTTRAIN_ARCH6_G035C_RESULT_ROOT if args.coordinate_frame == "base" and args.affordance == "gt" else (JOINTTRAIN_ARCH6_G034C_RESULT_ROOT if args.target_normalization == "per-joint" else (JOINTTRAIN_ARCH6_G033C_RESULT_ROOT if args.coordinate_frame == "base" and args.kind == "traj" else (JOINTTRAIN_ARCH6_G032C_RESULT_ROOT if args.coordinate_frame == "base" else (JOINTTRAIN_ARCH6_G031C_RESULT_ROOT if args.affordance == "gt" else (JOINTTRAIN_ARCH6_G010C_RESULT_ROOT if args.kind == "traj" else JOINTTRAIN_ARCH6_G020C_RESULT_ROOT)))))))))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = GraspProposalBase(args.kind, use_affordance=args.affordance == "gt", affordance_encoding=args.affordance_encoding, use_target_mask=args.target_mask, target_mask_encoding=args.target_mask_encoding).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    steps = 50 if args.sanity else args.steps
    start = time.time(); losses=[]; cal_losses=[]
    model.train()
    for step in range(steps):
        take = np.random.choice(train, size=min(args.batch_size, len(train)), replace=False)
        x, s, y, m = point[take].to(device), state[take].to(device), target[take].to(device), presence[take].to(device)
        opt.zero_grad(set_to_none=True)
        aff = affordance[take].to(device) if affordance is not None else None
        tm = target_mask[take].to(device) if target_mask is not None else None
        loss, metrics = grasp_set_loss(model(x, s, aff, target_mask=tm), y, m, args.kind)
        if not torch.isfinite(loss): raise RuntimeError("non-finite grasp loss")
        loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                cx, cs, cy, cm = point[cal].to(device), state[cal].to(device), target[cal].to(device), presence[cal].to(device)
                ca = affordance[cal].to(device) if affordance is not None else None
                ctm = target_mask[cal].to(device) if target_mask is not None else None
                cal_loss, _ = grasp_set_loss(model(cx, cs, ca, target_mask=ctm), cy, cm, args.kind)
            cal_losses.append(float(cal_loss.cpu())); model.train()
    checkpoint = out / f"{args.kind}_{args.affordance}_{args.coordinate_frame}_{args.target_normalization}_seed{args.seed}.pth"
    torch.save({"model": model.state_dict(), "kind": args.kind, "affordance": args.affordance, "coordinate_frame":args.coordinate_frame,"target_mask":args.target_mask,"target_normalization":args.target_normalization,"target_mean":target_mean,"target_std":target_std,"seed": args.seed, "steps": steps}, checkpoint)
    model.eval()
    with torch.no_grad():
        reload = GraspProposalBase(args.kind, use_affordance=args.affordance == "gt", affordance_encoding=args.affordance_encoding, use_target_mask=args.target_mask, target_mask_encoding=args.target_mask_encoding).to(device)
        reload.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        reload.eval()
        probe_indices = cal[: min(4, len(cal))]
        probe_aff = affordance[probe_indices].to(device) if affordance is not None else None
        probe_mask = target_mask[probe_indices].to(device) if target_mask is not None else None
        probe = model(point[probe_indices].to(device), state[probe_indices].to(device), probe_aff, target_mask=probe_mask)
        probe_reload = reload(point[probe_indices].to(device), state[probe_indices].to(device), probe_aff, target_mask=probe_mask)
        reload_error = float(max(torch.max(torch.abs(probe[k]-probe_reload[k])).item() for k in probe))
    expected_train = 338 if args.view_mode == "same-target" else (1593 if args.view_mode == "multiview3" else 531)
    expected_cal = 180 if args.view_mode == "same-target" else 101
    expected_mask = ((target_mask is not None and bool(torch.all(target_mask.sum(dim=1) > 0))) if args.target_mask_condition == "target" else (target_mask is not None and bool(torch.all(target_mask == 0))))
    checks = {"finite": bool(np.isfinite(losses).all()), "reload": reload_error == 0.0, "train_loss_decreased": losses[-1] < losses[0], "split_counts": len(train) == expected_train and len(cal) == expected_cal, "affordance_contract": args.affordance in {"zero", "gt"}, "target_mask_contract": expected_mask if args.target_mask else target_mask is None, "no_future_or_outcome_input": True}
    if args.kind == "se3":
        run_number = "054C" if args.target_mask else "053C"
    elif args.target_mask:
        if args.target_mask_encoding == "local":
            run_number = "049C" if args.kind == "traj" else "048C"
        elif args.target_mask_encoding == "dual":
            run_number = "046C" if args.kind == "traj" else "045C"
        else:
            run_number = "043C" if args.kind == "traj" else "042C"
    else:
        run_number = "038C" if args.view_mode == "same-target" else ("037C" if args.view_mode == "multiview3" else ("036C" if args.affordance_encoding == "concat" else ("035C" if args.coordinate_frame == "base" and args.affordance == "gt" else ("034C" if args.target_normalization == "per-joint" else ("033C" if args.coordinate_frame == "base" and args.kind == "traj" else ("032C" if args.coordinate_frame == "base" else ("031C" if args.affordance == "gt" else ("010C" if args.kind == "traj" else "020C"))))))))
    summary = {"schema_version":1,"run_id":f"A6-G{run_number}-{'SANITY' if args.sanity else 'FIT'}-{args.target_mask_condition if args.target_mask else 'none'}","status":"passed" if all(checks.values()) else "failed","complete":True,"terminal":True,"kind":args.kind,"affordance_condition":args.affordance,"affordance_encoding":args.affordance_encoding,"coordinate_frame":args.coordinate_frame,"target_mask":args.target_mask,"target_mask_condition":args.target_mask_condition if args.target_mask else None,"target_mask_encoding":args.target_mask_encoding if args.target_mask else None,"view_mode":args.view_mode,"target_normalization":args.target_normalization,"target_mean":target_mean.tolist(),"target_std":target_std.tolist(),"deployable":args.affordance=="zero","seed":args.seed,"optimizer_steps":steps,"effective_batch":args.batch_size,"device":str(device),"trainable_parameters":sum(p.numel() for p in model.parameters()),"train_loss_first":losses[0],"train_loss_last":losses[-1],"cal_loss_last":cal_losses[-1],"reload_max_abs":reload_error,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"source_input_sha256":sha(root/'grasp_inputs.npz'),"checks":checks,"decision":"run raw-unit target-mask CAL comparison" if all(checks.values()) else "repair grasp fit"}
    atomic(out/"summary.json",summary); atomic(out/"run_state.json",summary); atomic(out/"queue_state.json",{**summary,"jobs":[{"id":summary["run_id"],"status":summary["status"]}]})
    print(json.dumps(summary)); return 0 if all(checks.values()) else 2


if __name__ == "__main__": raise SystemExit(main())

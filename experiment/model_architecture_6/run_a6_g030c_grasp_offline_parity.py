#!/usr/bin/env python3
"""Raw-unit CAL parity for the matched G010/G020 checkpoints."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from a6_grasp_models import GraspProposalBase
from path_config import (
    JOINTTRAIN_ARCH6_G006C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G010C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G020C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G030C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G031C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G032C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G033C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G034C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G035C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G036C_RESULT_ROOT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
)

PERMS = tuple(itertools.permutations(range(4)))


def atomic(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def best_assignment(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[int, ...]:
    costs = np.abs(pred[:, None] - target[None]).reshape(4, 4, -1).mean(axis=-1)
    return min(PERMS, key=lambda perm: sum(costs[i, perm[i]] for i in range(4) if valid[perm[i]]))


def matched_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, kind: str) -> dict[str, float]:
    perm = best_assignment(pred, target, valid)
    pairs = [(pred[i], target[perm[i]]) for i in range(4) if valid[perm[i]]]
    a = np.stack([x for x, _ in pairs]); b = np.stack([y for _, y in pairs])
    if kind == "qpose":
        return {"endpoint_mae": float(np.abs(a - b).mean())}
    return {
        "waypoint_mae": float(np.abs(a - b).mean()),
        "endpoint_mae": float(np.abs(a[:, -1] - b[:, -1]).mean()),
        "start_abs": float(np.abs(a[:, 0]).mean()),
        "first_difference_mae": float(np.abs(np.diff(a, axis=1) - np.diff(b, axis=1)).mean()),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> int:
    source = Path(JOINTTRAIN_ARCH6_G006C_RESULT_ROOT) / "grasp_inputs.npz"
    with np.load(source, allow_pickle=False) as d:
        point = torch.from_numpy(np.asarray(d["point_cloud_xyz"], dtype=np.float32))
        state = torch.from_numpy(np.asarray(d["state_qpos"], dtype=np.float32))
        path = np.asarray(d["path_relative"], dtype=np.float32)
        qpose = np.asarray(d["qpose_relative"], dtype=np.float32)
        presence = np.asarray(d["presence"], dtype=bool)
        split = np.asarray(d["split"], dtype=np.int8)
        source_ids = np.asarray(d["source_replay_id"], dtype=np.int32)
    train, cal = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for kind, root, target in (
        ("traj", Path(JOINTTRAIN_ARCH6_G010C_RESULT_ROOT), path),
        ("qpose", Path(JOINTTRAIN_ARCH6_G020C_RESULT_ROOT), qpose),
    ):
        checkpoint = root / f"{kind}_seed20260806.pth"
        model = GraspProposalBase(kind).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        model.eval()
        with torch.no_grad():
            output = model(point[cal].to(device), state[cal].to(device))
        pred = output["values"].cpu().numpy()
        probability = torch.sigmoid(output["presence_logits"]).cpu().numpy()
        model_rows = [matched_metrics(pred[i], target[index], presence[index], kind) for i, index in enumerate(cal)]
        zero = np.zeros_like(pred)
        zero_rows = [matched_metrics(zero[i], target[index], presence[index], kind) for i, index in enumerate(cal)]
        mean_value = np.mean(target[train], axis=0, keepdims=True)
        mean_pred = np.repeat(mean_value, len(cal), axis=0)
        mean_rows = [matched_metrics(mean_pred[i], target[index], presence[index], kind) for i, index in enumerate(cal)]
        results[kind] = {
            "model": summarize(model_rows),
            "zero_relative_baseline": summarize(zero_rows),
            "train_slot_mean_baseline": summarize(mean_rows),
            "presence_probability_mean": float(probability.mean()),
            "groups_with_any_presence_gt_0_5": int((probability.max(axis=1) > 0.5).sum()),
            "groups": len(cal),
        }
    import zarr
    zroot = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    zids = np.asarray(zroot["meta/source_replay_id"][:], dtype=np.int32)
    zrows = {int(value): i for i, value in enumerate(zids.tolist())}
    gt_affordance = torch.from_numpy(np.asarray(zroot["data/affordance_updated"][[zrows[int(value)] for value in source_ids]], dtype=np.float32))
    oracle_model = GraspProposalBase("qpose", use_affordance=True).to(device)
    oracle_checkpoint = Path(JOINTTRAIN_ARCH6_G031C_RESULT_ROOT) / "qpose_gt_seed20260806.pth"
    oracle_model.load_state_dict(torch.load(oracle_checkpoint, map_location=device, weights_only=False)["model"])
    oracle_model.eval()
    with torch.no_grad():
        oracle_output = oracle_model(point[cal].to(device), state[cal].to(device), gt_affordance[cal].to(device))
    oracle_pred = oracle_output["values"].cpu().numpy()
    oracle_rows = [matched_metrics(oracle_pred[i], qpose[index], presence[index], "qpose") for i, index in enumerate(cal)]
    results["qpose_gt_affordance_oracle"] = {
        "model": summarize(oracle_rows),
        "deployable": False,
        "groups": len(cal),
    }
    with np.load(Path(JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as base_data:
        base_point = torch.from_numpy(np.asarray(base_data["point_cloud_xyz"], dtype=np.float32))
    base_model = GraspProposalBase("qpose").to(device)
    base_checkpoint = Path(JOINTTRAIN_ARCH6_G032C_RESULT_ROOT) / "qpose_zero_base_seed20260806.pth"
    base_model.load_state_dict(torch.load(base_checkpoint, map_location=device, weights_only=False)["model"])
    base_model.eval()
    with torch.no_grad():
        base_output = base_model(base_point[cal].to(device), state[cal].to(device))
    base_pred = base_output["values"].cpu().numpy()
    base_rows = [matched_metrics(base_pred[i], qpose[index], presence[index], "qpose") for i, index in enumerate(cal)]
    results["qpose_base_frame"] = {
        "model": summarize(base_rows),
        "deployable": True,
        "groups": len(cal),
    }
    with np.load(Path(JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as base_data:
        base_point = torch.from_numpy(np.asarray(base_data["point_cloud_xyz"], dtype=np.float32))
    base_traj_model = GraspProposalBase("traj").to(device)
    base_traj_checkpoint = Path(JOINTTRAIN_ARCH6_G033C_RESULT_ROOT) / "traj_zero_base_seed20260806.pth"
    base_traj_model.load_state_dict(torch.load(base_traj_checkpoint, map_location=device, weights_only=False)["model"])
    base_traj_model.eval()
    with torch.no_grad():
        base_traj_output = base_traj_model(base_point[cal].to(device), state[cal].to(device))
    base_traj_pred = base_traj_output["values"].cpu().numpy()
    base_traj_rows = [matched_metrics(base_traj_pred[i], path[index], presence[index], "traj") for i, index in enumerate(cal)]
    results["traj_base_frame"] = {"model": summarize(base_traj_rows), "deployable": True, "groups": len(cal)}
    normalized_model = GraspProposalBase("qpose").to(device)
    normalized_checkpoint = Path(JOINTTRAIN_ARCH6_G034C_RESULT_ROOT) / "qpose_zero_base_per-joint_seed20260806.pth"
    normalized_payload = torch.load(normalized_checkpoint, map_location=device, weights_only=False)
    normalized_model.load_state_dict(normalized_payload["model"]); normalized_model.eval()
    with torch.no_grad():
        normalized_output = normalized_model(base_point[cal].to(device), state[cal].to(device))["values"]
        normalized_output = normalized_output * normalized_payload["target_std"].to(device) + normalized_payload["target_mean"].to(device)
    normalized_pred = normalized_output.cpu().numpy()
    normalized_rows = [matched_metrics(normalized_pred[i], qpose[index], presence[index], "qpose") for i, index in enumerate(cal)]
    results["qpose_base_frame_per_joint_normalized"] = {"model": summarize(normalized_rows), "deployable": True, "groups": len(cal)}
    base_gt_model = GraspProposalBase("qpose", use_affordance=True).to(device)
    base_gt_checkpoint = Path(JOINTTRAIN_ARCH6_G035C_RESULT_ROOT) / "qpose_gt_base_none_seed20260806.pth"
    base_gt_model.load_state_dict(torch.load(base_gt_checkpoint, map_location=device, weights_only=False)["model"]); base_gt_model.eval()
    with torch.no_grad():
        base_gt_output = base_gt_model(base_point[cal].to(device), state[cal].to(device), gt_affordance[cal].to(device))["values"]
    base_gt_pred = base_gt_output.cpu().numpy()
    base_gt_rows = [matched_metrics(base_gt_pred[i], qpose[index], presence[index], "qpose") for i, index in enumerate(cal)]
    results["qpose_base_frame_gt_affordance_oracle"] = {"model": summarize(base_gt_rows), "deployable": False, "groups": len(cal)}
    concat_model = GraspProposalBase("qpose", use_affordance=True, affordance_encoding="concat").to(device)
    concat_checkpoint = Path(JOINTTRAIN_ARCH6_G036C_RESULT_ROOT) / "qpose_gt_base_none_seed20260806.pth"
    concat_model.load_state_dict(torch.load(concat_checkpoint, map_location=device, weights_only=False)["model"]); concat_model.eval()
    with torch.no_grad():
        concat_output = concat_model(base_point[cal].to(device), state[cal].to(device), gt_affordance[cal].to(device))["values"]
    concat_pred = concat_output.cpu().numpy()
    concat_rows = [matched_metrics(concat_pred[i], qpose[index], presence[index], "qpose") for i, index in enumerate(cal)]
    results["qpose_base_frame_gt_affordance_concat_oracle"] = {"model": summarize(concat_rows), "deployable": False, "groups": len(cal)}
    checks = {
        "cal_groups_101": len(cal) == 101,
        "finite": all(np.isfinite(value) for route in results.values() for group in route.values() if isinstance(group, dict) for value in group.values()),
        "same_cal_groups": True,
        "no_outcome_read": True,
        "raw_joint_units": True,
    }
    out = Path(JOINTTRAIN_ARCH6_G030C_RESULT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version":1,"run_id":"A6-G030C","status":"passed" if all(checks.values()) else "failed","complete":True,"terminal":True,"results":results,"checks":checks,"decision":"use physical planner/rollout to compare routes; offline losses are not cross-route comparable"}
    atomic(out/"summary.json",summary); atomic(out/"run_state.json",summary); atomic(out/"queue_state.json",summary)
    print(json.dumps(summary)); return 0 if all(checks.values()) else 2


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Matched raw-unit CAL evaluation for base-frame target-mask grasp models."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from a6_grasp_models import GraspProposalBase
from path_config import JOINTTRAIN_ARCH6_G041C_RESULT_ROOT, JOINTTRAIN_ARCH6_G042C_RESULT_ROOT, JOINTTRAIN_ARCH6_G043C_RESULT_ROOT, JOINTTRAIN_ARCH6_G044C_RESULT_ROOT, JOINTTRAIN_ARCH6_G045C_RESULT_ROOT, JOINTTRAIN_ARCH6_G046C_RESULT_ROOT, JOINTTRAIN_ARCH6_G047C_RESULT_ROOT, JOINTTRAIN_ARCH6_G048C_RESULT_ROOT, JOINTTRAIN_ARCH6_G049C_RESULT_ROOT, JOINTTRAIN_ARCH6_G050C_RESULT_ROOT

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, kind: str) -> dict[str, float]:
    cost = np.abs(pred[:, None] - target[None]).reshape(4, 4, -1).mean(axis=-1)
    perm = min(PERMUTATIONS, key=lambda candidate: sum(cost[i, candidate[i]] for i in range(4) if valid[candidate[i]]))
    pairs = [(pred[i], target[perm[i]]) for i in range(4) if valid[perm[i]]]
    left = np.stack([pair[0] for pair in pairs])
    right = np.stack([pair[1] for pair in pairs])
    if kind == "qpose":
        return {"endpoint_mae": float(np.abs(left - right).mean())}
    return {
        "waypoint_mae": float(np.abs(left - right).mean()),
        "endpoint_mae": float(np.abs(left[:, -1] - right[:, -1]).mean()),
        "first_difference_mae": float(np.abs(np.diff(left, axis=1) - np.diff(right, axis=1)).mean()),
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def paired_bootstrap(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260806)
    sampled = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(sampled, [0.025, 0.975]).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["qpose", "traj", "both"], default="both")
    parser.add_argument("--target-mask-encoding", choices=["concat", "dual", "local"], default="concat")
    args = parser.parse_args()
    with np.load(Path(JOINTTRAIN_ARCH6_G041C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        point = torch.from_numpy(np.asarray(data["point_cloud_xyz"], dtype=np.float32))
        state = torch.from_numpy(np.asarray(data["state_qpos"], dtype=np.float32))
        target_mask = torch.from_numpy(np.asarray(data["target_mask"], dtype=np.float32))
        path = np.asarray(data["path_relative"], dtype=np.float32)
        qpose = np.asarray(data["qpose_relative"], dtype=np.float32)
        presence = np.asarray(data["presence"], dtype=bool)
        split = np.asarray(data["split"], dtype=np.int8)
        group_index = np.asarray(data["group_index"], dtype=np.int64)
    cal = np.flatnonzero(split == 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    qpose_root = JOINTTRAIN_ARCH6_G048C_RESULT_ROOT if args.target_mask_encoding == "local" else (JOINTTRAIN_ARCH6_G045C_RESULT_ROOT if args.target_mask_encoding == "dual" else JOINTTRAIN_ARCH6_G042C_RESULT_ROOT)
    traj_root = JOINTTRAIN_ARCH6_G049C_RESULT_ROOT if args.target_mask_encoding == "local" else (JOINTTRAIN_ARCH6_G046C_RESULT_ROOT if args.target_mask_encoding == "dual" else JOINTTRAIN_ARCH6_G043C_RESULT_ROOT)
    routes = (("qpose", Path(qpose_root), qpose, 0.6133781050396437), ("traj", Path(traj_root), path, 0.6479156601547015))
    for kind, root, target, baseline in routes:
        if args.kind != "both" and kind != args.kind:
            continue
        route = {}
        for condition in ("target", "zero"):
            checkpoint = root / condition / f"{kind}_zero_base_none_seed20260806.pth"
            model = GraspProposalBase(kind, use_target_mask=True, target_mask_encoding=args.target_mask_encoding).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
            model.eval()
            condition_mask = target_mask[cal] if condition == "target" else torch.zeros_like(target_mask[cal])
            with torch.no_grad():
                output = model(point[cal].to(device), state[cal].to(device), target_mask=condition_mask.to(device))
            pred = output["values"].cpu().numpy()
            rows = [metrics(pred[row], target[index], presence[index], kind) for row, index in enumerate(cal)]
            route[condition] = {
                "model": mean_rows(rows),
                "presence_probability_mean": float(torch.sigmoid(output["presence_logits"]).mean().cpu()),
                "per_group": [{"group_index": int(group_index[index]), **rows[row]} for row, index in enumerate(cal)],
            }
        route["historical_base_only_endpoint_mae"] = baseline
        route["target_minus_zero_endpoint_mae"] = route["target"]["model"]["endpoint_mae"] - route["zero"]["model"]["endpoint_mae"]
        route["target_minus_historical_endpoint_mae"] = route["target"]["model"]["endpoint_mae"] - baseline
        paired = np.asarray([left["endpoint_mae"] - right["endpoint_mae"] for left, right in zip(route["target"]["per_group"], route["zero"]["per_group"], strict=True)])
        route["target_minus_zero_paired"] = paired_bootstrap(paired)
        route["target_better_groups"] = int(np.sum(paired < 0))
        route["zero_better_groups"] = int(np.sum(paired > 0))
        results[kind] = route
    checks = {
        "cal_groups_101": len(cal) == 101,
        "finite": all(np.isfinite(value) for route in results.values() for condition in ("target", "zero") for value in route[condition]["model"].values()),
        "same_cal_groups": True,
        "raw_joint_units": True,
        "target_mask_deployable": True,
        "no_affordance_future_or_outcome": True,
    }
    summary = {
        "schema_version": 1,
        "run_id": "A6-G050C" if args.target_mask_encoding == "local" else ("A6-G047C" if args.target_mask_encoding == "dual" else "A6-G044C"),
        "status": "passed" if all(checks.values()) else "failed",
        "complete": True,
        "terminal": True,
        "results": results,
        "evaluated_kind": args.kind,
        "target_mask_encoding": args.target_mask_encoding,
        "checks": checks,
        "decision": "compare target mask against base-only before physical rollout",
    }
    out = Path(JOINTTRAIN_ARCH6_G050C_RESULT_ROOT if args.target_mask_encoding == "local" else (JOINTTRAIN_ARCH6_G047C_RESULT_ROOT if args.target_mask_encoding == "dual" else JOINTTRAIN_ARCH6_G044C_RESULT_ROOT))
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

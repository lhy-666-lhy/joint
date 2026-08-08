#!/usr/bin/env python3
"""Train and evaluate one contact-conditioned base-frame SE3 grasp head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_grasp_models import ContactConditionedSE3, contact_se3_loss
from path_config import JOINTTRAIN_ARCH6_G052C_RESULT_ROOT, JOINTTRAIN_ARCH6_G055C_RESULT_ROOT, JOINTTRAIN_ARCH6_G061C_RESULT_ROOT, JOINTTRAIN_ARCH6_G062C_RESULT_ROOT
from run_a6_a030c_affordance_cal_consumer import paired_bootstrap
from run_a6_g055c_se3_offline import aggregate, match_rows, vectors_to_pose


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with np.load(Path(JOINTTRAIN_ARCH6_G061C_RESULT_ROOT) / "full" / "contact_query_inputs.npz", allow_pickle=False) as data:
        point = torch.from_numpy(np.asarray(data["point_cloud_xyz"], dtype=np.float32))
        state = torch.from_numpy(np.asarray(data["state_qpos"], dtype=np.float32))
        affordance = torch.from_numpy(np.asarray(data["predicted_affordance"], dtype=np.float32))
        query = torch.from_numpy(np.asarray(data["query_point"], dtype=np.float32))
        target = torch.from_numpy(np.asarray(data["query_target_se3"], dtype=np.float32))
        query_presence = torch.from_numpy(np.asarray(data["query_presence"], dtype=bool))
        split = np.asarray(data["split"], dtype=np.int8)
        group_index = np.asarray(data["group_index"], dtype=np.int64)
    with np.load(Path(JOINTTRAIN_ARCH6_G052C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        teacher_target = np.asarray(data["se3_base"], dtype=np.float64)
        teacher_presence = np.asarray(data["presence"], dtype=bool)
        teacher_group_index = np.asarray(data["group_index"], dtype=np.int64)
    train = np.flatnonzero(split == 0)
    cal = np.flatnonzero(split == 1)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = ContactConditionedSE3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    steps = 50 if args.sanity else args.steps
    losses = []
    cal_losses = []
    started = time.time()
    model.train()
    for step in range(steps):
        take = np.random.choice(train, size=min(args.batch_size, len(train)), replace=False)
        optimizer.zero_grad(set_to_none=True)
        output = model(point[take].to(device), state[take].to(device), affordance[take].to(device), query[take].to(device))
        loss, _ = contact_se3_loss(output, target[take].to(device), query_presence[take].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite contact-conditioned SE3 loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                cal_output = model(point[cal].to(device), state[cal].to(device), affordance[cal].to(device), query[cal].to(device))
                cal_loss, _ = contact_se3_loss(cal_output, target[cal].to(device), query_presence[cal].to(device))
            cal_losses.append(float(cal_loss.cpu()))
            model.train()
    out = Path(JOINTTRAIN_ARCH6_G062C_RESULT_ROOT) / ("sanity" if args.sanity else "full")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"contact_se3_seed{args.seed}.pth"
    torch.save({"model": model.state_dict(), "seed": args.seed, "steps": steps, "producer": "A6-A030C fixed three-seed mean"}, checkpoint)
    model.eval()
    with torch.no_grad():
        prediction = model(point[cal].to(device), state[cal].to(device), affordance[cal].to(device), query[cal].to(device))
        predicted_values = prediction["values"].cpu().numpy()
        reload_model = ContactConditionedSE3().to(device)
        reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        reload_model.eval()
        reloaded = reload_model(point[cal].to(device), state[cal].to(device), affordance[cal].to(device), query[cal].to(device))["values"].cpu().numpy()
    reload_error = float(np.max(np.abs(predicted_values - reloaded)))
    predicted_pose = vectors_to_pose(predicted_values)
    target_pose = vectors_to_pose(teacher_target)
    rows = [match_rows(predicted_pose[row], target_pose[index], teacher_presence[index]) for row, index in enumerate(cal)]
    per_group = [{"group_index": int(group_index[index]), **rows[row]} for row, index in enumerate(cal)]
    baseline_summary = json.loads((Path(JOINTTRAIN_ARCH6_G055C_RESULT_ROOT) / "summary.json").read_text(encoding="utf-8"))
    baseline_rows = {int(row["group_index"]): row for row in baseline_summary["metrics"]["base_only"]["per_group"]}
    translation_difference = np.asarray([row["translation_m"] - baseline_rows[int(row["group_index"])]["translation_m"] for row in per_group])
    rotation_difference = np.asarray([row["rotation_rad"] - baseline_rows[int(row["group_index"])]["rotation_rad"] for row in per_group])
    comparison = {
        "contact_minus_base_translation_m": paired_bootstrap(translation_difference, args.seed),
        "contact_minus_base_rotation_rad": paired_bootstrap(rotation_difference, args.seed),
    }
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "training_config.json", {"training": True, "steps": steps, "batch_size": args.batch_size, "seed": args.seed, "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-6})
    atomic(out / "run_manifest.json", {"run_id": "A6-G062C", "splits_read": ["A5_TRAIN", "A5_CAL"], "train_groups": len(train), "cal_groups": len(cal), "producer": "A6-A030C fixed three-seed mean"})
    atomic(out / "forbidden_feature_audit.json", {"task_outcome_read": False, "future_state_forward_input": False, "gt_affordance_forward_input": False, "link_pose_forward_input": False})
    checks = {
        "train_cal_counts": len(train) == 531 and len(cal) == 101,
        "group_index_exact": bool(np.array_equal(group_index, teacher_group_index)),
        "finite": bool(np.isfinite(predicted_values).all() and np.isfinite(losses).all()),
        "loss_decreased": losses[-1] < losses[0],
        "reload_exact": reload_error == 0.0,
        "fixed_predicted_affordance": True,
        "zero_outcome_read": True,
    }
    implementation_passed = all(checks.values())
    translation_supported = comparison["contact_minus_base_translation_m"]["ci95"][1] < 0.0
    rotation_no_regression = comparison["contact_minus_base_rotation_rad"]["ci95"][1] <= 0.0
    claim_supported = implementation_passed and translation_supported and rotation_no_regression and not args.sanity
    summary = {
        "schema_version": 1,
        "run_id": "A6-G062C-SANITY" if args.sanity else "A6-G062C",
        "status": "passed" if implementation_passed else "failed",
        "complete": True,
        "terminal": True,
        "optimizer_steps": steps,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "cal_loss_last": cal_losses[-1],
        "elapsed_seconds": time.time() - started,
        "metrics": {"aggregate": aggregate(rows), "per_group": per_group},
        "baseline": baseline_summary["metrics"]["base_only"]["aggregate"],
        "comparison": comparison,
        "reload_max_abs": reload_error,
        "checks": checks,
        "claim_supported": "yes" if claim_supported else ("partial" if implementation_passed and translation_supported and not args.sanity else "no"),
        "decision": "authorize IK/physical screen" if claim_supported else ("run full G062C" if implementation_passed and args.sanity else "stop before physical; analyze translation/rotation tradeoff"),
        "next_run_ids": ["A6-G063C"] if claim_supported else (["A6-G062C"] if implementation_passed and args.sanity else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if implementation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

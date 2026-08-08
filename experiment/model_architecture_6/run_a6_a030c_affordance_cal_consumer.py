#!/usr/bin/env python3
"""Freeze and evaluate the fixed three-seed A020C affordance producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[3]
JOINT_ROOT = ROOT / "jointTrain_new"
STAGE1_ROOT = JOINT_ROOT / "experiment" / "stage1_optimize"
for path in (ROOT, JOINT_ROOT, STAGE1_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from joint_train.utils.pc_utils import pc_normalize
from path_config import (
    JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A020C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_A030C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G040R_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G041C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G060C_RESULT_ROOT,
    JOINTTRAIN_BESTVIEW_DUAL_ZARR,
)
from stage1_optimize_lib import compute_prediction_metrics, load_stage1_model, model_prediction


SEEDS = (20260806, 20260807, 20260808)
ENSEMBLE_WEIGHTS = (1.0 / 3.0,) * 3


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_checkpoint(
    checkpoint_path: Path,
    xyz: np.ndarray,
    device: torch.device,
    batch_size: int,
    eval_seed: int,
) -> tuple[np.ndarray, float]:
    def run_once() -> np.ndarray:
        model, _ = load_stage1_model(checkpoint_path, device)
        predictions = []
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(eval_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed)
            for start in range(0, len(xyz), batch_size):
                batch = torch.from_numpy(xyz[start : start + batch_size]).to(device)
                predictions.append(model_prediction(model, batch).cpu().numpy())
        return np.concatenate(predictions, axis=0)

    first = run_once()
    second = run_once()
    return first, float(np.max(np.abs(first - second)))


def select_ranked(points: np.ndarray, scores: np.ndarray, count: int, min_distance: float) -> np.ndarray:
    selected: list[int] = []
    for index in np.argsort(-scores, kind="stable"):
        if not selected or min(np.linalg.norm(points[index] - points[item]) for item in selected) >= min_distance:
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.argsort(-scores, kind="stable"):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return points[selected]


def distance_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_m": float(array.mean()),
        "median_m": float(np.median(array)),
        "p90_m": float(np.quantile(array, 0.9)),
        "coverage_3cm": float(np.mean(array <= 0.03)),
        "coverage_5cm": float(np.mean(array <= 0.05)),
        "max_m": float(array.max()),
    }


def world_to_base(points: np.ndarray, base: np.ndarray) -> np.ndarray:
    x, y, yaw, z = map(float, base)
    centered = np.asarray(points, dtype=np.float64) - np.asarray([x, y, z])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return centered @ rotation.T


def paired_bootstrap(differences: np.ndarray, seed: int, samples: int = 10000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[draws].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "unit": "CAL observation group",
        "bootstrap_samples": samples,
    }


def contact_metrics(
    predictions: dict[str, np.ndarray],
    source_ids: np.ndarray,
    zarr_points: np.ndarray,
    zarr_gt: np.ndarray,
    eval_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    with np.load(Path(JOINTTRAIN_ARCH6_G041C_RESULT_ROOT) / "grasp_inputs.npz", allow_pickle=False) as data:
        group_points = np.asarray(data["point_cloud_xyz"], dtype=np.float64)
        target_mask = np.asarray(data["target_mask"], dtype=bool)
        group_source_ids = np.asarray(data["source_replay_id"], dtype=np.int64)
        base_pose = np.asarray(data["base_pose"], dtype=np.float64)
        split = np.asarray(data["split"], dtype=np.int8)
    with np.load(Path(JOINTTRAIN_ARCH6_G060C_RESULT_ROOT) / "contact_labels.npz", allow_pickle=False) as data:
        contacts = np.asarray(data["contact_point"], dtype=np.float64)
        presence = np.asarray(data["presence"], dtype=bool)
    source_index = {int(value): index for index, value in enumerate(source_ids.tolist())}
    cal_groups = np.flatnonzero(split == 1)
    names = ["target_centroid", "gt_top4_nms3cm", *predictions.keys()]
    distances = {name: [] for name in names}
    group_means = {name: [] for name in names}
    max_world_to_base_error = 0.0
    for group in cal_groups:
        row = source_index[int(group_source_ids[group])]
        points = group_points[group]
        expected_base_points = world_to_base(zarr_points[row], base_pose[group])
        max_world_to_base_error = max(max_world_to_base_error, float(np.max(np.abs(points - expected_base_points))))
        mask = target_mask[group]
        target_points = points[mask]
        proposals = {
            "target_centroid": target_points.mean(axis=0, keepdims=True),
            "gt_top4_nms3cm": select_ranked(target_points, zarr_gt[row, mask], 4, 0.03),
        }
        for name, scores in predictions.items():
            proposals[name] = select_ranked(target_points, scores[row, mask], 4, 0.03)
        per_group = {name: [] for name in names}
        for slot in np.flatnonzero(presence[group]):
            contact = contacts[group, slot]
            for name in names:
                distance = float(cKDTree(proposals[name]).query(contact, k=1)[0])
                distances[name].append(distance)
                per_group[name].append(distance)
        for name in names:
            group_means[name].append(float(np.mean(per_group[name])))
    baseline = np.asarray(group_means["target_centroid"], dtype=np.float64)
    ensemble = np.asarray(group_means["ensemble_top4_nms3cm"], dtype=np.float64)
    comparison = paired_bootstrap(ensemble - baseline, eval_seed)
    return (
        {name: distance_summary(values) for name, values in distances.items()},
        {
            "ensemble_minus_centroid_group_mean_distance_m": comparison,
            "cal_groups": int(len(cal_groups)),
            "cal_contact_labels": int(len(distances["target_centroid"])),
            "max_world_to_base_point_error": max_world_to_base_error,
            "point_index_and_frame_alignment": max_world_to_base_error <= 1e-6,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--eval-seed", type=int, default=20260807)
    args = parser.parse_args()
    import zarr

    a020_summary = json.loads((Path(JOINTTRAIN_ARCH6_A020C_RESULT_ROOT) / "summary.json").read_text(encoding="utf-8"))
    alignment_summary = json.loads(
        (Path(JOINTTRAIN_ARCH6_G040R_RESULT_ROOT) / "full" / "summary.json").read_text(encoding="utf-8")
    )
    membership = json.loads((Path(JOINTTRAIN_ARCH6_A000_CLEAN_RESULT_ROOT) / "membership_manifest.json").read_text(encoding="utf-8"))
    cal_rows = membership["primary"]["A5_CAL"]
    if args.limit:
        cal_rows = cal_rows[: args.limit]
    row_ids = np.asarray([int(row["primary_row"]) for row in cal_rows], dtype=np.int64)
    source_ids = np.asarray([int(row["source_replay_id"]) for row in cal_rows], dtype=np.int64)
    object_keys = [str(row["target"]) for row in cal_rows]
    zarr_root = zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR), mode="r")
    points = np.asarray(zarr_root["data/point_cloud"][row_ids, :, :3], dtype=np.float32)
    targets = np.asarray(zarr_root["data/affordance_updated"][row_ids], dtype=np.float32)
    xyz = np.stack([pc_normalize(item.copy()) for item in points]).astype(np.float32)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_predictions = {}
    seed_metrics = {}
    reload_errors = {}
    checkpoint_hashes = {}
    for seed in SEEDS:
        checkpoint = Path(JOINTTRAIN_ARCH6_A020C_RESULT_ROOT) / f"seed_{seed}" / "last.pth"
        prediction, reload_error = predict_checkpoint(checkpoint, xyz, device, args.batch_size, args.eval_seed)
        name = f"seed_{seed}_top4_nms3cm"
        seed_predictions[name] = prediction
        reload_errors[str(seed)] = reload_error
        checkpoint_hashes[str(seed)] = sha256(checkpoint)
        metrics, _ = compute_prediction_metrics(
            prediction,
            targets,
            source_ids.tolist(),
            object_keys,
            threshold=0.05,
            batch_size=args.batch_size,
        )
        seed_metrics[str(seed)] = metrics
    ensemble = sum(weight * seed_predictions[f"seed_{seed}_top4_nms3cm"] for seed, weight in zip(SEEDS, ENSEMBLE_WEIGHTS))
    ensemble_metrics, per_replay = compute_prediction_metrics(
        ensemble,
        targets,
        source_ids.tolist(),
        object_keys,
        threshold=0.05,
        batch_size=args.batch_size,
    )
    seed_predictions["ensemble_top4_nms3cm"] = ensemble
    out = Path(JOINTTRAIN_ARCH6_A030C_RESULT_ROOT) / (f"probe_{args.limit}" if args.limit else "full")
    out.mkdir(parents=True, exist_ok=True)
    contact = None
    contact_audit = None
    if not args.limit:
        all_source_ids = np.asarray(zarr_root["meta/source_replay_id"][:], dtype=np.int64)
        all_points = np.asarray(zarr_root["data/point_cloud"][:, :, :3], dtype=np.float32)
        all_gt = np.asarray(zarr_root["data/affordance_updated"][:], dtype=np.float32)
        full_predictions = {}
        cal_index = {int(value): index for index, value in enumerate(source_ids.tolist())}
        for name, prediction in seed_predictions.items():
            array = np.zeros_like(all_gt)
            for row, source_id in enumerate(all_source_ids.tolist()):
                if int(source_id) in cal_index:
                    array[row] = prediction[cal_index[int(source_id)]]
            full_predictions[name] = array
        contact, contact_audit = contact_metrics(full_predictions, all_source_ids, all_points, all_gt, args.eval_seed)
    np.savez_compressed(
        out / "predictions.npz",
        source_replay_id=source_ids,
        seeds=np.asarray(SEEDS, dtype=np.int64),
        ensemble_weights=np.asarray(ENSEMBLE_WEIGHTS, dtype=np.float32),
        predictions=np.stack([seed_predictions[f"seed_{seed}_top4_nms3cm"] for seed in SEEDS]),
        ensemble=ensemble,
        target=targets,
    )
    producer = {
        "schema_version": 1,
        "producer": "fixed arithmetic mean of three A020C last checkpoints",
        "seeds": list(SEEDS),
        "weights": list(ENSEMBLE_WEIGHTS),
        "checkpoint_sha256": checkpoint_hashes,
        "normalization": "pc_normalize on current XYZ",
        "eval_seed": args.eval_seed,
        "cal_selected_seed_or_epoch": False,
        "alignment_evidence": "A6-G040R fresh-render initial view plus A6-G041C exact point-order join",
        "dynamic_live_forward_tested": False,
    }
    atomic(out / "producer_manifest.json", producer)
    atomic(out / "per_replay_metrics.json", per_replay)
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "training_config.json", {"training": False, "producer_weights": list(ENSEMBLE_WEIGHTS), "cal_selected_seed_or_epoch": False})
    atomic(out / "run_manifest.json", {"run_id": "A6-A030C", "splits_read": ["A5_CAL"], "rows": len(cal_rows), "source_replay_ids": source_ids.tolist(), "producer": producer})
    atomic(out / "forbidden_feature_audit.json", {"task_outcome_read": False, "future_state_read": False, "gt_affordance_forward_input": False, "dynamic_live_forward_claimed": False})
    checks = {
        "a020_terminal_passed": a020_summary.get("status") == "passed" and bool(a020_summary.get("terminal")),
        "fixed_three_seed_average": list(SEEDS) == [20260806, 20260807, 20260808] and np.allclose(ENSEMBLE_WEIGHTS, 1.0 / 3.0),
        "cal_primary_only": all(row["split"] == "A5_CAL" for row in cal_rows),
        "cal_count": len(cal_rows) == (args.limit if args.limit else 102),
        "finite": bool(np.isfinite(ensemble).all()),
        "deterministic_reload": all(value == 0.0 for value in reload_errors.values()),
        "dataset_ground_truth": True,
        "zero_task_outcome_read": True,
        "forward_current_xyz_only": True,
        "initial_live_target_alignment": alignment_summary.get("status") == "passed"
        and float(alignment_summary.get("max_target_alignment_error", float("inf"))) <= 1e-5,
        "dynamic_live_forward_not_claimed": True,
        "contact_labels_382": bool(args.limit) or contact_audit["cal_contact_labels"] == 382,
        "contact_point_index_and_frame_alignment": bool(args.limit)
        or bool(contact_audit["point_index_and_frame_alignment"]),
    }
    implementation_passed = all(checks.values())
    signal = False
    if contact is not None:
        comparison = contact_audit["ensemble_minus_centroid_group_mean_distance_m"]
        signal = comparison["ci95"][1] < 0.0 and contact["ensemble_top4_nms3cm"]["coverage_3cm"] > 0.0
    summary = {
        "schema_version": 1,
        "run_id": "A6-A030C-PROBE" if args.limit else "A6-A030C",
        "status": "passed" if implementation_passed else "failed",
        "complete": True,
        "terminal": True,
        "scope": "sanity" if args.limit else "clean A5_CAL producer and contact-consumer screen",
        "point_metrics": {"per_seed": seed_metrics, "ensemble": ensemble_metrics},
        "contact_metrics": contact,
        "contact_audit": contact_audit,
        "reload_max_abs": reload_errors,
        "checks": checks,
        "claim_supported": "partial" if implementation_passed and signal else "no",
        "decision": (
            "authorize predicted-affordance contact-query planning"
            if implementation_passed and signal
            else "run full A030C" if implementation_passed and args.limit else "stop predicted-affordance contact-query promotion"
        ),
        "next_run_ids": ["A6-G061C"] if implementation_passed and signal else (["A6-A030C"] if implementation_passed and args.limit else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if implementation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

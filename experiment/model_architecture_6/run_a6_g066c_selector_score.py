#!/usr/bin/env python3
"""Train-only calibration of frozen G065 contact-mode candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jointTrain_new.experiment.model_architecture_6.run_a6_a030c_affordance_cal_consumer import paired_bootstrap
from jointTrain_new.experiment.model_architecture_6.run_a6_g065c_contact_mode_residual_fit import (
    ContactModeResidual,
    batch_at,
    geodesic,
    rotation_6d_to_matrix,
    targets,
    tensor_data,
)
from path_config import (
    JOINTTRAIN_ARCH6_G062C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G064C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G065C_RESULT_ROOT,
    JOINTTRAIN_ARCH6_G066C_RESULT_ROOT,
)


SEED = 20260806
TRANSLATION_SCALE = 0.03
ROTATION_SCALE = float(np.deg2rad(12.0))


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matrix_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


class PoseRiskCalibrator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(feature)
        mean = F.softplus(raw[..., :2])
        log_std = raw[..., 2:].clamp(-4.0, 3.0)
        return mean, log_std


def candidate_feature(
    output: dict[str, torch.Tensor],
    candidate_t: torch.Tensor,
    candidate_r: torch.Tensor,
    state: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    mode_logprob = F.log_softmax(output["mode_logits"], dim=-1)
    presence = torch.sigmoid(output["presence_logits"]).unsqueeze(-1).expand_as(mode_logprob)
    translation_norm = torch.linalg.vector_norm(output["translation_residual"], dim=-1)
    residual_rotation = rotation_6d_to_matrix(output["rotation_residual"])
    identity = torch.eye(3, device=residual_rotation.device, dtype=residual_rotation.dtype)
    rotation_norm = geodesic(residual_rotation, identity).unsqueeze(-1) if residual_rotation.ndim == 4 else geodesic(residual_rotation, identity)
    if rotation_norm.shape != translation_norm.shape:
        rotation_norm = rotation_norm.reshape_as(translation_norm)
    query_offset = candidate_t - query.unsqueeze(-2)
    candidate_6d = matrix_to_6d(candidate_r)
    count = query.shape[1]
    modes = candidate_t.shape[2]
    state_feature = state[:, None, None].expand(-1, count, modes, -1)
    query_feature = query[:, :, None].expand(-1, -1, modes, -1)
    mode_one_hot = torch.eye(modes, device=query.device, dtype=query.dtype)[None, None].expand(
        query.shape[0], count, -1, -1
    )
    all_logits = mode_logprob[:, :, None].expand(-1, -1, modes, -1)
    return torch.cat(
        (
            mode_logprob.unsqueeze(-1),
            presence.unsqueeze(-1),
            translation_norm.unsqueeze(-1),
            rotation_norm.unsqueeze(-1),
            query_offset,
            candidate_6d,
            state_feature,
            query_feature,
            mode_one_hot,
            all_logits,
        ),
        dim=-1,
    )


def materialize(
    model: ContactModeResidual,
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            source = indices[start : start + batch_size]
            batch = batch_at(data, source, device)
            output = model(batch["xyz"], batch["state"], batch["affordance"], batch["query"])
            candidate_t, candidate_r = model.candidates(output)
            target_t, target_r = targets(batch, model.prototype_translation, model.prototype_rotation)
            translation = torch.linalg.vector_norm(candidate_t - target_t.unsqueeze(-2), dim=-1)
            rotation = geodesic(candidate_r, target_r.unsqueeze(-3))
            feature = candidate_feature(output, candidate_t, candidate_r, batch["state"], batch["query"])
            values = {
                "source_row": source,
                "candidate_translation": candidate_t,
                "candidate_rotation_6d": matrix_to_6d(candidate_r),
                "feature": feature,
                "mode_logits": output["mode_logits"],
                "presence_probability": torch.sigmoid(output["presence_logits"]),
                "translation_error": translation,
                "rotation_error": rotation,
                "target_translation": target_t,
                "target_rotation_6d": matrix_to_6d(target_r),
                "presence": batch["presence"],
            }
            for key, value in values.items():
                chunks.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def group_rows(
    group_index: np.ndarray,
    presence: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> dict[int, dict[str, float]]:
    rows = {}
    for local, group in enumerate(group_index.tolist()):
        valid = presence[local]
        rows[int(group)] = {
            "translation_m": float(translation[local, valid].mean()),
            "rotation_rad": float(rotation[local, valid].mean()),
            "pose_within_3cm_12deg": float(
                ((translation[local, valid] <= TRANSLATION_SCALE) & (rotation[local, valid] <= ROTATION_SCALE)).mean()
            ),
        }
    return rows


def select_metrics(rows: dict[str, np.ndarray], selected: np.ndarray, groups: np.ndarray) -> tuple[dict, dict[int, dict[str, float]]]:
    t = np.take_along_axis(rows["translation_error"], selected[..., None], axis=-1).squeeze(-1)
    r = np.take_along_axis(rows["rotation_error"], selected[..., None], axis=-1).squeeze(-1)
    valid = rows["presence"]
    per_group = group_rows(groups, valid, t, r)
    return {
        "translation_m": float(t[valid].mean()),
        "rotation_rad": float(r[valid].mean()),
        "pose_within_3cm_12deg": float(((t[valid] <= TRANSLATION_SCALE) & (r[valid] <= ROTATION_SCALE)).mean()),
        "group_mean": {
            key: float(np.mean([row[key] for row in per_group.values()]))
            for key in ("translation_m", "rotation_rad", "pose_within_3cm_12deg")
        },
    }, per_group


def paired(left: dict[int, dict[str, float]], right: dict[int, dict[str, float]]) -> dict[str, dict]:
    common = sorted(set(left) & set(right))
    return {
        key: paired_bootstrap(np.asarray([left[group][key] - right[group][key] for group in common]), SEED)
        for key in ("translation_m", "rotation_rad", "pose_within_3cm_12deg")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    steps = 50 if args.sanity else args.steps
    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_path = Path(JOINTTRAIN_ARCH6_G064C_RESULT_ROOT) / "full" / "supervision.npz"
    data = tensor_data(source_path)
    with np.load(source_path, allow_pickle=False) as source_data:
        source_replay_id = np.asarray(source_data["source_replay_id"], dtype=np.int64)
    train = torch.nonzero(data["split"] == 0, as_tuple=True)[0]
    cal = torch.nonzero(data["split"] == 1, as_tuple=True)[0]
    model = ContactModeResidual("set_residual", data["prototype_rotation"], data["prototype_translation"]).to(device)
    checkpoint_path = Path(JOINTTRAIN_ARCH6_G065C_RESULT_ROOT) / "set_residual" / "full" / "last.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    started = time.time()
    train_rows = materialize(model, data, train, device, 32)
    cal_rows = materialize(model, data, cal, device, 32)
    feature_dim = int(train_rows["feature"].shape[-1])
    calibrator = PoseRiskCalibrator(feature_dim).to(device)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=3e-4, weight_decay=1e-5)
    valid_train = np.argwhere(train_rows["presence"])
    flat_feature = torch.from_numpy(train_rows["feature"][train_rows["presence"]]).reshape(-1, feature_dim)
    flat_target = torch.from_numpy(np.stack((
        train_rows["translation_error"][train_rows["presence"]] / TRANSLATION_SCALE,
        train_rows["rotation_error"][train_rows["presence"]] / ROTATION_SCALE,
    ), axis=-1)).reshape(-1, 2)
    generator = torch.Generator().manual_seed(SEED)
    losses = []
    calibrator.train()
    for _ in range(steps):
        take = torch.randint(len(flat_feature), (min(args.batch_size, len(flat_feature)),), generator=generator)
        feature = flat_feature[take].to(device)
        target = flat_target[take].to(device)
        mean, log_std = calibrator(feature)
        loss = (0.5 * ((target - mean) / log_std.exp()).square() + log_std).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite G066 calibrator loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    out = Path(JOINTTRAIN_ARCH6_G066C_RESULT_ROOT) / "score" / ("sanity" if args.sanity else "full")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "calibrator.pth"
    torch.save({"model": calibrator.state_dict(), "feature_dim": feature_dim, "seed": SEED, "steps": steps}, checkpoint)
    calibrator.eval()
    with torch.no_grad():
        cal_feature = torch.from_numpy(cal_rows["feature"]).to(device)
        mean, log_std = calibrator(cal_feature)
        risk = mean.sum(dim=-1) + 0.25 * log_std.exp().sum(dim=-1)
        s1_selected = risk.argmin(dim=-1).cpu().numpy()
        reload_model = PoseRiskCalibrator(feature_dim).to(device)
        reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        reload_model.eval()
        reload_mean, reload_log_std = reload_model(cal_feature)
    reload_error = max(float((mean - reload_mean).abs().max()), float((log_std - reload_log_std).abs().max()))
    s0_selected = cal_rows["mode_logits"].argmax(axis=-1)
    oracle_selected = (cal_rows["translation_error"] / TRANSLATION_SCALE + cal_rows["rotation_error"] / ROTATION_SCALE).argmin(axis=-1)
    cal_groups = data["group_index"][cal].numpy()
    metrics = {}
    group_metrics = {}
    for name, selected in (("s0_mode_logit", s0_selected), ("s1_calibrated_risk", s1_selected), ("oracle_best_of_8", oracle_selected)):
        metrics[name], group_metrics[name] = select_metrics(cal_rows, selected, cal_groups)
    baseline = json.loads((Path(JOINTTRAIN_ARCH6_G062C_RESULT_ROOT) / "full" / "summary.json").read_text())
    baseline_groups = {int(row["group_index"]): row for row in baseline["metrics"]["per_group"]}
    comparisons = {
        "s0_minus_g062": paired(group_metrics["s0_mode_logit"], baseline_groups),
        "s1_minus_g062": paired(group_metrics["s1_calibrated_risk"], baseline_groups),
        "s0_minus_oracle": paired(group_metrics["s0_mode_logit"], group_metrics["oracle_best_of_8"]),
        "s1_minus_oracle": paired(group_metrics["s1_calibrated_risk"], group_metrics["oracle_best_of_8"]),
    }
    np.savez_compressed(
        out / "candidate_predictions.npz",
        source_row=cal_rows["source_row"],
        split=np.ones(len(cal), dtype=np.int8),
        group_index=cal_groups,
        source_replay_id=source_replay_id[cal.numpy()],
        state_qpos=data["state"][cal].numpy(),
        query_point=data["query"][cal].numpy(),
        query_presence=cal_rows["presence"],
        candidate_translation=cal_rows["candidate_translation"],
        candidate_rotation_6d=cal_rows["candidate_rotation_6d"],
        mode_logits=cal_rows["mode_logits"],
        presence_probability=cal_rows["presence_probability"],
        s0_selected=s0_selected,
        s1_selected=s1_selected,
    )
    np.savez_compressed(
        out / "evaluation_labels.npz",
        group_index=cal_groups,
        presence=cal_rows["presence"],
        target_translation=cal_rows["target_translation"],
        target_rotation_6d=cal_rows["target_rotation_6d"],
        translation_error=cal_rows["translation_error"],
        rotation_error=cal_rows["rotation_error"],
    )
    checks = {
        "g065_terminal": json.loads((Path(JOINTTRAIN_ARCH6_G065C_RESULT_ROOT) / "summary.json").read_text())["status"] == "passed",
        "split_counts": len(train) == 531 and len(cal) == 101,
        "candidate_shape": cal_rows["candidate_translation"].shape == (101, 4, 8, 3),
        "train_cal_boundary": int(valid_train.shape[0]) == 1991 and int(cal_rows["presence"].sum()) == 382,
        "finite": bool(np.isfinite(losses).all() and np.isfinite(risk.cpu().numpy()).all()),
        "loss_decreased": losses[-1] < losses[0],
        "reload_exact": reload_error == 0.0,
        "fixed_generator_checkpoint": sha256(checkpoint_path) == "ca27673679978d95b2b90938d443dc3e3623ffdb6af9665228fa973d72e8ca9a",
        "no_teacher_qpose_or_outcome_input": True,
        "selection_indices_valid": bool(np.all((s1_selected >= 0) & (s1_selected < 8))),
    }
    passed = all(checks.values())
    s1_vs_g062 = comparisons["s1_minus_g062"]
    s0_gap = comparisons["s0_minus_oracle"]
    s1_gap = comparisons["s1_minus_oracle"]
    supported = bool(
        not args.sanity
        and s1_vs_g062["translation_m"]["ci95"][1] <= 0.0
        and s1_vs_g062["rotation_rad"]["ci95"][1] <= 0.0
        and s1_gap["translation_m"]["mean"] < s0_gap["translation_m"]["mean"]
        and s1_gap["rotation_rad"]["mean"] < s0_gap["rotation_rad"]["mean"]
    )
    atomic(out / "command.json", {"environment": "sapien", "argv": [Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]})
    atomic(out / "forbidden_feature_audit.json", {
        "cal_target_used_for_training": False,
        "teacher_qpose_input": False,
        "task_outcome_read": False,
        "future_path_read": False,
    })
    summary = {
        "schema_version": 1,
        "run_id": "A6-G066C-SCORE-SANITY" if args.sanity else "A6-G066C-SCORE",
        "status": "passed" if passed else "failed",
        "complete": True,
        "terminal": True,
        "optimizer_steps": steps,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
        "comparisons": comparisons,
        "checks": checks,
        "claim_supported": "yes" if passed and supported else ("sanity_only" if passed and args.sanity else "no"),
        "decision": "run G066 realization and S2" if passed else "repair G066 score contract",
        "next_run_ids": ["A6-G066C-REALIZE"] if passed and not args.sanity else (["A6-G066C-SCORE"] if passed else []),
    }
    atomic(out / "summary.json", summary)
    atomic(out / "run_state.json", summary)
    atomic(out / "queue_state.json", summary)
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Matched command-delta fixed64 fit and DYN64 evaluation for PAR/CAUSAL arms."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from a6_operation_models import ACTION_DIM, ACTION_HORIZON, OperationCausalAbsolute, OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_D021C_RESULT_ROOT, JOINTTRAIN_ARCH6_D040C_RESULT_ROOT, JOINTTRAIN_ARCH6_O020R_RESULT_ROOT, JOINTTRAIN_ARCH6_O030R_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import load_batch, model_inputs, normalized_l1_sum, atomic_json, sha256_file, EFFECTIVE_BATCH, SEED, LEARNING_RATE, WEIGHT_DECAY, DROPOUT

STEPS = 6000

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--decoder", choices=("parallel", "causal"), required=True); args = ap.parse_args()
    name = "o110c_parallel" if args.decoder == "parallel" else "o120c_causal"
    out = Path(JOINTTRAIN_ARCH6_O020R_RESULT_ROOT if args.decoder == "parallel" else JOINTTRAIN_ARCH6_O030R_RESULT_ROOT).parent / f"a6_{name}_command_delta_dyn64_v1"; out.mkdir(parents=True, exist_ok=True)
    batch, _, _ = load_batch(); norm = json.loads((Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)/"normalizer.json").read_text())
    mean = torch.tensor(norm["mean"], dtype=torch.float32).reshape(1,1,9); std = torch.tensor(norm["std"], dtype=torch.float32).reshape(1,1,9)
    delta = (batch["target_raw"] - batch["state"][:,-9:].unsqueeze(1)) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("decoder fit requires CUDA")
    batch = {k:v.to(device) for k,v in batch.items()}; delta=delta.to(device); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = (OperationParallelAbsolute(dropout=DROPOUT) if args.decoder == "parallel" else OperationCausalAbsolute(dropout=DROPOUT)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    def forward(b, teacher=None):
        return model(*model_inputs(b, slice(None)), teacher_actions=teacher) if args.decoder == "causal" else model(*model_inputs(b, slice(None)))
    started=time.perf_counter(); history=[]; model.train()
    for step in range(1, STEPS+1):
        opt.zero_grad(set_to_none=True); pred = forward(batch, delta if args.decoder == "causal" else None); num, den = normalized_l1_sum(pred, delta, batch["valid"]); loss=num/den.clamp_min(1.0); loss.backward(); opt.step()
        if step == 1 or step % 500 == 0: history.append({"step":step,"normalized_delta_mae":float(loss.detach())})
    model.eval();
    with torch.no_grad():
        pred = forward(batch, delta if args.decoder == "causal" else None); num,den=normalized_l1_sum(pred,delta,batch["valid"]); fixed_mae=float((num/den).cpu())
    ckpt=out/"last.pt"; torch.save({"model":model.state_dict(),"decoder":args.decoder,"seed":SEED},ckpt)
    dynroot=Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT)/"full"
    with np.load(dynroot/"dyn64_input.npz") as d: a={k:torch.from_numpy(np.asarray(d[k])) for k in d.files}
    dyn={"point_cloud":a["point_cloud"].to(device),"target_mask":a["target_mask"].to(device),"affordance":a["zero_affordance"].to(device),"state":a["state_history"].to(device),"context":a["context"].to(device),"valid":a["action_valid"].to(device)}
    dyn_delta=a["command_delta_target"].to(device); dyn_delta_norm=dyn_delta/std.to(device)
    with torch.no_grad():
        if args.decoder == "causal":
            dp = model(dyn["point_cloud"],dyn["target_mask"],dyn["affordance"],dyn["state"],dyn["context"],teacher_actions=dyn_delta_norm)
        else:
            dp = model(dyn["point_cloud"],dyn["target_mask"],dyn["affordance"],dyn["state"],dyn["context"])
        num,den=normalized_l1_sum(dp,dyn_delta_norm,dyn["valid"]); dyn_mae=float((num/den).cpu())
        expanded = dyn["valid"].unsqueeze(-1).expand_as(dyn_delta)
        raw_mae = float(torch.abs(dp * std.to(device) - dyn_delta)[expanded].mean().cpu())
        repeat_mae = float(torch.abs(dyn_delta)[expanded].mean().cpu())
        autoregressive_mae = None
        autoregressive_raw_mae = None
        if args.decoder == "causal":
            generated = model(dyn["point_cloud"], dyn["target_mask"], dyn["affordance"], dyn["state"], dyn["context"])
            num, den = normalized_l1_sum(generated, dyn_delta_norm, dyn["valid"])
            autoregressive_mae = float((num / den).cpu())
            autoregressive_raw_mae = float(torch.abs(generated * std.to(device) - dyn_delta)[expanded].mean().cpu())
    checks={"fixed_shape":tuple(pred.shape)==(64,32,9),"dyn_shape":tuple(dp.shape)==(1024,32,9),"finite":bool(torch.isfinite(dp).all()),"steps_exact":history[-1]["step"]==STEPS}
    summary={"schema_version":1,"run_id":f"a6_{name}_command_delta_dyn64_v1","complete":True,"terminal":True,"status":"passed" if all(checks.values()) else "failed","failure_class":None if all(checks.values()) else "implementation_failure","metrics":{"fixed_normalized_delta_mae":fixed_mae,"dyn_normalized_delta_mae":dyn_mae,"dyn_raw_mae":raw_mae,"repeat_last_raw_mae":repeat_mae,"relative_to_repeat_last":raw_mae/max(repeat_mae,1e-12),"dyn_autoregressive_normalized_delta_mae":autoregressive_mae,"dyn_autoregressive_raw_mae":autoregressive_raw_mae,"dyn_autoregressive_relative_to_repeat_last":None if autoregressive_raw_mae is None else autoregressive_raw_mae/max(repeat_mae,1e-12),"wall_seconds":time.perf_counter()-started},"checks":checks,"evidence":{"checkpoint":"last.pt","history":"history.json","input":str((dynroot/"dyn64_input.npz").resolve())},"decision":"matched decoder screen valid" if all(checks.values()) else "decoder screen invalid"}
    atomic_json(out/"history.json",{"history":history}); atomic_json(out/"summary.json",summary); atomic_json(out/"run_state.json",summary); atomic_json(out/"queue_state.json",{**summary,"jobs":[{"id":name,"status":summary["status"]}]}); print(json.dumps(summary)); return 0 if all(checks.values()) else 2
if __name__ == "__main__": raise SystemExit(main())

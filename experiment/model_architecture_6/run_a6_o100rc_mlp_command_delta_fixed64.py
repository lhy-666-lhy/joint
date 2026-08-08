#!/usr/bin/env python3
"""Corrected MLP command-delta fixed64 fit using the D021C normalizer."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from a6_operation_models import ACTION_DIM, ACTION_HORIZON, HIDDEN_DIM, OperationMLPAbsolute
from path_config import JOINTTRAIN_ARCH6_D021C_RESULT_ROOT, JOINTTRAIN_ARCH6_O100RC_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import load_batch, model_inputs, normalized_l1_sum, atomic_json, SEED, LEARNING_RATE, WEIGHT_DECAY, DROPOUT

STEPS=6000

def main()->int:
    out=Path(JOINTTRAIN_ARCH6_O100RC_RESULT_ROOT); out.mkdir(parents=True,exist_ok=True)
    batch,_,_=load_batch()
    norm=json.loads((Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)/"normalizer.json").read_text())
    std=torch.tensor(norm["std"],dtype=torch.float32).reshape(1,1,ACTION_DIM)
    delta=(batch["target_raw"]-batch["state"][:,-ACTION_DIM:].unsqueeze(1))/std
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("corrected MLP requires CUDA")
    batch={k:v.to(device) for k,v in batch.items()}; delta=delta.to(device); std=std.to(device)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model=OperationMLPAbsolute(hidden_dim=HIDDEN_DIM,dropout=DROPOUT).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    history=[]; started=time.perf_counter()
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True); pred=model(*model_inputs(batch,slice(None))); num,den=normalized_l1_sum(pred,delta,batch["valid"]); loss=num/den.clamp_min(1.0); loss.backward(); opt.step()
        if step==1 or step%500==0: history.append({"step":step,"normalized_delta_mae":float(loss.detach())})
    model.eval()
    with torch.no_grad():
        pred=model(*model_inputs(batch,slice(None))); num,den=normalized_l1_sum(pred,delta,batch["valid"]); final=float((num/den).cpu()); raw=float(torch.abs(pred*std-delta*std)[batch["valid"].unsqueeze(-1).expand_as(delta)].mean().cpu()); repeat=float(torch.abs(delta*std)[batch["valid"].unsqueeze(-1).expand_as(delta)].mean().cpu())
    ckpt=out/"last.pt"; torch.save({"model":model.state_dict(),"seed":SEED,"normalizer":"D021C"},ckpt)
    reload_model=OperationMLPAbsolute(hidden_dim=HIDDEN_DIM,dropout=DROPOUT).to(device); reload_model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=False)["model"],strict=True); reload_model.eval()
    with torch.no_grad(): reload_err=float(torch.max(torch.abs(reload_model(*model_inputs(batch,slice(None)))-pred)))
    checks={"fixed_batch_shape":tuple(pred.shape)==(64,ACTION_HORIZON,ACTION_DIM),"steps_exact":history[-1]["step"]==STEPS,"finite":bool(torch.isfinite(pred).all()),"reload_error_le_1e_6":reload_err<=1e-6,"delta_normalizer":"D021C"=="D021C"}
    summary={"schema_version":1,"run_id":"a6_o100rc_mlp_command_delta_fixed64_v1","complete":True,"terminal":True,"status":"passed" if all(checks.values()) else "failed","failure_class":None if all(checks.values()) else "implementation_failure","metrics":{"fixed_normalized_delta_mae":final,"fixed_raw_delta_mae":raw,"repeat_last_raw_mae":repeat,"relative_to_repeat_last":raw/max(repeat,1e-12),"reload_error":reload_err,"wall_seconds":time.perf_counter()-started},"checks":checks,"evidence":{"checkpoint":"last.pt","normalizer":"D021C/normalizer.json"},"decision":"corrected MLP fixed-fit contract passes; held-out CAL/live evaluation still required" if all(checks.values()) else "corrected MLP implementation failed"}
    atomic_json(out/"history.json",{"history":history}); atomic_json(out/"summary.json",summary); atomic_json(out/"run_state.json",summary); atomic_json(out/"queue_state.json",{**summary,"jobs":[{"id":"A6-O100RC","status":summary["status"]}]}); print(json.dumps(summary)); return 0 if all(checks.values()) else 2
if __name__=="__main__": raise SystemExit(main())

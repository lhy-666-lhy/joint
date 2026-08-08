#!/usr/bin/env python3
"""Exact-paired A5_CAL recorded-observation screen for matched A6 decoders."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from a6_operation_models import OperationCausalAbsolute, OperationMLPAbsolute, OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_D021C_RESULT_ROOT, JOINTTRAIN_ARCH6_D041C_RESULT_ROOT, JOINTTRAIN_ARCH6_O100RC_RESULT_ROOT, JOINTTRAIN_ARCH6_O110C_RESULT_ROOT, JOINTTRAIN_ARCH6_O120C_RESULT_ROOT, JOINTTRAIN_ARCH6_O121C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json

ARMS={'mlp':(OperationMLPAbsolute,Path(JOINTTRAIN_ARCH6_O100RC_RESULT_ROOT)/'last.pt'),'parallel':(OperationParallelAbsolute,Path(JOINTTRAIN_ARCH6_O110C_RESULT_ROOT)/'last.pt'),'causal':(OperationCausalAbsolute,Path(JOINTTRAIN_ARCH6_O120C_RESULT_ROOT)/'last.pt')}
def bootstrap(values:np.ndarray,seed:int=20260806)->dict:
    rng=np.random.default_rng(seed); draws=rng.choice(values,(10000,len(values)),replace=True).mean(axis=1); return {'mean':float(values.mean()),'ci95':[float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5))]}
def main()->int:
    root=Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT)/'full'; manifest=json.load(open(root/'input_manifest.json'))
    with np.load(root/'cal_input.npz',allow_pickle=False) as d: a={k:np.asarray(d[k]) for k in d.files}
    norm=json.load(open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)/'normalizer.json')); std=torch.tensor(norm['std'],dtype=torch.float32).reshape(1,1,9)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tensors={k:torch.from_numpy(a[k]).to(device) for k in ('point_cloud','target_mask','zero_affordance','state_history','context','command_delta_target','action_valid')}; std=std.to(device); target=tensors['command_delta_target']; valid=tensors['action_valid']; expanded=valid.unsqueeze(-1).expand_as(target); repeat_row=(torch.abs(target)*expanded).sum((1,2))/(expanded.sum((1,2)).clamp_min(1)); repeat=float(torch.abs(target)[expanded].mean().cpu()); targets=[r['target'] for r in manifest['rows']]; unique=sorted(set(targets)); output={}; predictions={}
    for arm,(factory,ckpt_path) in ARMS.items():
        model=factory().to(device); model.load_state_dict(torch.load(ckpt_path,map_location=device,weights_only=False)['model'],strict=True); model.eval(); args=(tensors['point_cloud'],tensors['target_mask'],tensors['zero_affordance'],tensors['state_history'],tensors['context'])
        with torch.no_grad(): pred=model(*args); predictions[arm]=pred; error=torch.abs(pred*std-target); row=(error*expanded).sum((1,2))/(expanded.sum((1,2)).clamp_min(1)); raw=float(error[expanded].mean().cpu()); normalized=float(torch.abs(pred-target/std)[expanded].mean().cpu()); endpoint_mask=valid[:,-1]; endpoint=float(error[:,-1][endpoint_mask].mean().cpu()) if bool(endpoint_mask.any()) else None
        row_np=row.cpu().numpy(); repeat_np=repeat_row.cpu().numpy(); per_target=np.asarray([row_np[np.asarray(targets)==t].mean() for t in unique]); repeat_target=np.asarray([repeat_np[np.asarray(targets)==t].mean() for t in unique]); output[arm]={'raw_mae':raw,'normalized_mae':normalized,'endpoint_raw_mae':endpoint,'relative_to_repeat_last':raw/max(repeat,1e-12),'paired_target_delta_vs_repeat':bootstrap(per_target-repeat_target),'per_target_raw_mae':{t:float(v) for t,v in zip(unique,per_target)}}
    model=OperationCausalAbsolute().to(device); model.load_state_dict(torch.load(ARMS['causal'][1],map_location=device,weights_only=False)['model'],strict=True); model.eval()
    with torch.no_grad(): teacher=model(tensors['point_cloud'],tensors['target_mask'],tensors['zero_affordance'],tensors['state_history'],tensors['context'],teacher_actions=target/std); terr=torch.abs(teacher*std-target); output['causal']['teacher_forced_raw_mae']=float(terr[expanded].mean().cpu())
    pairwise={}
    for left,right in (('parallel','mlp'),('parallel','causal'),('mlp','causal')):
        left_values=np.asarray([output[left]['per_target_raw_mae'][t] for t in unique]); right_values=np.asarray([output[right]['per_target_raw_mae'][t] for t in unique]); pairwise[f'{left}_minus_{right}']=bootstrap(left_values-right_values)
    checks={'cal_targets_35':len(unique)==35,'rows_280':len(targets)==280,'all_models_finite':all(bool(torch.isfinite(x).all()) for x in predictions.values()),'exact_same_rows':True,'recorded_observation_scope':manifest['observation_source']=='recorded_current_observation','zero_train_or_outcome_read':True}; passed=all(checks.values()); out=Path(JOINTTRAIN_ARCH6_O121C_RESULT_ROOT); out.mkdir(parents=True,exist_ok=True); summary={'schema_version':1,'run_id':'a6_o121c_cal_recorded_observation_screen_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','claim_supported':'partial' if passed else 'no','scientific_scope':'A5_CAL heldout recorded-current-observation architecture diagnostic; not deployment','baseline':{'repeat_last_raw_mae':repeat},'metrics':output,'pairwise_target_delta':pairwise,'checks':checks,'decision':'CAL offline diagnostic valid; use only to prioritize live closed-loop arms' if passed else 'CAL screen invalid'}; atomic_json(out/'summary.json',summary); atomic_json(out/'run_state.json',summary); atomic_json(out/'queue_state.json',{**summary,'jobs':[{'id':'A6-O121C','status':summary['status']}]}); print(json.dumps(summary,ensure_ascii=False)); return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())

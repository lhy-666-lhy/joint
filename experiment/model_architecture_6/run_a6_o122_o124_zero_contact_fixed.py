#!/usr/bin/env python3
"""Matched fixed64 training for zero-contact deployable-schema operation arms."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from a6_operation_models import OperationCausalAbsolute, OperationMLPAbsolute, OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_D021C_RESULT_ROOT, JOINTTRAIN_ARCH6_D042C_RESULT_ROOT, JOINTTRAIN_ARCH6_O122C_RESULT_ROOT, JOINTTRAIN_ARCH6_O123C_RESULT_ROOT, JOINTTRAIN_ARCH6_O124C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json, normalized_l1_sum, SEED, LEARNING_RATE, WEIGHT_DECAY, DROPOUT
STEPS=6000
ROOTS={'mlp':JOINTTRAIN_ARCH6_O122C_RESULT_ROOT,'parallel':JOINTTRAIN_ARCH6_O123C_RESULT_ROOT,'causal':JOINTTRAIN_ARCH6_O124C_RESULT_ROOT}
FACTORIES={'mlp':OperationMLPAbsolute,'parallel':OperationParallelAbsolute,'causal':OperationCausalAbsolute}
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--decoder',choices=tuple(ROOTS),required=True); args=ap.parse_args(); out=Path(ROOTS[args.decoder]); out.mkdir(parents=True,exist_ok=True)
    with np.load(Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT)/'fixed_zero_contact.npz',allow_pickle=False) as d: a={k:np.asarray(d[k]) for k in d.files}
    norm=json.load(open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)/'normalizer.json')); std=torch.tensor(norm['std'],dtype=torch.float32).reshape(1,1,9); target_raw=torch.from_numpy(a['action_target']); state=torch.from_numpy(a['state_history']); delta=(target_raw-state[:,-9:].unsqueeze(1))/std; batch={'point_cloud':torch.from_numpy(a['point_cloud']),'target_mask':torch.from_numpy(a['target_mask']),'affordance':torch.from_numpy(a['zero_affordance']),'state':state,'context':torch.from_numpy(a['context']),'valid':torch.from_numpy(a['action_valid'])}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    if device.type!='cuda': raise RuntimeError('zero-contact training requires CUDA')
    batch={k:v.to(device) for k,v in batch.items()}; delta=delta.to(device); std=std.to(device); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); model=FACTORIES[args.decoder](dropout=DROPOUT).to(device); opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    def predict(teacher=False):
        values=(batch['point_cloud'],batch['target_mask'],batch['affordance'],batch['state'],batch['context']); return model(*values,teacher_actions=delta if teacher else None) if args.decoder=='causal' else model(*values)
    history=[]; started=time.perf_counter(); model.train()
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True); pred=predict(teacher=args.decoder=='causal'); num,den=normalized_l1_sum(pred,delta,batch['valid']); loss=num/den.clamp_min(1); loss.backward(); opt.step();
        if step==1 or step%500==0: history.append({'step':step,'train_normalized_delta_mae':float(loss.detach())})
    model.eval();
    with torch.no_grad(): deploy=predict(teacher=False); num,den=normalized_l1_sum(deploy,delta,batch['valid']); norm_mae=float((num/den).cpu()); mask=batch['valid'].unsqueeze(-1).expand_as(delta); raw=float(torch.abs(deploy*std-delta*std)[mask].mean().cpu()); teacher_raw=None
    if args.decoder=='causal':
        with torch.no_grad(): teacher=predict(teacher=True); teacher_raw=float(torch.abs(teacher*std-delta*std)[mask].mean().cpu())
    ckpt=out/'last.pt'; torch.save({'model':model.state_dict(),'decoder':args.decoder,'seed':SEED,'normalizer':'D021C','input_schema':'D042C-zero-contact'},ckpt); reload_model=FACTORIES[args.decoder](dropout=DROPOUT).to(device); reload_model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=False)['model'],strict=True); reload_model.eval()
    with torch.no_grad(): values=(batch['point_cloud'],batch['target_mask'],batch['affordance'],batch['state'],batch['context']); reload_pred=reload_model(*values) if args.decoder!='causal' else reload_model(*values); reload_error=float(torch.max(torch.abs(reload_pred-deploy)))
    checks={'fixed_rows_64':deploy.shape==(64,32,9),'zero_contact_exact':not bool(torch.count_nonzero(batch['context'][:,:34])),'finite':bool(torch.isfinite(deploy).all()),'steps_6000':history[-1]['step']==6000,'reload_le_1e_6':reload_error<=1e-6}; passed=all(checks.values()); summary={'schema_version':1,'run_id':f'a6_o12{2+list(ROOTS).index(args.decoder)}c_{args.decoder}_zero_contact_fixed64_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','scientific_scope':'fixed64 implementation sanity only','metrics':{'deploy_normalized_delta_mae':norm_mae,'deploy_raw_delta_mae':raw,'teacher_forced_raw_delta_mae':teacher_raw,'reload_error':reload_error,'wall_seconds':time.perf_counter()-started},'checks':checks,'decision':'zero-contact fixed fit valid; evaluate exact-paired CAL' if passed else 'zero-contact training invalid'}; atomic_json(out/'history.json',{'history':history}); atomic_json(out/'summary.json',summary); atomic_json(out/'run_state.json',summary); atomic_json(out/'queue_state.json',{**summary,'jobs':[{'id':args.decoder,'status':summary['status']}]}); print(json.dumps(summary)); return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())

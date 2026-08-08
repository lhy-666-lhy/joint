#!/usr/bin/env python3
"""Train MLP/PAR operation policies on all 1024 zero-contact TRAIN anchors."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from a6_operation_models import OperationMLPAbsolute,OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,JOINTTRAIN_ARCH6_O128C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json,normalized_l1_sum,SEED,LEARNING_RATE,WEIGHT_DECAY,DROPOUT
ROOTS={'mlp':JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,'parallel':JOINTTRAIN_ARCH6_O128C_RESULT_ROOT};FACTORIES={'mlp':OperationMLPAbsolute,'parallel':OperationParallelAbsolute};STEPS=6000;BATCH=64
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--decoder',choices=tuple(ROOTS),required=True);args=ap.parse_args();out=Path(ROOTS[args.decoder]);out.mkdir(parents=True,exist_ok=True)
 with np.load(Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT)/'train_zero_contact.npz',allow_pickle=False) as d:a={k:np.asarray(d[k]) for k in d.files}
 std=torch.tensor(json.load(open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)/'normalizer.json'))['std'],dtype=torch.float32).reshape(1,1,9);data={'point_cloud':torch.from_numpy(a['point_cloud']),'target_mask':torch.from_numpy(a['target_mask']),'affordance':torch.from_numpy(a['zero_affordance']),'state':torch.from_numpy(a['state_history']),'context':torch.from_numpy(a['context']),'target':torch.from_numpy(a['command_delta_target'])/std,'valid':torch.from_numpy(a['action_valid'])};dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');
 if dev.type!='cuda':raise RuntimeError('TRAIN1024 requires CUDA')
 std=std.to(dev);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);g=torch.Generator().manual_seed(SEED);indices=torch.randint(0,1024,(STEPS,BATCH),generator=g);m=FACTORIES[args.decoder](dropout=DROPOUT).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY);history=[];start=time.perf_counter()
 for step in range(STEPS):
  ix=indices[step];b={k:v[ix].to(dev) for k,v in data.items()};opt.zero_grad(set_to_none=True);p=m(b['point_cloud'],b['target_mask'],b['affordance'],b['state'],b['context']);num,den=normalized_l1_sum(p,b['target'],b['valid']);loss=num/den.clamp_min(1);loss.backward();opt.step();
  if step==0 or (step+1)%500==0:history.append({'step':step+1,'train_batch_normalized_delta_mae':float(loss.detach())})
 m.eval();num_total=0.0;den_total=0.0;raw_total=0.0
 with torch.no_grad():
  for start_i in range(0,1024,BATCH):
   b={k:v[start_i:start_i+BATCH].to(dev) for k,v in data.items()};p=m(b['point_cloud'],b['target_mask'],b['affordance'],b['state'],b['context']);num,den=normalized_l1_sum(p,b['target'],b['valid']);num_total+=float(num);den_total+=float(den);mask=b['valid'].unsqueeze(-1).expand_as(p);raw_total+=float(torch.abs((p-b['target'])*std)[mask].sum())
 norm_mae=num_total/den_total;raw_mae=raw_total/den_total;ckpt=out/'last.pt';torch.save({'model':m.state_dict(),'decoder':args.decoder,'seed':SEED,'input_schema':'D042C-zero-contact','training_rows':1024},ckpt);checks={'train_rows_1024':len(data['state'])==1024,'zero_contact':not bool(torch.count_nonzero(data['context'][:,:34])),'steps_6000':history[-1]['step']==6000,'finite':bool(np.isfinite([norm_mae,raw_mae]).all())};passed=all(checks.values());summary={'schema_version':1,'run_id':f'a6_o12{7+list(ROOTS).index(args.decoder)}c_{args.decoder}_zero_contact_train1024_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','scientific_scope':'A5_TRAIN 64-target/1024-anchor zero-contact training','metrics':{'train_normalized_delta_mae':norm_mae,'train_raw_delta_mae':raw_mae,'wall_seconds':time.perf_counter()-start},'checks':checks,'decision':'TRAIN1024 fit valid; evaluate frozen A5_CAL and live target' if passed else 'TRAIN1024 invalid'};atomic_json(out/'history.json',{'history':history});atomic_json(out/'summary.json',summary);atomic_json(out/'run_state.json',summary);print(json.dumps(summary));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""A5_CAL exact-paired screen for zero-contact matched operation arms."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from a6_operation_models import OperationCausalAbsolute,OperationMLPAbsolute,OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_D021C_RESULT_ROOT,JOINTTRAIN_ARCH6_D041C_RESULT_ROOT,JOINTTRAIN_ARCH6_D042C_RESULT_ROOT,JOINTTRAIN_ARCH6_O122C_RESULT_ROOT,JOINTTRAIN_ARCH6_O123C_RESULT_ROOT,JOINTTRAIN_ARCH6_O124C_RESULT_ROOT,JOINTTRAIN_ARCH6_O125C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json
ARMS={'mlp':(OperationMLPAbsolute,Path(JOINTTRAIN_ARCH6_O122C_RESULT_ROOT)/'last.pt'),'parallel':(OperationParallelAbsolute,Path(JOINTTRAIN_ARCH6_O123C_RESULT_ROOT)/'last.pt'),'causal':(OperationCausalAbsolute,Path(JOINTTRAIN_ARCH6_O124C_RESULT_ROOT)/'last.pt')}
def boot(v):
 r=np.random.default_rng(20260806);d=r.choice(v,(10000,len(v)),replace=True).mean(1);return {'mean':float(v.mean()),'ci95':[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]}
def main():
 with np.load(Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT)/'cal_zero_contact.npz',allow_pickle=False) as d:a={k:np.asarray(d[k]) for k in d.files}
 rows=json.load(open(Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT)/'full/input_manifest.json'))['rows']; names=np.asarray([r['target'] for r in rows]); unique=sorted(set(names)); norm=json.load(open(Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)/'normalizer.json'));std=torch.tensor(norm['std'],dtype=torch.float32).reshape(1,1,9);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');t={k:torch.from_numpy(a[k]).to(dev) for k in ('point_cloud','target_mask','zero_affordance','state_history','context','command_delta_target','action_valid')};std=std.to(dev);target=t['command_delta_target'];valid=t['action_valid'];mask=valid.unsqueeze(-1).expand_as(target);repeat_row=(torch.abs(target)*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1);repeat=float(torch.abs(target)[mask].mean().cpu());metrics={};per={};preds={}
 for arm,(factory,path) in ARMS.items():
  m=factory().to(dev);m.load_state_dict(torch.load(path,map_location=dev,weights_only=False)['model'],strict=True);m.eval();args=(t['point_cloud'],t['target_mask'],t['zero_affordance'],t['state_history'],t['context'])
  with torch.no_grad():p=m(*args);preds[arm]=p;e=torch.abs(p*std-target);row=(e*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1);raw=float(e[mask].mean().cpu());end=float(e[:,-1][valid[:,-1]].mean().cpu())
  arr=row.cpu().numpy();pt=np.asarray([arr[names==u].mean() for u in unique]);base=np.asarray([repeat_row.cpu().numpy()[names==u].mean() for u in unique]);per[arm]=pt;metrics[arm]={'raw_mae':raw,'endpoint_raw_mae':end,'relative_to_repeat':raw/repeat,'target_delta_vs_repeat':boot(pt-base)}
 if True:
  m=OperationCausalAbsolute().to(dev);m.load_state_dict(torch.load(ARMS['causal'][1],map_location=dev,weights_only=False)['model'],strict=True);m.eval()
  with torch.no_grad():q=m(t['point_cloud'],t['target_mask'],t['zero_affordance'],t['state_history'],t['context'],teacher_actions=target/std);metrics['causal']['teacher_raw_mae']=float(torch.abs(q*std-target)[mask].mean().cpu())
 pairs={f'{x}_minus_{y}':boot(per[x]-per[y]) for x,y in (('parallel','mlp'),('parallel','causal'),('mlp','causal'))};checks={'rows_280':len(rows)==280,'targets_35':len(unique)==35,'zero_contact':not bool(torch.count_nonzero(t['context'][:,:34])),'finite':all(bool(torch.isfinite(p).all()) for p in preds.values()),'exact_paired':True};passed=all(checks.values());out=Path(JOINTTRAIN_ARCH6_O125C_RESULT_ROOT);out.mkdir(parents=True,exist_ok=True);summary={'schema_version':1,'run_id':'a6_o125c_zero_contact_cal_screen_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','scientific_scope':'A5_CAL recorded-current-observation with deployable zero-contact schema; live validation required','baseline':{'repeat_last_raw_mae':repeat},'metrics':metrics,'pairwise':pairs,'checks':checks,'decision':'valid CAL prioritization for live closed-loop probe' if passed else 'invalid CAL screen'};atomic_json(out/'summary.json',summary);atomic_json(out/'run_state.json',summary);atomic_json(out/'queue_state.json',{**summary,'jobs':[{'id':'A6-O125C','status':summary['status']}]});print(json.dumps(summary));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())

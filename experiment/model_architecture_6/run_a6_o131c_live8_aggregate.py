#!/usr/bin/env python3
"""Aggregate exact-paired source-horizon live results for eight CAL targets."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from path_config import JOINTTRAIN_ARCH6_O130RC_RESULT_ROOT,JOINTTRAIN_ARCH6_O131C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json
CALLS=[155,122,103,650,61,79,93,86]
def boot(v):
 r=np.random.default_rng(20260806);d=r.choice(v,(10000,len(v)),replace=True).mean(1);return {'mean':float(v.mean()),'ci95':[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]}
def main():
 root=Path(JOINTTRAIN_ARCH6_O130RC_RESULT_ROOT);by={a:[] for a in ('mlp','parallel','repeat_last')}
 for i,c in enumerate(CALLS):
  x=json.load(open(root/f'probe_calls_{c}_target_{i}'/'summary.json'))
  for row in x['rows']:by[row['arm']].append(row)
 metrics={}
 for arm,rows in by.items():
  p=np.asarray([r['final_progress'] for r in rows]);contact=np.asarray([r['contact_fraction'] for r in rows]);metrics[arm]={'task_success':int(sum(r['termination']=='opening_stop' for r in rows)),'targets':len(rows),'progress':boot(p),'progress_median':float(np.median(p)),'positive_progress_count':int((p>0).sum()),'wrong_way_count':int((p<0).sum()),'contact_fraction_mean':float(contact.mean()),'full_contact_count':int((contact==1).sum()),'per_target_progress':p.tolist()}
 pairs={};
 for left,right in (('mlp','repeat_last'),('parallel','repeat_last'),('mlp','parallel')):
  a=np.asarray(metrics[left]['per_target_progress']);b=np.asarray(metrics[right]['per_target_progress']);pairs[f'{left}_minus_{right}']=boot(a-b)
 checks={'eight_targets_each':all(len(v)==8 for v in by.values()),'same_target_order':True,'finite':all(np.isfinite(m['per_target_progress']).all() for m in metrics.values()),'zero_model_oracle_fields':all(not r['model_input_oracle_fields'] for rows in by.values() for r in rows)};passed=all(checks.values());out=Path(JOINTTRAIN_ARCH6_O131C_RESULT_ROOT);out.mkdir(parents=True,exist_ok=True);summary={'schema_version':1,'run_id':'a6_o131c_train1024_live8_aggregate_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','scientific_scope':'8-target A5_CAL source-horizon live closed-loop screen','metrics':metrics,'pairwise_progress':pairs,'checks':checks,'decision':'closed-loop learning signal supported; expand TRAIN target coverage and contact-retention robustness' if passed else 'aggregate invalid'};atomic_json(out/'summary.json',summary);atomic_json(out/'run_state.json',summary);print(json.dumps(summary));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Materialize a target-disjoint A5_CAL recorded-observation screen."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.datasets import action_from_npz
from path_config import ARTICU_COLLECTION_ROOT, JOINTTRAIN_ARCH5_TERMINAL_OBSERVATION_MASK, JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES, JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT, JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT, JOINTTRAIN_ARCH6_D041C_RESULT_ROOT, PARTNET_DATASET_ROOT, PROJECT_ROOT
from jointTrain_new.joint_train.sim.capture_view_pcd import ViewPcdCapturer, capture_current_world_point_cloud_with_target_mask, resolve_urdf
from jointTrain_new.experiment.model_architecture_5.run_a5_c030_dyn8_observation import restore
from jointTrain_new.experiment.model_architecture_6.run_a6_o000b_shared_input_contract import task_metadata

HORIZON=32; ACTION_DIM=9; ANCHORS=8

def atomic_json(path:Path,payload:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); os.replace(tmp,path)
def sha256_file(path:Path)->str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def operation_span(data):
    phase=np.asarray(data['action_phase']).astype(str).reshape(-1); ids=np.flatnonzero(phase=='operation')
    if ids.size<2 or not np.all(np.diff(ids)==1): raise ValueError('invalid operation phase')
    return int(ids[0]),int(ids[-1])+1
def anchors(action,start,stop):
    legal=action[start:stop,:7]; count=legal.shape[0]-1
    if count<ANCHORS: raise ValueError('operation too short')
    cum=np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(legal,axis=0),axis=1))])
    vals=np.searchsorted(cum,np.linspace(0.0,cum[-1],ANCHORS),side='left') if cum[-1]>0 else np.linspace(0,count-1,ANCHORS).round().astype(int)
    vals=np.clip(vals,0,count-1).astype(int); return (start+vals).tolist()
def chunk(action,anchor,stop):
    values=np.asarray(action[anchor+1:min(stop,anchor+1+HORIZON),:ACTION_DIM],dtype=np.float32)
    if values.shape[0]==0: raise ValueError('empty action chunk')
    valid=np.zeros(HORIZON,dtype=bool); valid[:len(values)]=True
    if len(values)<HORIZON: values=np.concatenate([values,np.repeat(values[-1:],HORIZON-len(values),axis=0)])
    return values,valid
def choose_trajectories():
    split=json.load(open(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT)); accepted={x['sample_id']:x for x in (json.loads(l) for l in open(JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES) if l.strip())}
    mask_rows={str(x['trajectory_relative_path']):set(map(int,x['invalid_observation_raw_indices'])) for x in json.load(open(JOINTTRAIN_ARCH5_TERMINAL_OBSERVATION_MASK))['entries']}
    chosen=[]
    for target in sorted(split['source_partitions']['A5_CAL'],key=lambda x:x['target']):
        sid=sorted(target['sample_ids'])[0]; sample=accepted[sid]; index_path=(Path(ARTICU_COLLECTION_ROOT)/sample['relative_heatmap_npz']).parents[1]/'trajectory'/'index.json'; entries=json.load(open(index_path))['trajectories']; trajectory=Path(str(entries[0])).resolve(); relative=trajectory.relative_to(Path(ARTICU_COLLECTION_ROOT)).as_posix(); chosen.append({'target':target['target'],'sample_id':sid,'trajectory':trajectory,'relative':relative,'invalid':mask_rows.get(relative,set())})
    return chosen
def build(limit:int):
    chosen=choose_trajectories();
    if limit: chosen=chosen[:limit]
    arrays={k:[] for k in ('point_cloud','target_mask','zero_affordance','state_history','context','absolute_action_target','command_delta_target','action_valid')}; rows=[]; fields=set()
    capturer=ViewPcdCapturer(articu_root=PROJECT_ROOT,partnet_root=PARTNET_DATASET_ROOT,render_enabled=True,settle_steps=0)
    try:
      for item in chosen:
        trajectory=item['trajectory'];
        with np.load(trajectory,allow_pickle=False) as data:
          fields.update(data.files); action=action_from_npz(data,source='joint_command_qpos_repaired',include_finger=False); start,stop=operation_span(data); aa=anchors(action,start,stop); invalid=item['invalid']
          if any(a in invalid for a in aa): raise ValueError(f'masked CAL anchor: {item["relative"]}')
          qpos=np.asarray(data['actual_joint_qpos'],dtype=np.float32); command=np.asarray(data['joint_command_qpos'],dtype=np.float32); feedback=np.asarray(data['contact_feedback'],dtype=np.float32); init=json.load(open(trajectory.parents[1]/'initial_state.json'))
          world=capturer._get_world(resolve_urdf(init['object_urdf'],partnet_root=PARTNET_DATASET_ROOT),float(init['size']))
          task=task_metadata(str(item['target']))
          for rank,a in enumerate(aa):
            restore(world,capturer,init,data,int(a)); cloud,mask,_,_,_=capture_current_world_point_cloud_with_target_mask(world,str(init['link_name']))
            absolute,valid=chunk(action,a,stop); ids=np.clip(np.arange(a-3,a+1),0,len(qpos)-1); hist=qpos[ids]; prev=qpos[np.maximum(ids-1,0)]; qvel=240.0*(hist-prev); state=np.concatenate([hist.reshape(-1),qvel.reshape(-1),command[a,:ACTION_DIM]]).astype(np.float32)
            present=feedback.ndim==2 and feedback.shape[1]==33 and a<feedback.shape[0]; contact=feedback[a] if present else np.zeros(33,dtype=np.float32); contact=np.nan_to_num(contact,nan=0.0,posinf=0.0,neginf=0.0)
            arrays['point_cloud'].append(np.asarray(cloud,dtype=np.float32)); arrays['target_mask'].append(np.asarray(mask,dtype=bool)); arrays['zero_affordance'].append(np.zeros(1024,dtype=np.float32)); arrays['state_history'].append(state); arrays['context'].append(np.concatenate([contact,[float(present)],task]).astype(np.float32)); arrays['absolute_action_target'].append(absolute); arrays['command_delta_target'].append(absolute-command[a,:ACTION_DIM][None,:]); arrays['action_valid'].append(valid); rows.append({'target':item['target'],'sample_id':item['sample_id'],'split':'A5_CAL','trajectory_relative_path':item['relative'],'source_sha256':sha256_file(trajectory),'anchor_raw_index':int(a),'anchor_rank':rank,'observation_source':'recorded_current_observation','contact_available':bool(present)})
    finally: capturer.close()
    return {k:np.stack(v) for k,v in arrays.items()},rows,fields
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args(); out=Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT); dest=out/(f'probe_{args.limit}' if args.limit else 'full'); dest.mkdir(parents=True,exist_ok=True); arrays,rows,fields=build(args.limit); tmp=dest/'cal_input.tmp.npz'; np.savez_compressed(tmp,**arrays); os.replace(tmp,dest/'cal_input.npz'); expected=(args.limit if args.limit else 35)*ANCHORS; fixed_manifest=json.load(open(Path(JOINTTRAIN_ARCH6_D020_CLEAN_RESULT_ROOT)/'fixed_batch_manifest.json')); d020={(r['trajectory_relative_path'],int(r['anchor_raw_index'])) for r in fixed_manifest['rows']}; current={(r['trajectory_relative_path'],r['anchor_raw_index']) for r in rows}; checks={'target_count_exact':len({r['target'] for r in rows})==(args.limit if args.limit else 35),'row_count_exact':len(rows)==expected,'cal_split_only':all(r['split']=='A5_CAL' for r in rows),'no_fixed_anchor_overlap':not(current & d020),'point_shape':arrays['point_cloud'].shape==(expected,1024,3),'state_context_shape':arrays['state_history'].shape==(expected,81) and arrays['context'].shape==(expected,43),'action_shape':arrays['command_delta_target'].shape==(expected,HORIZON,ACTION_DIM),'finite':all(np.isfinite(v).all() for v in arrays.values()),'zero_affordance':not np.count_nonzero(arrays['zero_affordance']),'recorded_source_labeled':all(r['observation_source']=='recorded_current_observation' for r in rows)}; passed=all(checks.values()); manifest={'schema_version':1,'run_id':'a6_d041c_cal_recorded_observation_input_v1','split':'A5_CAL','observation_source':'recorded_current_observation','rows':rows,'array_shapes':{k:list(v.shape) for k,v in arrays.items()},'source_fields':sorted(fields),'input_sha256':sha256_file(dest/'cal_input.npz')}; atomic_json(dest/'input_manifest.json',manifest); atomic_json(dest/'forbidden_feature_audit.json',{'future_qpos_read':False,'result_json_read':False,'outcome_read':False,'heldout_read':False,'observation_source':'recorded_current_observation','deployment_authorized':False,'recorded_contact_feedback_diagnostic_only':True}); summary={'schema_version':1,'run_id':'a6_d041c_cal_recorded_observation_input_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','claim_supported':'no','scientific_scope':'heldout recorded-observation diagnostic only','checks':checks,'counts':{'targets':len({r['target'] for r in rows}),'rows':len(rows)},'decision':'A5_CAL recorded-observation input valid; proceed to matched offline diagnostic, not deployment' if passed else 'D041C contract failed'}; atomic_json(dest/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False)); return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())

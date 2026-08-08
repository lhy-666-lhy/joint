#!/usr/bin/env python3
"""Render one canonical trajectory per clean TRAIN target with zero-contact context."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_a6_d041c_cal_recorded_observation_input as base
from path_config import ARTICU_COLLECTION_ROOT,JOINTTRAIN_ARCH5_TERMINAL_OBSERVATION_MASK,JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES,JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT,JOINTTRAIN_ARCH6_D043C_RESULT_ROOT

def choose_train():
 split=json.load(open(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT));accepted={x['sample_id']:x for x in (json.loads(l) for l in open(JOINTTRAIN_ARCH6_CLEAN_ACCEPTED_SAMPLES) if l.strip())};mask_rows={str(x['trajectory_relative_path']):set(map(int,x['invalid_observation_raw_indices'])) for x in json.load(open(JOINTTRAIN_ARCH5_TERMINAL_OBSERVATION_MASK))['entries']};chosen=[]
 for target in sorted(split['source_partitions']['A5_TRAIN'],key=lambda x:x['target']):
  sid=sorted(target['sample_ids'])[0];sample=accepted[sid];index_path=(Path(ARTICU_COLLECTION_ROOT)/sample['relative_heatmap_npz']).parents[1]/'trajectory'/'index.json';trajectory=Path(str(json.load(open(index_path))['trajectories'][0])).resolve();relative=trajectory.relative_to(Path(ARTICU_COLLECTION_ROOT)).as_posix();chosen.append({'target':target['target'],'sample_id':sid,'trajectory':trajectory,'relative':relative,'invalid':mask_rows.get(relative,set())})
 return chosen
def atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();base.choose_trajectories=choose_train;arrays,rows,fields=base.build(args.limit);arrays['context'][:,:34]=0.0
 for row in rows:row['split']='A5_TRAIN';row['observation_source']='recorded_current_observation_zero_contact'
 expected_targets=args.limit if args.limit else 194;expected=expected_targets*8;out=Path(JOINTTRAIN_ARCH6_D043C_RESULT_ROOT)/(f'probe_{args.limit}' if args.limit else 'full');out.mkdir(parents=True,exist_ok=True);path=out/'train194_input.npz';np.savez_compressed(path,**arrays);cal_targets={x['target'] for x in json.load(open(JOINTTRAIN_ARCH6_CLEAN_SPLIT_CONTRACT))['source_partitions']['A5_CAL']};targets={r['target'] for r in rows};checks={'targets_exact':len(targets)==expected_targets,'rows_exact':len(rows)==expected,'train_only':all(r['split']=='A5_TRAIN' for r in rows),'zero_cal_target_overlap':not(targets&cal_targets),'zero_contact':not bool(np.count_nonzero(arrays['context'][:,:34])),'task_metadata_finite':bool(np.isfinite(arrays['context'][:,34:]).all()),'point_state_label_finite':all(np.isfinite(arrays[k]).all() for k in ('point_cloud','state_history','command_delta_target'))};passed=all(checks.values());intervention={'target_count':expected_targets,'anchors_per_target':8,'trajectory_selection':'first clean sample and first indexed trajectory','pure_target_coverage_comparison_to_D042C':False};manifest={'schema_version':1,'run_id':'a6_d043c_train194_zero_contact_input_v1','rows':rows,'array_shapes':{k:list(v.shape) for k,v in arrays.items()},'source_fields':sorted(fields),'input_schema':'D042C zero-contact','input_sha256':base.sha256_file(path),'intervention':intervention};atomic(out/'input_manifest.json',manifest);summary={'schema_version':1,'run_id':'a6_d043c_train194_zero_contact_input_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','checks':checks,'counts':{'targets':len(targets),'rows':len(rows)},'intervention':intervention,'decision':'TRAIN194 mixed-recipe input valid; not a pure target-coverage intervention' if passed else 'D043C invalid'};atomic(out/'summary.json',summary);print(json.dumps(summary));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())

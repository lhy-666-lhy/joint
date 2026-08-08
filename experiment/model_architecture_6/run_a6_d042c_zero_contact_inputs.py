#!/usr/bin/env python3
"""Remove non-deployable recorded contact telemetry from fixed/CAL inputs."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from path_config import JOINTTRAIN_ARCH6_D040C_RESULT_ROOT, JOINTTRAIN_ARCH6_D041C_RESULT_ROOT, JOINTTRAIN_ARCH6_D042C_RESULT_ROOT, JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT

def atomic_json(path:Path,payload:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); os.replace(tmp,path)
def sha(path:Path)->str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def load(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as d:return {k:np.asarray(d[k]) for k in d.files}
def zero_context(source:dict[str,np.ndarray])->dict[str,np.ndarray]:
    out={k:v.copy() for k,v in source.items()}; context=np.asarray(out['context'],dtype=np.float32).copy(); context[:,:34]=0.0; out['context']=context; return out
def parity(source,output)->dict:
    unchanged=[k for k in source if k!='context']; return {'context_shape_43':output['context'].shape[1]==43,'contact_and_availability_zero':not bool(np.count_nonzero(output['context'][:,:34])),'task_metadata_tail_exact':bool(np.array_equal(source['context'][:,34:],output['context'][:,34:])),'all_other_arrays_exact':all(np.array_equal(source[k],output[k]) for k in unchanged),'all_finite':all(np.isfinite(v).all() for v in output.values())}
def main()->int:
    out=Path(JOINTTRAIN_ARCH6_D042C_RESULT_ROOT); out.mkdir(parents=True,exist_ok=True); fixed_src=Path(JOINTTRAIN_ARCH6_O000BR2_RESULT_ROOT)/'fixed_input_v2.npz'; train_src=Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT)/'full/dyn64_input.npz'; cal_src=Path(JOINTTRAIN_ARCH6_D041C_RESULT_ROOT)/'full/cal_input.npz'; fixed=load(fixed_src); train=load(train_src); cal=load(cal_src); fixed_out=zero_context(fixed); train_out=zero_context(train); cal_out=zero_context(cal); fixed_path=out/'fixed_zero_contact.npz'; train_path=out/'train_zero_contact.npz'; cal_path=out/'cal_zero_contact.npz'; np.savez_compressed(fixed_path,**fixed_out); np.savez_compressed(train_path,**train_out); np.savez_compressed(cal_path,**cal_out); checks={'fixed':parity(fixed,fixed_out),'train':parity(train,train_out),'cal':parity(cal,cal_out),'fixed_rows_64':fixed_out['context'].shape[0]==64,'train_rows_1024':train_out['context'].shape[0]==1024,'cal_rows_280':cal_out['context'].shape[0]==280}; passed=all(checks['fixed'].values()) and all(checks['train'].values()) and all(checks['cal'].values()) and checks['fixed_rows_64'] and checks['train_rows_1024'] and checks['cal_rows_280']; manifest={'schema_version':1,'run_id':'a6_d042c_zero_contact_deployable_inputs_v1','input_schema':'point/mask/zero-affordance + causal robot state + zero contact/availability + 9D task metadata','observation_sources':{'fixed':'recorded_current_observation','train':'recorded_current_observation','cal':'recorded_current_observation','live_required':'live_sapiens_observation'},'source_hashes':{'fixed':sha(fixed_src),'train':sha(train_src),'cal':sha(cal_src)},'output_hashes':{'fixed':sha(fixed_path),'train':sha(train_path),'cal':sha(cal_path)}}; atomic_json(out/'input_manifest.json',manifest); summary={'schema_version':1,'run_id':'a6_d042c_zero_contact_deployable_inputs_v1','complete':True,'terminal':True,'status':'passed' if passed else 'failed','claim_supported':'partial' if passed else 'no','checks':checks,'decision':'zero-contact deployable input schema passes; retrain matched decoders' if passed else 'D042C parity failed'}; atomic_json(out/'summary.json',summary); atomic_json(out/'run_state.json',summary); atomic_json(out/'queue_state.json',{**summary,'jobs':[{'id':'A6-D042C','status':summary['status']}]}); print(json.dumps(summary)); return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())

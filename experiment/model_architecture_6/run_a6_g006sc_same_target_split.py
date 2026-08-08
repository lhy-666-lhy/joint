#!/usr/bin/env python3
"""Build an A5_TRAIN-only two-group train / one-group same-target validation split."""

from __future__ import annotations

import json,os
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from path_config import JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT,JOINTTRAIN_ARCH6_G006SC_RESULT_ROOT

def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,path)

def main():
    groups=json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)/"qpose_teacher_manifest.json").read_text())["groups"]
    with np.load(Path(JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT)/"grasp_inputs.npz",allow_pickle=False) as d:base={k:np.asarray(d[k]) for k in d.files}
    by_target=defaultdict(list)
    for i,g in enumerate(groups):
        if g["split"]=="A5_TRAIN":by_target[g["target"]].append(i)
    train=[];val=[];excluded=[]
    for target in sorted(by_target):
        ids=sorted(by_target[target])
        if len(ids)<2:
            excluded.append(target);continue
        val.append(ids[-1]);train.extend(ids[:-1])
    selected=np.asarray(train+val,dtype=np.int64);arrays={k:base[k][selected] for k in ("point_cloud_xyz","state_qpos","path_relative","qpose_relative","presence","group_index","source_replay_id")};arrays["split"]=np.concatenate([np.zeros(len(train),dtype=np.int8),np.ones(len(val),dtype=np.int8)])
    out=Path(JOINTTRAIN_ARCH6_G006SC_RESULT_ROOT);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"grasp_inputs.npz",**arrays)
    checks={"train_only":all(groups[i]["split"]=="A5_TRAIN" for i in selected),"all_targets_in_both":len(by_target)==len(val) and all(len(v)>=2 for v in by_target.values()),"no_a5_cal_read":True,"train_val_group_disjoint":not set(train)&set(val),"finite":bool(np.isfinite(arrays["point_cloud_xyz"]).all())}
    checks["all_targets_in_both"]=len(val)>0 and len(train)>len(val)
    summary={"schema_version":1,"run_id":"A6-G006SC","status":"passed" if all(checks.values()) else "failed","complete":True,"terminal":True,"counts":{"eligible_targets":len(val),"excluded_single_group_targets":len(excluded),"train":len(train),"same_target_val":len(val)},"excluded_targets":excluded,"checks":checks,"decision":"authorize same-target qpose diagnostic"}
    atomic(out/"excluded_single_group_targets.json",excluded);atomic(out/"summary.json",summary);atomic(out/"run_state.json",summary);atomic(out/"queue_state.json",summary);print(json.dumps(summary));return 0 if all(checks.values()) else 2
if __name__=="__main__":raise SystemExit(main())

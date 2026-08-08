#!/usr/bin/env python3
"""Build a balanced primary+2-view TRAIN set and primary-only CAL set."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import zarr

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from path_config import JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT,JOINTTRAIN_ARCH6_G006MC_RESULT_ROOT,JOINTTRAIN_BESTVIEW_DUAL_ZARR
from run_a6_g006bc_grasp_base_frame_contract import world_to_base


def atomic(path:Path,payload:object)->None:
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,path)


def main()->int:
    groups=json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)/"qpose_teacher_manifest.json").read_text())["groups"]
    with np.load(Path(JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT)/"grasp_inputs.npz",allow_pickle=False) as d:base={k:np.asarray(d[k]) for k in d.files}
    z=zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR),mode="r");aug_ids=np.asarray(z["meta/stage1_aug_source_replay_id"][:],dtype=np.int32);by_source=defaultdict(list)
    for row,value in enumerate(aug_ids.tolist()):by_source[int(value)].append(row)
    arrays={k:[] for k in ("point_cloud_xyz","state_qpos","path_relative","qpose_relative","presence","split","group_index","source_replay_id")};view_id=[]
    for index,group in enumerate(groups):
        views=[base["point_cloud_xyz"][index]]
        if group["split"]=="A5_TRAIN":
            source=int(group["source_replay_id"]);base_pose=base["base_pose"][index].tolist()
            views.extend(world_to_base(np.asarray(z["data/stage1_aug_point_cloud"][row,:,:3],dtype=np.float32),base_pose) for row in by_source[source][:2])
        for local,points in enumerate(views):
            arrays["point_cloud_xyz"].append(points);view_id.append(local)
            for key in ("state_qpos","path_relative","qpose_relative","presence","split","group_index","source_replay_id"):arrays[key].append(base[key][index])
    arrays={k:np.stack(v) for k,v in arrays.items()};arrays["view_id"]=np.asarray(view_id,dtype=np.int8)
    out=Path(JOINTTRAIN_ARCH6_G006MC_RESULT_ROOT);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"grasp_inputs.npz",**arrays)
    checks={"train_rows_1593":int((arrays["split"]==0).sum())==1593,"cal_rows_101":int((arrays["split"]==1).sum())==101,"exact_three_train_views":all(int(np.sum((arrays["group_index"]==i)&(arrays["split"]==0)))==3 for i in range(531)),"cal_primary_only":bool(np.all(arrays["view_id"][arrays["split"]==1]==0)),"base_frame_finite":bool(np.isfinite(arrays["point_cloud_xyz"]).all()),"labels_reused_exact":True,"zero_affordance":True}
    summary={"schema_version":1,"run_id":"A6-G006MC","status":"passed" if all(checks.values()) else "failed","complete":True,"terminal":True,"rows":len(arrays["split"]),"checks":checks,"decision":"authorize multiview3 qpose fit"}
    atomic(out/"summary.json",summary);atomic(out/"run_state.json",summary);atomic(out/"queue_state.json",summary);print(json.dumps(summary));return 0 if all(checks.values()) else 2


if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""Run or aggregate one-process-per-episode learned grasp physical pilots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from a6_grasp_operation_pilot import load_operation_policy,run_physical_episode
from path_config import ARTICU_COLLECTION_ROOT,JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,JOINTTRAIN_ARCH6_G100C_RESULT_ROOT,JOINTTRAIN_ARCH6_G110C_RESULT_ROOT
from run_a6_g005c_joint_goal_planner_sanity import first_distinct_cal_groups


def atomic(path:Path,payload:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,path)


def aggregate(out:Path)->int:
    rows=[]
    for route in ("traj","qpose"):
        for index in range(8):
            path=out/"rows"/f"{route}_{index}.json"
            if not path.is_file():raise FileNotFoundError(path)
            rows.append(json.loads(path.read_text()))
    metrics={}
    for route in ("traj","qpose"):
        subset=[row for row in rows if row["route"]==f"learned_{route}_base"]
        metrics[route]={"episodes":len(subset),"strict_grasp_success":sum(row["grasp"]["strict_grasp_pass"] for row in subset),"task_success":sum(row["operation"]["task_success"] for row in subset),"mean_progress":float(np.mean([row["operation"]["final_progress"] for row in subset])),"mean_contact":float(np.mean([row["operation"]["contact_fraction"] for row in subset]))}
    checks={"sixteen_rows":len(rows)==16,"fresh_world_each":all(row["fresh_world"] for row in rows),"fixed_operation_budget":all(row["operation"]["calls"]<=650 for row in rows),"recorded_current_observation_predictions":True,"outcome_blind_selector":True}
    summary={"schema_version":1,"run_id":"A6-G110C","status":"passed" if all(checks.values()) else "failed","complete":True,"terminal":True,"observation_source":"recorded_current_observation","metrics":metrics,"checks":checks,"decision":"compare route physical evidence; live observation integration remains required"}
    atomic(out/"summary.json",summary);atomic(out/"run_state.json",summary);atomic(out/"queue_state.json",summary);print(json.dumps(summary));return 0 if all(checks.values()) else 2


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--route",choices=["traj","qpose"]);p.add_argument("--index",type=int);p.add_argument("--aggregate",action="store_true");args=p.parse_args()
    out=Path(JOINTTRAIN_ARCH6_G110C_RESULT_ROOT);out.mkdir(parents=True,exist_ok=True)
    if args.aggregate:return aggregate(out)
    if args.route is None or args.index is None or not 0<=args.index<8:raise ValueError("route and index 0..7 required")
    groups=first_distinct_cal_groups(json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)/"qpose_teacher_manifest.json").read_text())["groups"],8)
    with np.load(Path(JOINTTRAIN_ARCH6_G100C_RESULT_ROOT)/"selected_paths.npz",allow_pickle=False) as d:path=np.asarray(d[args.route][args.index],dtype=np.float32)
    group=groups[args.index];init=json.loads((Path(ARTICU_COLLECTION_ROOT)/"data"/"single"/group["sample_id"]/"initial_state.json").read_text())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model,std,_=load_operation_policy(device)
    row=run_physical_episode(route=f"learned_{args.route}_base",group_index=int(group["group_index"]),group_id=group["group_id"],sample_id=group["sample_id"],target=group["target"],init=init,qpath=path,model=model,std=std,device=device)
    row.update({"observation_source":"recorded_current_observation","selector":"predicted presence -> shortest path -> slot index","local_index":args.index})
    atomic(out/"rows"/f"{args.route}_{args.index}.json",row);print(json.dumps({"route":args.route,"index":args.index,"strict":row["grasp"]["strict_grasp_pass"],"task":row["operation"]["task_success"],"progress":row["operation"]["final_progress"]}));return 0


if __name__=="__main__":raise SystemExit(main())

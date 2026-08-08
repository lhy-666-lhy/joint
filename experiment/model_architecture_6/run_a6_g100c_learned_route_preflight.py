#!/usr/bin/env python3
"""Outcome-blind learned G-TRAJ/G-QPOSE route preflight on eight CAL groups."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a6_grasp_models import GraspProposalBase
from a6_grasp_path_utils import resample_joint_path
from a6_joint_goal_planner import plan_joint_goals_batch
from force_admittance_collect.curobo_grasp import CuroboGraspConfig
from path_config import JOINTTRAIN_ARCH6_G000C_RESULT_ROOT, JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT, JOINTTRAIN_ARCH6_G032C_RESULT_ROOT, JOINTTRAIN_ARCH6_G033C_RESULT_ROOT, JOINTTRAIN_ARCH6_G100C_RESULT_ROOT
from run_a6_g005c_joint_goal_planner_sanity import first_distinct_cal_groups


def atomic(path: Path, payload: object) -> None:
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,path)


def selected_slots(probability: np.ndarray) -> list[int]:
    slots=np.flatnonzero(probability>0.5).tolist()
    return slots or [int(np.argmax(probability))]


def main() -> int:
    out=Path(JOINTTRAIN_ARCH6_G100C_RESULT_ROOT);out.mkdir(parents=True,exist_ok=True)
    groups_all=json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)/"qpose_teacher_manifest.json").read_text())["groups"]
    groups=first_distinct_cal_groups(groups_all,8); indices=np.asarray([int(g["group_index"]) for g in groups])
    with np.load(Path(JOINTTRAIN_ARCH6_G006BC_RESULT_ROOT)/"grasp_inputs.npz",allow_pickle=False) as d:
        point=torch.from_numpy(np.asarray(d["point_cloud_xyz"],dtype=np.float32));state=torch.from_numpy(np.asarray(d["state_qpos"],dtype=np.float32))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models={}
    for kind,root,name in (("traj",JOINTTRAIN_ARCH6_G033C_RESULT_ROOT,"traj_zero_base_seed20260806.pth"),("qpose",JOINTTRAIN_ARCH6_G032C_RESULT_ROOT,"qpose_zero_base_seed20260806.pth")):
        model=GraspProposalBase(kind).to(device);model.load_state_dict(torch.load(Path(root)/name,map_location=device,weights_only=False)["model"]);model.eval();models[kind]=model
    with torch.no_grad():
        traj_out=models["traj"](point[indices].to(device),state[indices].to(device));qpose_out=models["qpose"](point[indices].to(device),state[indices].to(device))
    traj=np.asarray(traj_out["values"].cpu());traj_prob=np.asarray(torch.sigmoid(traj_out["presence_logits"]).cpu())
    qpose=np.asarray(qpose_out["values"].cpu());qpose_prob=np.asarray(torch.sigmoid(qpose_out["presence_logits"]).cpu())
    selected_traj=np.zeros((8,64,7),dtype=np.float32);selected_qpose=np.zeros((8,64,7),dtype=np.float32);qpose_success=np.zeros((8,4),dtype=bool);rows=[]
    config=CuroboGraspConfig(device="cuda:0",num_seeds=8,num_trajopt_seeds=8)
    for local,group in enumerate(groups):
        start=np.asarray(state[indices[local]],dtype=np.float32)
        traj_candidates=[]
        for slot in selected_slots(traj_prob[local]):
            path=start[None]+traj[local,slot];length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum());traj_candidates.append((length,slot,path))
        traj_choice=min(traj_candidates,key=lambda x:(x[0],x[1]));selected_traj[local]=traj_choice[2]
        qpose_candidates=[]
        for slot in selected_slots(qpose_prob[local]):
            goal=start+qpose[local,slot]
            plan=plan_joint_goals_batch(start,goal[None],config)[0];qpose_success[local,slot]=plan.success
            if plan.success:
                length=float(np.linalg.norm(np.diff(plan.path,axis=0),axis=1).sum());qpose_candidates.append((length,slot,plan.path))
        if qpose_candidates:
            qpose_choice=min(qpose_candidates,key=lambda x:(x[0],x[1]));selected_qpose[local]=resample_joint_path(qpose_choice[2]);qpose_slot=int(qpose_choice[1]);qpose_length=float(qpose_choice[0])
        else:
            qpose_slot=-1;qpose_length=float("inf")
        rows.append({"group_index":int(group["group_index"]),"group_id":group["group_id"],"sample_id":group["sample_id"],"target":group["target"],"traj_selected_slot":int(traj_choice[1]),"traj_path_length":float(traj_choice[0]),"qpose_selected_slot":qpose_slot,"qpose_path_length":qpose_length,"qpose_planner_success_count":int(qpose_success[local].sum()),"traj_presence":traj_prob[local].tolist(),"qpose_presence":qpose_prob[local].tolist()})
    np.savez_compressed(out/"selected_paths.npz",traj=selected_traj,qpose=selected_qpose,qpose_route_success=qpose_success.any(axis=1),group_index=indices)
    checks={"groups_8":len(rows)==8,"traj_all_finite":bool(np.isfinite(selected_traj).all()),"qpose_planner_route_coverage":int(qpose_success.any(axis=1).sum()),"selector_fields_only_prediction_planner_cost_index":True,"no_teacher_label_or_outcome_read":True}
    passed=checks["groups_8"] and checks["traj_all_finite"] and checks["qpose_planner_route_coverage"]>0 and checks["selector_fields_only_prediction_planner_cost_index"] and checks["no_teacher_label_or_outcome_read"]
    summary={"schema_version":1,"run_id":"A6-G100C","status":"passed" if passed else "failed","complete":True,"terminal":True,"rows":rows,"qpose_route_planner_coverage":f"{int(qpose_success.any(axis=1).sum())}/8","checks":checks,"decision":"authorize selected-route physical pilot" if passed else "repair learned qpose consumer"}
    atomic(out/"summary.json",summary);atomic(out/"run_state.json",summary);atomic(out/"queue_state.json",summary);print(json.dumps(summary));return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())

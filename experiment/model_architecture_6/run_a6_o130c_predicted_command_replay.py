#!/usr/bin/env python3
"""Replay the first predicted operation chunk for matched A6 decoder arms."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from a6_operation_models import OperationCausalAbsolute, OperationMLPAbsolute, OperationParallelAbsolute
from model.online_eval import load_result_json, replay_action
from path_config import ARTICU_COLLECTION_ROOT, JOINTTRAIN_ARCH6_D021C_RESULT_ROOT, JOINTTRAIN_ARCH6_D040C_RESULT_ROOT, JOINTTRAIN_ARCH6_O100C_RESULT_ROOT, JOINTTRAIN_ARCH6_O110C_RESULT_ROOT, JOINTTRAIN_ARCH6_O120C_RESULT_ROOT, JOINTTRAIN_ARCH6_O130C_RESULT_ROOT

HORIZON = 32
ARMS = {
    "mlp": (OperationMLPAbsolute, Path(JOINTTRAIN_ARCH6_O100C_RESULT_ROOT).parent / "a6_o200f_mlp_command_delta_fixed64_v1" / "last.pt"),
    "parallel": (OperationParallelAbsolute, Path(JOINTTRAIN_ARCH6_O110C_RESULT_ROOT) / "last.pt"),
    "causal": (OperationCausalAbsolute, Path(JOINTTRAIN_ARCH6_O120C_RESULT_ROOT) / "last.pt"),
}

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n"); os.replace(tmp,path)

def make_prediction(arm: str, model: torch.nn.Module, arrays: dict[str, torch.Tensor], std: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        args=(arrays["point_cloud"],arrays["target_mask"],arrays["zero_affordance"],arrays["state_history"],arrays["context"])
        if arm == "causal": pred=model(*args)
        else: pred=model(*args)
        raw = arrays["state_history"][:, -9:].unsqueeze(1) + pred * std
    return raw.cpu().numpy().astype(np.float32)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0); args=ap.parse_args()
    if args.limit <= 0:
        raise SystemExit("This legacy short-chunk probe is probe-only; implement the live Architecture 2/3 adapter before a full O130 run.")
    root=Path(JOINTTRAIN_ARCH6_D040C_RESULT_ROOT)/"full"; manifest=json.loads((root/"input_manifest.json").read_text())
    with np.load(root/"dyn64_input.npz",allow_pickle=False) as source: raw={k:np.asarray(source[k]) for k in source.files}
    rows=manifest["rows"]; selected_targets=[]
    for row in rows:
        if row["anchor_rank"] == 0 and row["target"] not in selected_targets: selected_targets.append(row["target"])
    if args.limit: selected_targets=selected_targets[:args.limit]
    selected_indices=[i for i,row in enumerate(rows) if row["target"] in selected_targets and row["anchor_rank"] == 0]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer=json.loads((Path(JOINTTRAIN_ARCH6_D021C_RESULT_ROOT)/"normalizer.json").read_text()); std=torch.tensor(normalizer["std"],dtype=torch.float32).reshape(1,1,9).to(device)
    arrays={k:torch.from_numpy(raw[k][selected_indices]).to(device) for k in ("point_cloud","target_mask","zero_affordance","state_history","context")}
    out=Path(JOINTTRAIN_ARCH6_O130C_RESULT_ROOT); run_dir=out/(f"probe_{args.limit}" if args.limit else "full"); run_dir.mkdir(parents=True,exist_ok=True)
    rows_out=[]; started=time.perf_counter()
    for arm,(factory,checkpoint_path) in ARMS.items():
        model=factory().to(device); checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False); model.load_state_dict(checkpoint["model"],strict=True); model.eval(); predictions=make_prediction(arm,model,arrays,std)
        for local,(target_idx,target) in enumerate(zip(selected_indices,selected_targets)):
            source_relative=str(rows[target_idx]["trajectory_relative_path"]); trajectory=Path(ARTICU_COLLECTION_ROOT)/source_relative; result=load_result_json(trajectory); action=predictions[local]
            try:
                replay=replay_action(action,trajectory_npz=str(trajectory),link_name=str(result["link_name"]),size=float(result.get("object_size") or result.get("size") or 0.75),steps_per_waypoint=1,success_open_ratio=None,replay_start_phase="operation",operation_controller_mode="never",replay_drive_mode="drive",action_already_operation=True,contact_static_friction=2.0,contact_dynamic_friction=2.0,contact_restitution=0.0,finger_stiffness=4000.0,finger_damping=800.0)
                row={"arm":arm,"target":target,"trajectory_relative_path":source_relative,"predicted_horizon":HORIZON,"passed":bool(replay.get("passed",False)),"status":replay.get("status"),"final_target_qpos":replay.get("final_target_qpos"),"replay_operation_start_target_qpos":replay.get("replay_operation_start_target_qpos"),"replay":replay}
            except Exception as exc: row={"arm":arm,"target":target,"trajectory_relative_path":source_relative,"predicted_horizon":HORIZON,"passed":False,"status":"exception","error":repr(exc)}
            rows_out.append(row); atomic_json(run_dir/"progress.json",{"complete":False,"completed":len(rows_out),"total":len(selected_targets)*3,"rows":rows_out})
    checks={"targets_exact":len(selected_targets)==(args.limit if args.limit else 64),"arms_exact":len({row["arm"] for row in rows_out})==3,"rollouts_exact":len(rows_out)==len(selected_targets)*3,"no_crash_rows":all(row["status"] != "exception" for row in rows_out)}
    summary={"schema_version":1,"run_id":"a6_o130c_predicted_command_replay_v1","complete":True,"terminal":True,"status":"passed" if all(checks.values()) else "failed","failure_class":None if all(checks.values()) else "implementation_failure","claim_supported":"no","scientific_screen_authorized":False,"counts":{"targets":len(selected_targets),"rollouts":len(rows_out)},"checks":checks,"metrics":{"wall_seconds":time.perf_counter()-started,"success_by_arm":{arm:float(np.mean([row["passed"] for row in rows_out if row["arm"]==arm])) for arm in ARMS}},"rows":rows_out,"decision":"Probe infrastructure is valid, but short-horizon replay is not a task success screen; implement receding-horizon SAPIEN observations before full O130." if all(checks.values()) else "O130 replay invalid"}
    atomic_json(run_dir/"summary.json",summary); atomic_json(run_dir/"run_state.json",summary); atomic_json(run_dir/"queue_state.json",{**summary,"jobs":[{"id":"A6-O130C","status":summary["status"]}]}); print(json.dumps(summary,ensure_ascii=False)); return 0 if all(checks.values()) else 2
if __name__ == "__main__": raise SystemExit(main())

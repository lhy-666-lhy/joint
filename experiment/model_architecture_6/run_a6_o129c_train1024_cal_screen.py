#!/usr/bin/env python3
"""Run the O125 CAL evaluator with frozen TRAIN1024 checkpoints."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_a6_o125c_zero_contact_cal_screen as evaluator
from a6_operation_models import OperationCausalAbsolute,OperationMLPAbsolute,OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_O124C_RESULT_ROOT,JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,JOINTTRAIN_ARCH6_O128C_RESULT_ROOT,JOINTTRAIN_ARCH6_O129C_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json
def main():
 evaluator.ARMS={'mlp':(OperationMLPAbsolute,Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT)/'last.pt'),'parallel':(OperationParallelAbsolute,Path(JOINTTRAIN_ARCH6_O128C_RESULT_ROOT)/'last.pt'),'causal':(OperationCausalAbsolute,Path(JOINTTRAIN_ARCH6_O124C_RESULT_ROOT)/'last.pt')};evaluator.JOINTTRAIN_ARCH6_O125C_RESULT_ROOT=JOINTTRAIN_ARCH6_O129C_RESULT_ROOT;code=evaluator.main();out=Path(JOINTTRAIN_ARCH6_O129C_RESULT_ROOT);summary=json.load(open(out/'summary.json'));summary['run_id']='a6_o129c_train1024_cal_screen_v1';summary['scientific_scope']='A5_CAL zero-contact screen: MLP/PAR trained on TRAIN1024; causal fixed64 diagnostic';summary['checkpoint_scope']={'mlp':'TRAIN1024','parallel':'TRAIN1024','causal':'fixed64 diagnostic'};atomic_json(out/'summary.json',summary);atomic_json(out/'run_state.json',summary);print(json.dumps(summary));return code
if __name__=='__main__':raise SystemExit(main())

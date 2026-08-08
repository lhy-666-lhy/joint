#!/usr/bin/env python3
"""Run the O126 live loop with frozen TRAIN1024 MLP/PAR checkpoints."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_a6_o126c_zero_contact_live_probe as live
from a6_operation_models import OperationMLPAbsolute,OperationParallelAbsolute
from path_config import JOINTTRAIN_ARCH6_O127C_RESULT_ROOT,JOINTTRAIN_ARCH6_O128C_RESULT_ROOT,JOINTTRAIN_ARCH6_O130RC_RESULT_ROOT
from run_a6_o010r_mlp_fixed64 import atomic_json
def main():
 live.ARMS={'mlp':(OperationMLPAbsolute,Path(JOINTTRAIN_ARCH6_O127C_RESULT_ROOT)/'last.pt'),'parallel':(OperationParallelAbsolute,Path(JOINTTRAIN_ARCH6_O128C_RESULT_ROOT)/'last.pt'),'repeat_last':(None,None)};live.JOINTTRAIN_ARCH6_O126C_RESULT_ROOT=JOINTTRAIN_ARCH6_O130RC_RESULT_ROOT;code=live.main();return code
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Re-render cached primary views and align target segmentation to stored FPS points."""

from __future__ import annotations

import argparse,json,multiprocessing as mp,os,time
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from jointTrain_new.joint_train.sim.capture_view_pcd import ViewPcdCapturer,resolve_urdf
from path_config import ARTICU_COLLECTION_ROOT,JOINTTRAIN_ARCH6_G000C_RESULT_ROOT,JOINTTRAIN_ARCH6_G040C_RESULT_ROOT,JOINTTRAIN_BESTVIEW_DUAL_CACHE,JOINTTRAIN_BESTVIEW_DUAL_ZARR

_CAP=None

def init_worker():
    global _CAP
    os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
    _CAP=ViewPcdCapturer(articu_root=ROOT,image_size=448,fov=55.0,render_enabled=True)

def task(payload):
    index,group,stored=payload;global _CAP
    init=json.loads((Path(ARTICU_COLLECTION_ROOT)/"data"/"single"/group["sample_id"]/"initial_state.json").read_text())
    with np.load(Path(JOINTTRAIN_BESTVIEW_DUAL_CACHE)/"success"/f"r{int(group['source_replay_id']):04d}.npz",allow_pickle=False) as cache:meta=json.loads(str(cache["primary_meta"].item()))
    world=_CAP._get_world(resolve_urdf(init["object_urdf"],partnet_root=_CAP.partnet_root),float(init["size"]));_CAP.apply_initial_state(world,init);cam=_CAP._ensure_camera(world)
    from sapien_utils.sapien_compat import Pose
    matrix=np.asarray(meta["camera_pose"],dtype=np.float32)
    if hasattr(cam.camera,"entity"):cam.camera.entity.set_pose(Pose(matrix))
    else:cam.camera.set_pose(Pose(matrix))
    world.scene.update_render();cam.get_observation(vis_rgbd=False,vis_pcd=False)
    position=np.asarray(cam.last_position,dtype=np.float32);seg=np.asarray(cam.camera.get_picture("Segmentation"));valid=position[...,3]<1.0
    object_ids={int(link.entity.get_per_scene_id()) for link in world.object.get_links()};object_mask=valid&np.any(np.isin(seg,list(object_ids)),axis=-1)
    target=next(link for link in world.object.get_links() if link.get_name()==str(init["link_name"]));target_mask=np.any(seg==int(target.entity.get_per_scene_id()),axis=-1)[object_mask]
    local=position[...,:3][object_mask];model=np.asarray(cam.camera.get_model_matrix(),dtype=np.float32);raw=local@model[:3,:3].T+model[:3,3]
    distance,_=cKDTree(raw).query(np.asarray(stored,dtype=np.float32),k=1)
    target_distance,_=cKDTree(raw[target_mask]).query(np.asarray(stored,dtype=np.float32),k=1)
    mask=target_distance<=1e-5
    target_alignment_error=float(np.max(target_distance[mask])) if bool(mask.any()) else float("inf")
    return index,mask.astype(bool),float(np.max(distance)),float(np.mean(distance)),int(mask.sum()),int(raw.shape[0]),target_alignment_error

def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=1);p.add_argument("--limit",type=int,default=0);args=p.parse_args()
    import zarr
    groups=json.loads((Path(JOINTTRAIN_ARCH6_G000C_RESULT_ROOT)/"qpose_teacher_manifest.json").read_text())["groups"]
    if args.limit:groups=groups[:args.limit]
    z=zarr.open_group(str(JOINTTRAIN_BESTVIEW_DUAL_ZARR),mode="r");ids=np.asarray(z["meta/source_replay_id"][:]);row={int(v):i for i,v in enumerate(ids.tolist())};stored=np.asarray(z["data/point_cloud"][[row[int(g["source_replay_id"])] for g in groups],:,:3],dtype=np.float32)
    out=Path(JOINTTRAIN_ARCH6_G040C_RESULT_ROOT)/(f"probe_{args.limit}_w{args.workers}" if args.limit else "full");out.mkdir(parents=True,exist_ok=True);started=time.time();row_dir=out/"rows";row_dir.mkdir(exist_ok=True)
    completed={}
    for path in row_dir.glob("*.npz"):
        with np.load(path,allow_pickle=False) as d:
            index=int(d["index"]);whole=float(d["max_error"]);target_error=float(d["target_alignment_error"]) if "target_alignment_error" in d.files else (0.0 if whole<=1e-5 else float("inf"));completed[index]=(index,np.asarray(d["mask"],dtype=bool),whole,float(d["mean_error"]),int(d["target_points"]),int(d["raw_points"]),target_error)
    pending=[(i,g,stored[i]) for i,g in enumerate(groups) if i not in completed]
    ctx=mp.get_context("spawn")
    with ctx.Pool(args.workers,initializer=init_worker,maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(task,pending):
            completed[result[0]]=result
            np.savez_compressed(row_dir/f"{result[0]:04d}.npz",index=result[0],mask=result[1],max_error=result[2],mean_error=result[3],target_points=result[4],raw_points=result[5],target_alignment_error=result[6])
            atomic(out/"progress.json",{"complete":len(completed),"total":len(groups),"elapsed_seconds":time.time()-started})
    results=list(completed.values())
    results.sort();masks=np.stack([r[1] for r in results]);rows=[{"group_index":int(groups[r[0]]["group_index"]),"max_alignment_error":r[2],"mean_alignment_error":r[3],"target_points":r[4],"raw_object_points":r[5]} for r in results]
    np.savez_compressed(out/"target_masks.npz",target_mask=masks,group_index=np.asarray([g["group_index"] for g in groups]))
    maxerr=max(r[2] for r in results);target_maxerr=max(r[6] for r in results);checks={"rows_exact":len(results)==len(groups),"shape_binary":masks.shape==(len(groups),1024) and bool(np.isin(masks,[False,True]).all()),"all_target_visible":bool(np.all(masks.sum(1)>0)),"exact_target_alignment":target_maxerr<=1e-5,"finite":bool(np.isfinite([r[2] for r in results]).all())}
    summary={"schema_version":1,"run_id":"A6-G040C","status":"passed" if all(checks.values()) else "failed","complete":True,"terminal":True,"workers":args.workers,"groups":len(groups),"elapsed_seconds":time.time()-started,"max_whole_object_alignment_error":maxerr,"max_target_alignment_error":target_maxerr,"target_points":{"min":int(masks.sum(1).min()),"median":float(np.median(masks.sum(1))),"max":int(masks.sum(1).max())},"checks":checks,"rows":rows,"decision":"authorize target-mask conditioned grasp fit" if all(checks.values()) else "repair target-link reproduction"}
    atomic(out/"summary.json",summary);atomic(out/"run_state.json",summary);atomic(out/"queue_state.json",summary);print(json.dumps(summary));return 0 if all(checks.values()) else 2
if __name__=="__main__":raise SystemExit(main())

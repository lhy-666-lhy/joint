#!/usr/bin/env python3
"""Load Stage-1 ckpt, infer on val, write GT vs Pred HTML (Plotly).

Supports dual-head: panels = GT | Combined(prob*value) | Prob | Value.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_train.utils.pc_utils import pc_normalize
from vendor.point_m2ae.Point_M2AE_Afford import Point_M2AE_Afford


def iou_at(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.3) -> float:
    p = (pred >= thresh).astype(np.int32)
    t = (gt >= thresh).astype(np.int32)
    if t.sum() == 0:
        return float("nan")
    return float((p & t).sum()) / float((p | t).sum() + 1e-8)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8
    return float((a * b).sum() / den)


@torch.no_grad()
def predict(model, xyz_norm: np.ndarray, device: torch.device, dual_head: bool):
    pts = torch.from_numpy(xyz_norm.astype(np.float32)).unsqueeze(0).to(device)
    pts = pts.transpose(1, 2).contiguous()
    if dual_head:
        prob, value = model(pts, return_parts=True)
        prob = prob.squeeze(0).detach().cpu().numpy().astype(np.float32)
        value = value.squeeze(0).detach().cpu().numpy().astype(np.float32)
        combined = (prob * value).astype(np.float32)
        return combined, prob, value
    out = model(pts).squeeze(-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out, None, None


def pack_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def sample_indices(gt: np.ndarray, n: int, seed: int, thresh: float = 0.3) -> np.ndarray:
    n_pts = int(gt.shape[0])
    if n_pts <= n:
        return np.arange(n_pts, dtype=np.int64)
    rng = np.random.default_rng(seed)
    hi = np.nonzero(gt >= thresh)[0]
    rest = np.nonzero(gt < thresh)[0]
    take_hi = hi if len(hi) <= n // 2 else rng.choice(hi, size=n // 2, replace=False)
    remain = n - len(take_hi)
    take_rest = (
        rng.choice(rest, size=min(remain, len(rest)), replace=False)
        if len(rest)
        else np.array([], dtype=np.int64)
    )
    return np.concatenate([take_hi, take_rest])


def build_html(records: list[dict], meta: dict) -> str:
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    dual = bool(meta.get("dual_head", False))
    grid = "1fr 1fr 1fr 1fr" if dual else "1fr 1fr"
    extra_panels = ""
    if dual:
        extra_panels = """
  <div class="panel prob"><h2>Pred Prob (sigmoid)</h2><div class="plot" id="prob"></div></div>
  <div class="panel value"><h2>Pred Value (ReLU)</h2><div class="plot" id="value"></div></div>"""
    title = "GT · Combined · Prob · Value" if dual else "左 GT · 右 Pred"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Stage-1 val GT vs Pred</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing:border-box; }}
  html, body {{ height:100%; }}
  body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:#0f1115; color:#e8ecf1;
         display:flex; flex-direction:column; overflow:hidden; }}
  header {{ flex:0 0 auto; padding:10px 14px; border-bottom:1px solid #2a3344;
            display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
  h1 {{ margin:0; font-size:16px; }}
  .meta {{ color:#9aa3b2; font-size:12px; }}
  .controls {{ margin-left:auto; display:flex; gap:8px; align-items:center; }}
  select, button {{ background:#1a1f2a; color:#e8ecf1; border:1px solid #334055; border-radius:8px; padding:6px 10px; }}
  #plots {{ flex:1 1 auto; display:grid; grid-template-columns:{grid}; gap:8px; padding:8px; min-height:0; }}
  .panel {{ display:flex; flex-direction:column; min-height:0; background:#12161f;
            border:1px solid #2a3344; border-radius:10px; overflow:hidden; }}
  .panel h2 {{ margin:0; padding:8px 12px; font-size:12px; font-weight:600; border-bottom:1px solid #2a3344;
               background:#161b26; color:#ffb86b; letter-spacing:0.02em; }}
  .panel.pred h2 {{ color:#7bdff2; }}
  .panel.prob h2 {{ color:#b0f2ae; }}
  .panel.value h2 {{ color:#f2aeb0; }}
  .plot {{ flex:1 1 auto; min-height:0; }}
  #info {{ flex:0 0 auto; padding:8px 14px; color:#9aa3b2; font-size:12px; border-top:1px solid #2a3344; }}
  #err {{ display:none; margin:8px 12px; padding:10px; background:#3a1520; border:1px solid #a33; border-radius:8px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>{title}</h1>
    <div class="meta" id="metaLine"></div>
  </div>
  <div class="controls">
    <label>sample <select id="sel"></select></label>
    <button id="prev" type="button">◀</button>
    <button id="next" type="button">▶</button>
  </div>
</header>
<div id="err"></div>
<div id="plots">
  <div class="panel gt"><h2>GT affordance</h2><div class="plot" id="gt"></div></div>
  <div class="panel pred"><h2>Pred Combined</h2><div class="plot" id="pred"></div></div>
  {extra_panels}
</div>
<div id="info">loading…</div>
<script>
const RECORDS = {payload};
const META = {meta_json};
const DUAL = !!META.dual_head;
const err = document.getElementById('err');
function showErr(msg) {{ err.style.display='block'; err.textContent = msg; console.error(msg); }}

function b64ToBytes(b64) {{
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
  return out;
}}
function decodeF32(b64) {{ return new Float32Array(b64ToBytes(b64).buffer); }}

function unpack(r) {{
  const xyz = decodeF32(r.xyz);
  const n = xyz.length / 3;
  const x = new Array(n), y = new Array(n), z = new Array(n);
  for (let i=0;i<n;i++) {{ x[i]=xyz[i*3]; y[i]=xyz[i*3+1]; z[i]=xyz[i*3+2]; }}
  const gt = Array.from(decodeF32(r.gt));
  const pred = Array.from(decodeF32(r.pred));
  const out = {{x,y,z,gt,pred,n}};
  if (DUAL) {{
    out.prob = Array.from(decodeF32(r.prob));
    out.value = Array.from(decodeF32(r.value));
  }}
  return out;
}}

try {{
  if (typeof Plotly === 'undefined') throw new Error('Plotly CDN failed.');
  document.getElementById('metaLine').textContent =
    `ckpt=${{META.ckpt}} · epoch=${{META.epoch}} · best_iou=${{META.best_iou}} · dual=${{META.dual_head}} · n=${{RECORDS.length}} · mean_iou@${{META.thresh}}=${{META.mean_iou}} · pts=${{META.viz_points}}`;

  const sel = document.getElementById('sel');
  RECORDS.forEach((r,i) => {{
    const opt = document.createElement('option');
    opt.value = String(i);
    const iou = (r.iou==null) ? 'nan' : Number(r.iou).toFixed(3);
    opt.textContent = `${{i+1}}. ${{r.name}} | IoU=${{iou}}`;
    sel.appendChild(opt);
  }});

  function makeLayout() {{
    return {{
      paper_bgcolor:'#12161f', plot_bgcolor:'#12161f',
      font:{{color:'#cdd3de', size:11}},
      margin:{{l:0,r:0,t:4,b:0}},
      uirevision:'keep',
      scene:{{
        aspectmode:'data',
        xaxis:{{showbackground:false, showgrid:true, gridcolor:'#2a3344', zeroline:false, color:'#9aa3b2', title:''}},
        yaxis:{{showbackground:false, showgrid:true, gridcolor:'#2a3344', zeroline:false, color:'#9aa3b2', title:''}},
        zaxis:{{showbackground:false, showgrid:true, gridcolor:'#2a3344', zeroline:false, color:'#9aa3b2', title:''}},
        camera:{{eye:{{x:1.35,y:1.15,z:0.95}}}}
      }}
    }};
  }}

  function makeTrace(x,y,z,score, cmax, showbar) {{
    const cm = Math.max(cmax || 1, 1e-6);
    return {{
      type:'scatter3d', mode:'markers',
      x,y,z,
      hovertemplate:'v=%{{marker.color:.3f}}<extra></extra>',
      marker:{{
        size:2.0,
        color:score,
        colorscale:'Hot',
        cmin:0, cmax:cm,
        opacity:0.95,
        colorbar: showbar ? {{title:'', thickness:10, len:0.5, x:1.02}} : undefined,
        showscale: !!showbar
      }}
    }};
  }}

  let camera = null;
  let syncing = false;
  const panelIds = DUAL ? ['gt','pred','prob','value'] : ['gt','pred'];

  async function show(idx) {{
    const r = RECORDS[idx];
    const u = unpack(r);
    const cfg = {{responsive:true, displayModeBar:false, staticPlot:false}};
    const predMax = Math.max(...u.pred, 1.0);
    const valueMax = DUAL ? Math.max(...u.value, 1e-6) : 1.0;

    const layouts = {{}};
    panelIds.forEach(id => {{
      layouts[id] = makeLayout();
      if (camera) layouts[id].scene.camera = camera;
    }});

    await Plotly.react('gt', [makeTrace(u.x,u.y,u.z,u.gt,1.0,false)], layouts.gt, cfg);
    await Plotly.react('pred', [makeTrace(u.x,u.y,u.z,u.pred,predMax,true)], layouts.pred, cfg);
    if (DUAL) {{
      await Plotly.react('prob', [makeTrace(u.x,u.y,u.z,u.prob,1.0,false)], layouts.prob, cfg);
      await Plotly.react('value', [makeTrace(u.x,u.y,u.z,u.value,valueMax,true)], layouts.value, cfg);
    }}

    panelIds.forEach(srcId => {{
      const src = document.getElementById(srcId);
      src.removeAllListeners && src.removeAllListeners('plotly_relayout');
      src.on('plotly_relayout', (ev) => {{
        if (syncing || !ev['scene.camera']) return;
        syncing = true;
        camera = ev['scene.camera'];
        Promise.all(panelIds.filter(id => id!==srcId).map(id =>
          Plotly.relayout(id, {{'scene.camera': camera}})
        )).finally(() => {{ syncing = false; }});
      }});
    }});

    const iou = (r.iou==null) ? 'nan' : Number(r.iou).toFixed(3);
    let extra = '';
    if (DUAL) {{
      extra = ` · prob_mean=${{Number(r.prob_mean).toFixed(3)}} value_mean=${{Number(r.value_mean).toFixed(3)}} pred_max=${{Number(r.pred_max).toFixed(3)}}`;
    }}
    document.getElementById('info').innerHTML =
      `<b>${{r.name}}</b> · IoU@${{META.thresh}}=<b>${{iou}}</b> · MAE=${{Number(r.mae).toFixed(4)}} · corr=${{Number(r.corr).toFixed(3)}} · gt_pos=${{r.gt_pos}} pred_pos=${{r.pred_pos}}${{extra}} · N=${{u.n}} · replay=${{r.replay_id}}`;
    sel.value = String(idx);
    window.dispatchEvent(new Event('resize'));
  }}

  sel.onchange = () => show(parseInt(sel.value,10));
  document.getElementById('prev').onclick = () => show((parseInt(sel.value,10)-1+RECORDS.length)%RECORDS.length);
  document.getElementById('next').onclick = () => show((parseInt(sel.value,10)+1)%RECORDS.length);
  show(0);
}} catch (e) {{
  showErr(String(e && e.message ? e.message : e));
}}
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", type=Path, default=ROOT / "data" / "joint_door.zarr")
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "stage1_dual" / "best.pth")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "val_gt_vs_pred_stage1_dual.html")
    p.add_argument("--gpu", type=str, default="1")
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--mode", choices=["one_per_obj", "all"], default="one_per_obj")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--viz_points", type=int, default=1024)
    args = p.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    best_iou = ckpt.get("best_iou") if isinstance(ckpt, dict) else None
    dual_head = bool(ckpt.get("dual_head", False)) if isinstance(ckpt, dict) else False
    if not dual_head and isinstance(state, dict):
        dual_head = any(k.startswith("convs4_cls") for k in state)

    model = Point_M2AE_Afford(cls_dim=1, num_categories=16, dual_head=dual_head).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(
        f"loaded {args.ckpt} epoch={epoch} best_iou={best_iou} dual_head={dual_head} device={device}",
        flush=True,
    )

    root = zarr.open(str(args.zarr), mode="r")
    pc = root["data"]["point_cloud"]
    splits = np.asarray(root["meta"]["replay_split"][:])
    keys = [str(k) for k in root["meta"]["replay_obj_keys"][:]]
    val_ids = np.nonzero(splits == 1)[0].tolist()

    selected: list[int] = []
    if args.mode == "one_per_obj":
        seen: set[str] = set()
        for rid in val_ids:
            k = keys[rid]
            if k in seen:
                continue
            seen.add(k)
            selected.append(rid)
    else:
        selected = list(val_ids)
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    records = []
    ious = []
    for rid in tqdm(selected, desc="val infer"):
        cloud = np.asarray(pc[rid], dtype=np.float32)
        xyz_raw = cloud[:, :3]
        gt = np.clip(cloud[:, 3], 0.0, 1.0)
        xyz_norm = pc_normalize(xyz_raw)
        pred, prob, value = predict(model, xyz_norm, device, dual_head)
        iou = iou_at(pred, gt, args.iou_thresh)
        ious.append(iou)
        mae = float(np.abs(pred - gt).mean())
        corr = pearson(pred, gt)

        xyz_c = xyz_norm - xyz_norm.mean(axis=0, keepdims=True)
        idx = sample_indices(gt, args.viz_points, seed=rid, thresh=args.iou_thresh)
        xyz_v = xyz_c[idx]
        gt_v = gt[idx]
        pred_v = pred[idx]

        rec = {
            "name": keys[rid],
            "replay_id": int(rid),
            "xyz": pack_f32(xyz_v.reshape(-1)),
            "gt": pack_f32(gt_v),
            "pred": pack_f32(pred_v),
            "iou": float(iou) if iou == iou else None,
            "mae": round(mae, 6),
            "corr": round(corr, 6),
            "gt_pos": int((gt >= args.iou_thresh).sum()),
            "pred_pos": int((pred >= args.iou_thresh).sum()),
            "pred_max": round(float(pred.max()), 4),
        }
        if dual_head:
            rec["prob"] = pack_f32(prob[idx])
            rec["value"] = pack_f32(value[idx])
            rec["prob_mean"] = round(float(prob.mean()), 4)
            rec["value_mean"] = round(float(value.mean()), 4)
        records.append(rec)

    mean_iou = float(np.nanmean(ious)) if ious else float("nan")
    meta = {
        "ckpt": str(args.ckpt),
        "epoch": epoch,
        "best_iou": None if best_iou is None else float(best_iou),
        "mean_iou": round(mean_iou, 4),
        "thresh": args.iou_thresh,
        "n": len(records),
        "mode": args.mode,
        "viz_points": args.viz_points,
        "dual_head": dual_head,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(records, meta), encoding="utf-8")
    print(f"mean_iou@{args.iou_thresh}={mean_iou:.4f}")
    print(
        f"saved -> {args.out} ({args.out.stat().st_size/1024/1024:.1f} MB, {len(records)} samples)",
        flush=True,
    )


if __name__ == "__main__":
    main()

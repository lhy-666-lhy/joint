#!/usr/bin/env python3
"""Visualize jointTrain zarr train point clouds as interactive HTML (Three.js).

Default: one cloud per unique obj_key in the train split (keeps HTML size manageable).
Color = GT affordance (inferno). Output under jointTrain/data/.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]


def inferno_rgb(t: np.ndarray) -> np.ndarray:
    """Approx matplotlib inferno colormap, t in [0,1] -> uint8 RGB (N,3)."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    stops = np.array(
        [
            [0.001462, 0.000466, 0.013866],
            [0.087369, 0.044956, 0.224813],
            [0.258234, 0.038571, 0.406485],
            [0.416331, 0.090733, 0.434061],
            [0.578304, 0.148039, 0.404411],
            [0.735683, 0.215906, 0.330245],
            [0.865006, 0.316822, 0.226055],
            [0.947266, 0.465046, 0.127568],
            [0.988362, 0.662208, 0.144617],
            [0.987622, 0.891244, 0.313813],
            [0.988362, 0.998364, 0.644924],
        ],
        dtype=np.float64,
    )
    x = np.linspace(0.0, 1.0, len(stops))
    rgb = np.stack(
        [np.interp(t, x, stops[:, c]) for c in range(3)],
        axis=1,
    )
    return (rgb * 255.0).astype(np.uint8)


def pack_f32(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def pack_u8(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def aabb_cbrt(xyz: np.ndarray) -> float:
    extent = np.maximum(xyz.max(0) - xyz.min(0), 0.0)
    return float(np.prod(extent) ** (1.0 / 3.0))


def build_html(records: list[dict], title: str, viewer_height: int = 360) -> str:
    payload = json.dumps(records, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  :root {{
    --bg: #0f1115;
    --panel: #1a1f2a;
    --text: #e8ecf1;
    --muted: #9aa3b2;
    --accent: #ff9f43;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; height: auto; overflow: auto;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text);
  }}
  .wrap {{
    max-width: 720px; margin: 0 auto; padding: 8px;
  }}
  header {{
    padding: 8px 10px; background: #151a24;
    border: 1px solid #2a3344; border-radius: 8px 8px 0 0;
    display:flex; gap:8px; flex-wrap:wrap; align-items:center;
  }}
  h1 {{ margin:0; font-size:14px; font-weight:600; }}
  .meta {{ color: var(--muted); font-size:11px; }}
  .controls {{
    display:flex; gap:6px; flex-wrap:wrap; align-items:center; width:100%;
  }}
  select, button, input[type=range] {{
    background: var(--panel); color: var(--text); border:1px solid #334055;
    border-radius:6px; padding:4px 8px; font-size:12px;
  }}
  select {{ max-width: 100%; flex: 1 1 220px; }}
  button {{ cursor:pointer; }}
  button:hover {{ border-color: var(--accent); }}
  #viewer {{
    width:100%; height:{viewer_height}px; max-height:50vh;
    background: #0f1115; border-left:1px solid #2a3344; border-right:1px solid #2a3344;
  }}
  .bar {{
    padding:6px 10px; display:flex; gap:10px; flex-wrap:wrap; align-items:center;
    border:1px solid #2a3344; border-top:none; border-radius:0 0 8px 8px;
    background:#12161f; font-size:11px; color:var(--muted);
  }}
  .swatch {{
    width:100px; height:8px; border-radius:4px; display:inline-block; vertical-align:middle;
    background: linear-gradient(90deg,
      #040312, #2c115f, #721f81, #b63679, #f1605d, #feb078, #fcfdbf);
  }}
  #info strong {{ color: var(--text); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>{title}</h1>
    <div class="meta">drag rotate · scroll zoom</div>
  </div>
  <div class="controls">
    <select id="sel"></select>
    <label>size <input id="psize" type="range" min="0.5" max="4" step="0.1" value="1.5"/></label>
    <button id="prev">◀</button>
    <button id="next">▶</button>
    <button id="reset">reset</button>
  </div>
</header>
<div id="viewer"></div>
<div class="bar">
  <div>aff <span class="swatch"></span></div>
  <div id="info">loading…</div>
</div>
</div>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

const RECORDS = {payload};

function b64ToF32(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}
function b64ToU8(b64) {{
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}}

const viewer = document.getElementById("viewer");
const sel = document.getElementById("sel");
const info = document.getElementById("info");
const psize = document.getElementById("psize");

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(1.2, 0.9, 1.2);
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewer.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.85));
const grid = new THREE.GridHelper(2, 20, 0x334055, 0x1e2633);
grid.position.y = -0.01;
scene.add(grid);

let pointsObj = null;
let baseSize = 0.012;

function resize() {{
  const w = viewer.clientWidth, h = viewer.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}}
window.addEventListener("resize", resize);
resize();

function fitCamera(xyz) {{
  let minx=Infinity,miny=Infinity,minz=Infinity,maxx=-Infinity,maxy=-Infinity,maxz=-Infinity;
  for (let i=0;i<xyz.length;i+=3) {{
    const x=xyz[i], y=xyz[i+1], z=xyz[i+2];
    if (x<minx) minx=x; if (y<miny) miny=y; if (z<minz) minz=z;
    if (x>maxx) maxx=x; if (y>maxy) maxy=y; if (z>maxz) maxz=z;
  }}
  const cx=(minx+maxx)/2, cy=(miny+maxy)/2, cz=(minz+maxz)/2;
  const dx=maxx-minx, dy=maxy-miny, dz=maxz-minz;
  const radius = 0.5 * Math.max(dx, dy, dz, 1e-3);
  controls.target.set(cx, cy, cz);
  camera.position.set(cx + radius*1.8, cy + radius*1.2, cz + radius*1.8);
  controls.update();
  baseSize = Math.max(0.004, radius * 0.035);
  if (pointsObj) pointsObj.material.size = baseSize * parseFloat(psize.value);
}}

function show(idx) {{
  const rec = RECORDS[idx];
  const xyz = b64ToF32(rec.xyz_b64);
  const rgb = b64ToU8(rec.rgb_b64);
  const n = xyz.length / 3;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(rgb, 3, true));
  const mat = new THREE.PointsMaterial({{
    size: baseSize * parseFloat(psize.value),
    vertexColors: true,
    sizeAttenuation: true,
  }});
  if (pointsObj) {{
    scene.remove(pointsObj);
    pointsObj.geometry.dispose();
    pointsObj.material.dispose();
  }}
  pointsObj = new THREE.Points(geom, mat);
  scene.add(pointsObj);
  fitCamera(xyz);
  info.innerHTML = `<strong>${{rec.name}}</strong> · N=${{n}} · `
    + `aff mean=${{rec.aff_mean.toFixed(3)}} max=${{rec.aff_max.toFixed(3)}} · `
    + `pos@${{rec.pos_thresh.toFixed(3)}}=${{rec.n_pos}} · cbrtV=${{rec.cbrt.toFixed(3)}} · replay=${{rec.replay_id}}`;
  sel.value = String(idx);
}}

RECORDS.forEach((r, i) => {{
  const opt = document.createElement("option");
  opt.value = String(i);
  opt.textContent = `${{i+1}}. ${{r.name}}`;
  sel.appendChild(opt);
}});

sel.addEventListener("change", () => show(parseInt(sel.value, 10)));
document.getElementById("prev").onclick = () => {{
  const i = (parseInt(sel.value,10) - 1 + RECORDS.length) % RECORDS.length;
  show(i);
}};
document.getElementById("next").onclick = () => {{
  const i = (parseInt(sel.value,10) + 1) % RECORDS.length;
  show(i);
}};
document.getElementById("reset").onclick = () => show(parseInt(sel.value,10));
psize.addEventListener("input", () => {{
  if (pointsObj) pointsObj.material.size = baseSize * parseFloat(psize.value);
}});

show(0);
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", type=Path, default=ROOT / "data" / "joint_door.zarr")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "train_pointclouds.html")
    p.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="train",
        help="which replay_split to include (all = train+val)",
    )
    p.add_argument(
        "--mode",
        choices=["one_per_obj", "all"],
        default="one_per_obj",
        help="one_per_obj: first replay per obj_key; all: every selected replay",
    )
    p.add_argument("--max_samples", type=int, default=0, help="0=all selected")
    p.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="if >0, randomly keep this many points per cloud (shrink HTML)",
    )
    p.add_argument("--viewer_height", type=int, default=360, help="3D canvas height in px (compact for IDE preview)")
    p.add_argument(
        "--pos_thresh",
        type=float,
        default=0.3,
        help="count/highlight GT-positive points at this affordance threshold",
    )
    p.add_argument(
        "--color_mode",
        choices=["continuous", "mask"],
        default="continuous",
        help="continuous: full inferno; mask: below thresh=gray, above=inferno",
    )
    args = p.parse_args()

    root = zarr.open(str(args.zarr), mode="r")
    pc = root["data"]["point_cloud"]
    splits = np.asarray(root["meta"]["replay_split"][:])
    keys = [str(k) for k in root["meta"]["replay_obj_keys"][:]]
    if args.split == "train":
        pool_ids = np.nonzero(splits == 0)[0].tolist()
    elif args.split == "val":
        pool_ids = np.nonzero(splits == 1)[0].tolist()
    else:
        pool_ids = list(range(len(splits)))

    selected: list[int] = []
    if args.mode == "one_per_obj":
        seen: set[str] = set()
        for rid in pool_ids:
            k = keys[rid]
            if k in seen:
                continue
            seen.add(k)
            selected.append(rid)
    else:
        selected = list(pool_ids)

    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    rng = np.random.default_rng(0)
    thresh = float(args.pos_thresh)
    records = []
    for rid in selected:
        cloud = np.asarray(pc[rid], dtype=np.float32)
        if args.subsample > 0 and cloud.shape[0] > args.subsample:
            idx = rng.choice(cloud.shape[0], size=int(args.subsample), replace=False)
            cloud = cloud[idx]
        xyz = cloud[:, :3]
        aff = np.clip(cloud[:, 3], 0.0, 1.0)
        rgb = inferno_rgb(aff)
        if args.color_mode == "mask":
            neg = aff < thresh
            rgb[neg] = np.array([48, 52, 64], dtype=np.uint8)
        # center for nicer default view (keep relative geometry)
        xyz_c = xyz - xyz.mean(axis=0, keepdims=True)
        n_pos = int((aff >= thresh).sum())
        records.append(
            {
                "name": f"{keys[rid]} [rid={rid}|{('train' if int(splits[rid])==0 else 'val')}]",
                "replay_id": int(rid),
                "xyz_b64": pack_f32(xyz_c),
                "rgb_b64": pack_u8(rgb),
                "aff_mean": float(aff.mean()),
                "aff_max": float(aff.max()),
                "n_pos": n_pos,
                "pos_thresh": thresh,
                "cbrt": aabb_cbrt(xyz),
            }
        )
        print(
            f"[{len(records)}/{len(selected)}] {keys[rid]} rid={rid} "
            f"aff_max={aff.max():.3f} pos@{thresh:.3f}={n_pos}",
            flush=True,
        )

    title = (
        f"jointTrain PCD ({args.split}/{args.mode}, n={len(records)}"
        + (f", pts≤{args.subsample}" if args.subsample > 0 else "")
        + f", pos≥{thresh:.3f}/{args.color_mode}"
        + ")"
    )
    html = build_html(records, title=title, viewer_height=args.viewer_height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"saved -> {args.out} ({size_mb:.1f} MB, {len(records)} clouds)", flush=True)


if __name__ == "__main__":
    main()

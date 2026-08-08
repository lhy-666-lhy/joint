# 从原始点云与成功/失败抓取位点生成 Affordance GT

本文说明：**如何把原始点云 `(P, 3)` + 抓取成功/失败位点，变成训练用的 affordance 分数，并下采样成 `(N, 4)`（xyz ‖ score）**。

对应实现入口：`scripts/build_zarr_dataset.py` 中的 `build_replay_pcd()`。  
核心算法：`joint_train/affordance/heatmap.py`、`joint_train/affordance/fps.py`  
（当前工作树中该包位于 `data/joint_door.zarr/joint_train/affordance/`）。

---

## 1. 一句话结论

Affordance GT 是**规则生成的稠密热力图**，不是神经网络输出：

1. 以**成功抓取中心**为峰，做最近邻高斯；
2. 用**失败中心**做抑制（减去一部分高斯）；
3. 归一化并截断到 `[0, 1]`；
4. 用 **FPS** 从全分辨率点云抽到固定 `N` 点（默认 **4096**），把对应分数拼成第 4 通道。

---

## 2. 输入是什么？输出是什么？

### 2.1 输入（上游原始数据）

每个 replay 目录下有：

```text
.../base_*/heatmap/heatmap_data.npz
```

本流程实际用到的字段：

| 字段 | 形状 | 含义 |
|------|------|------|
| `points` | `(P, 3)` | 原始点云坐标（全分辨率，常达数万点） |
| `candidate_centers` | `(C, 3)` | 抓取/操作候选位点（成功+失败混在一起） |
| `candidate_success` | `(C,)` bool | 该位点是否成功；缺省时视为全成功 |

示例量级（真实文件）：`P≈6.76e4`，`C≈105`，其中成功约 **32** 个。

> 同文件里可能还有 `scores`、`candidate_centers_world` 等。  
> **建库脚本不直接用已有 `scores`**，而是用 centers + success **按本仓公式重算**，保证与 `heatmap.py` 定义一致。

### 2.2 输出（写入 Zarr）

| 产物 | 形状 | 含义 |
|------|------|------|
| `data/point_cloud[rid]` | `(N, 4)` | `[:, :3]=xyz`，`[:, 3]=affordance ∈ [0,1]` |
| 附带 meta（summary） | — | `sigma`、`success_count`、`aabb_volume_cbrt` 等 |

同一 replay 下多条轨迹**共享**这一份点云；轨迹通过 `meta/episode_replay_ids` 索引到 `rid`。

---

## 3. 文件分工总览

```text
collected_data_door/.../heatmap/heatmap_data.npz
        │  提供 points / centers / success
        ▼
scripts/build_zarr_dataset.py
  └─ build_replay_pcd()          # 编排：读盘 → 去重 → 调算法 → 拼 (N,4) → 返回
        │
        ├─► joint_train/affordance/heatmap.py
        │     resolve_heatmap_sigma()   # 由点云 AABB 体积算 σ
        │     heatmap_scores()          # 成功高斯 − 失败抑制 → [0,1]
        │
        └─► joint_train/affordance/fps.py
              fps_indices()             # 从 P 点采到 N 点的索引
                    │
                    ▼
          pc = concat([xyz[idx], scores[idx]])  → (N, 4)
                    │
                    ▼
          data/joint_door.zarr  data/point_cloud
```

| 文件 | 作用（干什么） | 不管什么 |
|------|----------------|----------|
| `heatmap_data.npz`（上游） | 存原始几何与抓取试探结果 | 不负责本仓最终 GT 公式 |
| `build_zarr_dataset.py` | **流水线编排**与写 Zarr；`build_replay_pcd` 是 affordance 赋值入口 | 不实现高斯细节 |
| `affordance/heatmap.py` | **算 σ、算每点分数**（核心数学） | 不做下采样、不写盘 |
| `affordance/fps.py` | **几何下采样**到固定点数 | 不改分数定义 |
| `affordance/__init__.py` | 导出上述 API，方便 `import` | 无算法 |
| `data/joint_door.zarr` | 持久化训练数据 | — |

---

## 4. 逐步讲解（结合代码）

### Step 0 — 读原始数据（`build_replay_pcd`）

文件：`scripts/build_zarr_dataset.py`

```python
points  = data["points"]              # (P, 3)
centers = data["candidate_centers"]   # (C, 3)
success = data["candidate_success"]   # (C,) bool；若无则全 True
```

校验：无中心、或没有任何成功 → 跳过该 replay（返回 `None`）。

---

### Step 1 — 位点去重与成功聚合（仍在 `build_replay_pcd`）

同一物理位置可能被多次试探。实现：

1. `round(centers, 5)` 得到去重键；
2. `np.unique` 得到唯一中心 `uniq`；
3. 对映射到同一中心的试探做 **OR**：只要有一次成功，该中心记为成功。

得到：

- `uniq`：`(C', 3)` 去重后的位点  
- `succ_u`：`(C',)` 去重后的成功掩码  

若去重后仍无成功中心 → 跳过。

**作用**：避免重复中心把高斯峰“叠胖”，并把“同点有成有败”统一成成功峰（失败抑制仍由其它失败中心贡献）。

---

### Step 2 — 体积自适应带宽 σ（`heatmap.py`）

文件：`joint_train/affordance/heatmap.py`  
函数：`resolve_heatmap_sigma` → `compute_volume_scaled_sigma`

\[
\sigma = \max\bigl(10^{-4},\; \texttt{sigma\_coeff}\cdot V^{1/3}\bigr)
\]

其中 \(V\) 是点云 AABB（轴对齐包围盒）的体积，`sigma_coeff` 默认 **0.04008**。

| 函数 | 作用 |
|------|------|
| `aabb_extent` | 算 xyz 三个方向的边长 |
| `aabb_volume` / `volume_cbrt` | 体积与体积立方根（特征长度） |
| `compute_volume_scaled_sigma` | 得到最终 σ |
| `resolve_heatmap_sigma` | 打包 σ 与调试信息（extent、volume 等） |

**作用**：大门/小把手尺度不同时，热斑宽度随物体尺度自动缩放，而不是死用固定 σ。

---

### Step 3 — 全分辨率 affordance 分数（`heatmap.py`）

文件：`joint_train/affordance/heatmap.py`  
函数：`heatmap_scores(points, centers, success_mask, sigma)`  
内部：`_nearest_gaussian`

对每个点 \(x\in\mathbb{R}^3\)：

1. **成功项**  
   找最近成功中心距离 \(d_+\)，  
   \(N_+(x)=\exp\bigl(-\tfrac12(d_+/\sigma)^2\bigr)\)

2. **失败抑制**（若存在失败中心）  
   同理得 \(N_-(x)\)，  
   \(\mathrm{raw}(x)=N_+(x)-0.35\cdot N_-(x)\)

3. **归一化与截断**  
   \(\mathrm{score}=\mathrm{clip}\bigl(\max(\mathrm{raw}/\max(\mathrm{raw}),\,0),\,0,\,1\bigr)\)

得到 `scores_full`，形状 `(P,)`，与原始 `points` 一一对应。

| 函数 | 作用 |
|------|------|
| `_nearest_gaussian` | 用 KDTree（或暴力）算到最近中心的高斯响应 |
| `heatmap_scores` | 成功峰 − 失败抑制 + 归一化，输出 `[0,1]` |

**直觉**：

- 靠近成功抓取点 → 分数高（“这里适合操作”）  
- 靠近失败点 → 分数被压低（“看起来可抓但不稳”）  
- 远离所有中心 → 分数接近 0  

---

### Step 4 — FPS 下采样（`fps.py`）

文件：`joint_train/affordance/fps.py`  
函数：`fps_indices(points, num_points, seed, device)`

最远点采样（Farthest Point Sampling）：

1. 随机选第 1 个点；
2. 反复选“距已选集合最远”的点；
3. 直到选满 `N`（默认 4096）。

支持 CPU / CUDA（`device="cuda:0"` 时走 `fps_indices_torch`）。

**作用**：把不等长的稠密点云变成**固定长度**输入，便于 batch 训练；尽量保持空间覆盖，而不是随机丢点。

注意：分数是在 **全点** 上算完再按索引取值，**不是**先 FPS 再插值。

---

### Step 5 — 赋值拼成 `(N, 4)`（回到 `build_replay_pcd`）

```python
idx = fps_indices(points, num_points, seed=..., device=...)
xyz = points[idx]          # (N, 3)
scores = scores_full[idx]  # (N,)
pc = concat([xyz, scores[:, None]])  # (N, 4)
```

第 4 通道即为 **GT affordance**，随后 stack 进 Zarr 的 `data/point_cloud`。

---

## 5. 端到端数据形状变化

```text
points            (P, 3)     原始点云
centers/success   (C, *)     抓取试探
        │ 去重
uniq / succ_u     (C', *)
        │ heatmap_scores
scores_full       (P,)       每点 affordance
        │ FPS 索引
xyz, scores       (N, 3), (N,)
        │ concat
point_cloud        (N, 4)     训练用 PCD + GT
```

典型：`P ~ 1e4–1e5` → `N = 4096`。

---

## 6. 关键超参（改哪里）

| 超参 | 默认 | 所在位置 | 影响 |
|------|------|----------|------|
| `sigma_coeff` | `0.04008` | `heatmap.DEFAULT_SIGMA_COEFF` / CLI `--sigma-coeff` | 热斑宽窄 |
| 失败抑制系数 | `0.35` | `heatmap_scores` 内硬编码 | 失败区打压力度 |
| `num_points` | `4096` | `build_zarr_dataset.py --num-points` | 下采样点数 |
| FPS `seed` | `0`（每 replay 偏移） | `build_replay_pcd(..., seed=)` | 采样可复现性 |

改 `heatmap.py` 或上述系数后，必须重新跑：

```bash
python scripts/build_zarr_dataset.py ... --overwrite
```

否则 Zarr 里仍是旧 GT。

---

## 7. 和训练阶段的关系（避免混淆）

| 阶段 | affordance 来源 |
|------|-----------------|
| 本文 / 建库 | 规则：点云 + 成功/失败位点 → GT |
| Stage-1 | 网络学习拟合该 GT |
| Stage-2 `--affordance_source gt` | 直接用 Zarr 第 4 通道 |
| Stage-2 `--affordance_source infer` | 用 Stage-1 网络预测 |

可视化可参考：

- `scripts/visualize_train_pcd_html.py`（看 GT）
- `scripts/visualize_val_gt_pred_html.py`（GT vs 预测）

---

## 8. 最小心智模型（给实现者）

把每个成功抓取位点想成热源，每个失败位点想成“冷却器”：

- 热源把附近点云“点亮”；
- 冷却器按 **0.35** 系数局部降温；
- 整幅热力图拉到 `[0,1]`；
- 再均匀抽 **4096** 个点带上热度，交给后续网络。

**编排在** `build_zarr_dataset.py`，**数学在** `heatmap.py`，**几何采样在** `fps.py`，**原始证据在** `heatmap_data.npz`。

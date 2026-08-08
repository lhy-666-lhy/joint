## 1. 数据预处理

### 1.0 使用命令
```bash
# 从hugging face下载数据
cd /data0/liuhongyu/data/articu_dataset_shards
for i in 00 01 02 03 04; do
  wget -c \
    "https://hf-mirror.com/datasets/Zhixi666/articu_dataset/resolve/main/articu_dataset_shards/shards/articu_dataset.part-000${i}.tar.zst" \
    -O "articu_dataset.part-000${i}.tar.zst"
done

cd /data0/liuhongyu/Data
for i in 00 01 02 03 04; do
  tar -I zstd -xf "/data0/liuhongyu/data/articu_dataset_shards/articu_dataset.part-000${i}.tar.zst"
done

# 一步生成单视角点云+affordance的zarr
mkdir -p data/cam_view_cache
CUDA_VISIBLE_DEVICES=0 python scripts/build_zarr_from_data.py \
  --data-root /data0/liuhongyu/Data/collected_data_offline_fixed_base \
  --pcd-source camera \
  --articu-root /data0/liditao/manipulation/articu_sapien \
  --output data/joint_from_data_cam.zarr \
  --camera-cache-dir data/cam_view_cache \
  --device cuda:0 \
  --overwrite
```

### 1.1 处理流程
构建流程在 jointTrain_new/scripts/build_zarr_from_data.py
1. 枚举有效 replay：必须同时存在 heatmap 和至少一条 trajectory。
2. 读取 heatmap 的 candidate_centers 与 candidate_success；没有成功 grasp center 的 replay 被跳过。
3. 对每条 trajectory 检查 result_json.passed 或 success status。
4. 从 joint_qpos[:, :9] 生成 state/action：
    - 每隔 20 帧取一次；
    - 最多 128 帧，但始终保留终点；
    - state = 9D joint_qpos + 2D grasp onehot；
    - action[t] = joint_qpos[t+1]，最后一步复制自身。
5. 每个 replay 只生成一次点云；其所有 trajectory 通过 episode_replay_ids 指向这份点云。
6. 按 obj_key = shape_id_link_name 划分 train/val，同一对象-连杆不会跨 split。

生成的zarr文件架构：

data/point_cloud      (816, 4096, 4)  xyz + affordance
data/state            (3212287, 11)  qpos + grasp onehot
data/action           (3212287, 9)   next qpos
meta/episode_ends     (39409,)
meta/episode_replay_ids
meta/replay_obj_keys
meta/replay_split

Zarr 的价值在于：压缩、分块、随机索引读取；避免每次训练都解析 39,409 个 NPZ；并让一份 replay 点云被数十条轨迹共享。当前点云本体约 816 x 4096 x 4 x 4B = 53.5MB，很适合缓存。

### 1.2 单视角点云与affordance处理
单视角点云由 jointTrain_new/joint_train/sim/capture_view_pcd.py获取，主要流程：
- 根据 initial_state.json 加载 URDF、对象关节状态、机器人状态和 base pose，用 SAPIEN DemoWorld 进行 20 个 settle physics step保证物体稳定；
- 固定相机：448x448、dist=2.0、phi=pi/5、theta=pi、FOV 35°；
- 优先取 object-only world-frame PCD；少于 256 点时退化到完整场景 PCD；
- 根据成功 grasp center 在该视角的点上生成 volume-scaled Gaussian affordance；
- 最后 FPS 到固定 4096 点。原始相机点数依物体而定，报告中通常约 12 万点。

affordance 的 sigma 为：
sigma = 0.04008 * cbrt(AABB volume)
成功 grasp center 给正 Gaussian，失败 center 以 0.35 倍负项抑制，再归一化到 [0,1]。

## 2. stage1： affordance预测网络预训练

stage1 训练结果已经保存在jointTrain_new/runs目录下，其中jointTrain_new/runs/stage1_from_data_cam_t005/best.pth就是目前的最好ckpt，测试集IoU为0.53

训练命令：
```bash
python scripts/train_stage1.py \
  --zarr data/joint_from_data_cam.zarr \
  --ckpts ckpts/pre-train.pth \
  --out_dir runs/stage1_from_data_cam_t005 \
  --epoch 100 --warmup_epoch 10 --batch_size 32 \
  --learning_rate 4e-4 --gpu 0 --mse_weight 10.0 \
  --iou_gt_thresh 0.05
```

可视化stage1的affordance预测效果：
```bash
# 训练集样例可视化
python scripts/visualize_train_pcd_html.py \
  --zarr data/joint_from_data_cam.zarr \
  --out data/joint_from_data_cam_pos005_mask_light.html \
  --split all --mode one_per_obj \
  --subsample 1024 --viewer_height 320 \
  --pos_thresh 0.05 --color_mode mask

# 测试集样例可视化
python scripts/visualize_val_gt_pred_html.py \
  --zarr data/joint_from_data_cam.zarr \
  --ckpt runs/stage1_from_data_cam_t005/best.pth \
  --out data/val_gt_vs_pred_stage1_cam_t005.html \
  --gpu 0 \
  --iou_thresh 0.05 \
  --mode all \
  --viz_points 1024
```

## 3. stage2: 动作生成训练

小规模测试命令：

```bash
# 冻结 Stage-1 + 用预测 affordance输入（主路径）
python scripts/train_stage2.py \
  --zarr $ZARR \
  --stage1_ckpt $S1 \
  --out_dir runs/stage2_cam_infer \
  --affordance_source infer \
  --gpu 0 \
  --batch_size 64 \
  --num_epochs 1000 \
  --max_train_objects 0 \
  --max_val_episodes 0

# 冻结 Stage-1 + 用 GT affordance输入
python scripts/train_stage2.py \
  --zarr $ZARR \
  --stage1_ckpt $S1 \
  --out_dir runs/stage2_cam_gt \
  --affordance_source gt \
  --gpu 0 \
  --batch_size 64 \
  --num_epochs 1000 \
  --max_train_objects 0 \
  --max_val_episodes 0

# 解冻 Stage-1 联合微调（小 lr）
python scripts/train_stage2.py \
  --zarr $ZARR \
  --stage1_ckpt $S1 \
  --out_dir runs/stage2_cam_unfreeze \
  --affordance_source infer \
  --unfreeze_stage1 \
  --stage1_lr 1e-5 \
  --lr 1e-4 \
  --gpu 0 \
  --batch_size 64 \
  --num_epochs 1000 \
  --max_train_objects 0 \
  --max_val_episodes 0
```

目前的stage2默认是小子集调试：max_train_objects=10、traj_per_object=10
全量训练务必加 --max_train_objects 0，max_val_episodes 0 表示全 val

```bash
# 用多卡跑全量训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_stage2.py \
  --zarr data/joint_from_data_cam.zarr 
  --stage1_ckpt jointTrain_new/runs/stage1_from_data_cam_t005/best.pth 
  --out_dir runs/stage2_cam_infer \
  --affordance_source infer 
  --batch_size 64 
  --max_train_objects 0 
  --max_val_episodes 0
```

## 4. 单视角 RGB 与点云质量检查

`scripts/export_camera_rgb.py` 会按照生成 zarr 时相同的 `initial_state.json` 和固定 SAPIEN 相机参数重新渲染 RGB。它将物体或目标 link 像素过少、原始可见点少于 4096，以及渲染/状态读取失败标为异常；物体或目标 link 触碰图像边界默认仅记录为 framing warning，不会使点云被判异常。

异常 replay 的 RGB 和 zarr 中实际训练使用的 `(4096, 4)` 点云保存到 `abnormal/`；正常 replay 随机保存部分代表样本到 `normal_sample/`。完整检查结果写入 `quality_report.json`。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sapien

python scripts/export_camera_rgb.py \
  --data-root /inspire/hdd/global_user/liditao-253108110078/datasets/partnet-mobility \
  --zarr data/joint_from_data_cam.zarr \
  --output-dir data/camera_rgb_v2 \
  --normal-samples 32

```

`--data-root` 可以传入数据集父目录、`articu_dataset` 或 `collected_data_offline_fixed_base`；脚本会自动解析到包含 `data/single/` 的目录。可用 `--split train` 或 `--split val` 仅检查一个 split，`--max-replays N` 进行快速测试。边界接触默认写入 `warnings`；若需要严格将其判为异常，增加 `--treat-border-as-abnormal`。可用 `--border-pixels 0` 关闭边界检查。

## 5. Target-aware 多视角 Stage-1 数据增广

`scripts/build_multiview_zarr_from_data.py` 构建 flexible 多视角 zarr。它使用对象和 target link 的 AABB 自适应计算相机距离，在 target-facing 侧采样多个候选相机位姿，并仅保留物体完整入框、target link 像素足够且未贴边、以及可见 affordance 正样本足够的视角。

输出 zarr 保留原始的 `data/point_cloud`、`state`、`action` 和 episode metadata，供 Stage-2 始终使用一个确定的主单视角点云；新视角保存为 `data/stage1_aug_point_cloud`，仅供 Stage-1 选择性增广。每个增广 view 继承 source replay 的 object-level train/val split，避免同一 `obj_key` 跨 split。

当前设备已通过 1 replay、2 view 的离屏 SAPIEN 3 冒烟测试。先运行小规模检查预览图：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sapien

python scripts/build_multiview_zarr_from_data.py \
  --data-root /inspire/hdd/global_user/liditao-253108110078/datasets/partnet-mobility \
  --max-replays 3 \
  --views-per-replay 2 \
  --max-view-attempts 20 \
  --output data/joint_from_data_multiview_debug.zarr \
  --overwrite
```

正式构建与 Stage-1 对照训练：

```bash
python scripts/build_multiview_zarr_from_data.py \
  --data-root /inspire/hdd/global_user/liditao-253108110078/datasets/partnet-mobility \
  --views-per-replay 4 \
  --max-view-attempts 40 \
  --output data/joint_from_data_multiview.zarr \
  --overwrite

python scripts/train_stage1.py \
  --zarr data/joint_from_data_multiview.zarr \
  --ckpts ckpts/pre-train.pth \
  --out_dir runs/stage1_multiview_t005 \
  --epoch 100 --warmup_epoch 10 --batch_size 32 \
  --learning_rate 4e-4 --gpu 0 --mse_weight 10.0 \
  --iou_gt_thresh 0.05 \
  --view_mode primary

# 使用主视角 + target-aware 多视角增广训练 Stage-1
python scripts/train_stage1.py \
  --zarr data/joint_from_data_multiview.zarr \
  --ckpts ckpts/pre-train.pth \
  --out_dir runs/stage1_primary_plus_multiview_t005 \
  --epoch 100 --warmup_epoch 10 --batch_size 32 \
  --learning_rate 4e-4 --gpu 0 --mse_weight 10.0 \
  --iou_gt_thresh 0.05 \
  --view_mode combined \
  --val_view_mode primary
```

`--view_mode primary` 只用主单视角，`augmentation` 只用新视角，`combined` 合并二者。`--val_view_mode` 默认 `primary`，保证 IoU 始终在原始主单视角验证集上比较。Stage-2 不需要增加任何参数：对 flexible zarr 使用 `train_stage2.py --zarr data/joint_from_data_multiview.zarr` 时，仍只读取主视角和原始 trajectory 数据。

每次构建会输出少量 RGB preview 到 `data/joint_from_data_multiview_previews/`。若 SAPIEN 无法自动选择离屏设备，可显式传入 `--render-device cuda:0`；构建器会在 renderer 不可用时终止并提示，而不会悄悄生成无效点云。

## 6. 最终 best-view 双标签数据集

`scripts/build_bestview_dual_affordance_zarr.py` 是推荐的正式构建器。每个 replay 默认最多采集 10 个 target-aware 视角，选择 target link 可见性最好的一个作为 `data/point_cloud` 主视角，替换旧固定近景视角并供 Stage-1、Stage-2 共用；其余视角保存为 `data/stage1_aug_point_cloud`。若在 `--max-view-attempts` 次候选中只得到 1--9 个合格视角，该 replay 仍会保留，并在 `.zarr_summary.json` 的 `accepted_views_per_replay` 中记录 shortfall；只有 0 个合格视角才会记为失败。

主视角和附加视角都保存两套标签：`updated` 是 collection/replay 成功结果筛选后的 affordance，`initial` 只使用 `grasp_dataset` 中同一个 target link 的 `contact_pairs` 构造，绝不混合其他 link。主视角保留完整 `state/action/episode_*`，因此可直接用于 Stage-2。

先运行 debug；它随机抽取 10 个 target。合格视角的左右 affordance 拼图保存在 `data/joint_bestview_dual_debug_previews/accepted/`（左 updated、右 initial）。若一个 target 在严格与自动放宽后的相机采样中仍无合格视角，候选 RGB、物体蓝色掩码、目标 link 绿色掩码及每次拒绝原因会保存在 `data/joint_bestview_dual_debug_previews/failed/rXXXX_<obj_key>/`；汇总见同目录的 `debug_report.json`。终端也会输出这一路径的绝对地址。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sapien

python scripts/build_bestview_dual_affordance_zarr.py \
  --mode debug \
  --output data/joint_bestview_dual_debug.zarr \
  --overwrite
```

构建按 replay 多进程并行。默认 `--workers 0` 会读取当前 CPU affinity 中的物理核心数；本机为 64 个物理核心，自动使用 8 个 worker。多 worker 时 FPS 使用 CPU，避免与 SAPIEN renderer 和训练进程争用同一 GPU；`--workers 1` 可用于逐条复现或排查问题，`--workers 8` 可显式固定并行度。

确认 `data/joint_bestview_dual_debug_previews/` 后，再完整构建：

```bash
python scripts/build_bestview_dual_affordance_zarr.py \
  --mode collect \
  --output data/joint_bestview_dual.zarr \
  --overwrite
```

实验开关：

```bash
# Stage-1: 主视角 / 主视角+多视角；updated / initial 标签
python scripts/train_stage1.py --zarr data/joint_bestview_dual.zarr --view_mode primary --label_source updated
python scripts/train_stage1.py --zarr data/joint_bestview_dual.zarr --view_mode combined --label_source updated
python scripts/train_stage1.py --zarr data/joint_bestview_dual.zarr --view_mode primary --label_source initial
python scripts/train_stage1.py --zarr data/joint_bestview_dual.zarr --view_mode combined --label_source initial

# Stage-2: 主视角 / 附加视角轮换 / 主+附加视角轮换；验证固定主视角
python scripts/train_stage2.py --zarr data/joint_bestview_dual.zarr --view_mode primary --val_view_mode primary
python scripts/train_stage2.py --zarr data/joint_bestview_dual.zarr --view_mode augmentation --val_view_mode primary
python scripts/train_stage2.py --zarr data/joint_bestview_dual.zarr --view_mode combined --val_view_mode primary
```

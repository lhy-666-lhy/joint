# Architecture 6

这是 `jointTrain_new` 的一次干净实验重启，目标是从 `force_admittance_collect` 的 affordance、抓取和 operation 轨迹学习可部署的铰链物体操作系统。

## 当前入口

- `EXPERIMENT_PLAN.md`：唯一权威实验计划。
- `EXPERIMENT_TRACKER.md`：运行状态、依赖和 terminal artifact 记录。
- `EXPERIMENT_PLAN_20260805_135511.md`、`EXPERIMENT_TRACKER_20260805_135511.md`：本次 revision 的不可混淆副本。

当前只授权 `A6-D000`。它是一次 CPU-only 的历史证据索引和路径冻结，不是训练或全量 replay。没有启动实验、后台 loop 或 watcher。

## 执行顺序

1. 先以 ZERO affordance 完成 operation-only 的 adapter、固定批量记忆、三种 joint-space decoder 和训练策略比较；split-safe affordance baseline 可独立并行训练。
2. 冻结 operation module 后，先做 GT 接口上限，再以 ZERO affordance 在同一合同下比较 `G-TRAJ`（直接预测完整抓取 qpath）和 `G-QPOSE`（只预测 `q_path_terminal_open`，从当前状态在线规划）。
3. 使用仅由 A5_TRAIN 重训的 affordance checkpoint，对 operation/grasp winner 做 matched `ZERO/PRED/GT` 干预，必要时才进行联合 affordance 训练。旧 split 的 E11 checkpoint 只允许加载兼容性 smoke，禁止正式 forward。
4. 建立 frozen pipeline，再做 action、grasp、affordance 的受控联合训练；只有完整系统候选冻结后才读取一次 `A5_MECH_DEV`，最后才读取 same-target/target-disjoint evaluator。

## 历史代码和结果

Architecture 1-4 的可复用代码、数值和禁止外推范围集中列在计划的 E17-E28。Architecture 5 目录只作为历史数据合同和失败迭代的 artifact 来源，不能恢复其旧 queue、authorization 或“exhausted”结论。计划中所有 `model_architecture*` 路径相对 `jointTrain_new/experiment/`，其余路径约定见计划第0节；新代码的真实路径必须通过 `path_config.py` 解析。

不要使用 `dual-autonomous-experiment-loop`；本目录也不包含自动启动循环的授权。

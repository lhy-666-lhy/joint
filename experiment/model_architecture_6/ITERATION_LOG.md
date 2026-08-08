# Architecture 6 Single-Conversation Iteration Log

**管理方式**：本文件由当前 Codex 对话维护。已停止 `autonomous-experiment-loop`；不再创建跨对话 worker、watcher 或 analysis TUI。

**目标**：在 clean A6 split 上验证 command-delta operation 表示，再依次比较 operation decoder、抓取 trajectory/qpose 两条路线、affordance consumer 和 frozen-vs-joint 系统。

**当前预算**：最多 3 个 focused rounds；每轮 1 个最小主实验，只有结果明确支持下一假设时才扩展。长任务使用 artifact/ETA 检查，不在 Codex turn 内等待完成。

## A6-G040R through G044C target-link mask round

- **Hypothesis**：base-frame model 的 target-disjoint 失败来自缺少目标 link 显式条件；先测试最小 `XYZ+binary target mask` concat encoder，并用同结构 zero-mask control 隔离参数/初始化影响。
- G040C 完成 632 rows，但 64 rows 的 whole-object alignment 失败。诊断确认 target link 本身可 exact 对齐，漂移来自同一物体其他活动 link；whole-object exact gate 不适用于 point-level target membership。
- G040R 只将与 target-link render 距离 `<=1e-5` 的 stored point 标为正；64 rows 串行修复，632 masks 非空/二值，最大 target alignment `9.77e-6`。G041 的 group/hash/split/no-affordance/outcome join 全通过。
- G042 target/zero 两支 50-step 与 6000-step 实现/reload checks 全通过，参数均为 `508744`。Raw-unit G044：target endpoint MAE `0.652161`，zero control `0.611153`，历史 base-only `0.613378`；paired target-minus-zero `+0.041007 [0.010256,0.071644]`，target 仅在 38/101 groups 更好。
- **Decision**：concat/global-max target mask 是 scoped negative；不训练其 trajectory arm、不做 physical rollout。下一轮 G045 将 target link 强制分成独立 masked-pool token，并与同参数 learned-null control matched；只有 qpose target-disjoint CAL 改善才扩展 trajectory。
- G045 dual-pool target/null 6000-step checks 通过。G047 raw-unit endpoint MAE 为 `0.620677/0.624526`，paired target-minus-null `-0.003849 [-0.035152,0.027645]`，target 仍差于历史 base-only `0.613378`。
- **Decision update**：dual pooling 为 no-signal，不训练 trajectory。G048 测试最后一个 target-local shape + base-frame centroid 表示；若仍不能同时超过 null 与历史 base-only，停止 target-mask encoder 调参并转向 K4 proposal supervision/selection。
- G048/G050 target-local target/null endpoint MAE 为 `0.658488/0.626526`，paired `+0.031962 [-0.005784,0.070531]`；target TRAIN loss更低但 CAL更差。三种 mask encoder均未改善历史 `0.613378`。
- **Decision final**：停止 mask encoder 与对应 trajectory/physical扩展。G051 先比较 joint-space terminal qpose 和 target-link-relative SE(3) label可迁移性与现有 IK consumer ceiling；只有表示诊断支持才新增 G-SE3 route。
- G051 2373 pose labels与旋转矩阵检查通过；632/632 slot0 hand pose 可从 current qpos IK，最大 position/rotation error `8.64e-5 m / 6.09e-4 rad`。IK 解相对 teacher qpose L2 median/p90/max `0.620/4.846/7.364 rad`，直接证明 joint-space teacher存在强等价解歧义。
- link-relative SE3 standardized NN 比 qpose更紧凑，但 current link pose不满足最终无 oracle边界；正式 learned route改为 deployable robot-base hand SE3。G052 grouped label contract 后，G053 base-only 与 G054 target-local matched训练，G055以 translation/rotation和IK coverage裁决。
- G055 base-only/target-local translation `0.224/0.234 m`，rotation `1.819/1.782 rad`，3cm/12°率近零；IK group coverage `97/101` 与 `101/101` 只说明预测pose可达。local-z 180° symmetry后rotation仍约 `1.14 rad`，不做physical。
- G058 的 2373 labels 中2252个主要offset轴为hand local-z；mean offset `[-0.005,0.000,-0.080] m`，正交norm median `0.0228 m`，affordance positive coverage `98.1%`。G059测试 point-normal + hinge-axis + TRAIN mean retreat geometry ceiling，之后再决定contact proposal训练。
- G059 fixed deploy rule与oracle sign/mapping的3cm/12°率为 `5.0%/10.5%`，rotation mean `1.444/0.997 rad`；PCA normal+hinge不足以确定grasp orientation，不做physical。
- G060 把GT hand投影到target mask形成K4 contact point，比较GT affordance top1/top4-NMS与centroid覆盖；这一步决定affordance是否适合作为contact proposal接口，而不是global feature。
- G060 CAL：GT affordance top4-NMS 3cm/5cm coverage `57.85%/70.68%`，positive-set oracle `85.71%/92.59%`，centroid 0；affordance作为contact proposal有正向下游上限。
- A010 v1 reload failure来自Point-M2AE随机grouping；同模型不reset seed会变，reset相同seed后checkpoint reload严格0且state exact。A010 v2 fixed8 loss `2.991->0.0595`、MAE `0.0427->0.00155`，通过。
- A020 resource pilot冻结microbatch48 x accum2=effective96、workers4；三seed `20260806/07/08` 已在tmux `a6_a020c` 的GPU1/2/3运行，实测预计约2.2h。训练只读clean TRAIN membership，A030等待三个fixed last terminal。
- A020恢复审计：三seed均已跑满7000 steps并terminal pass；上次会话中断发生在训练完成后的汇总阶段，不重跑训练。
- A030 round 1：预注册三个fixed last逐点等权平均，先8-row sanity，再以真实clean CAL label评测点指标与predicted top4-NMS contact coverage；只有group-paired distance相对centroid的95% CI支持改善才进入G061 contact-query规划。
- A030 terminal：实现gate初次因误把G041 base-frame XYZ与Zarr world XYZ逐值比较而失败；按保存base pose重算world-to-base后最大误差`1.12e-7`，同一科学配置通过。ensemble IoU/AUPRC `0.4598/0.6509`，PRED top4-NMS 3cm/5cm contact coverage `47.6%/51.6%`，相对centroid paired distance `-0.2249 m [-0.2602,-0.1873]`，进入G061。
- G061 round 2：先物化全部groups的frozen PRED K4 queries并做Hungarian contact/SE3监督重排；合同与paired contact distance通过后才训练单一contact-conditioned SE3 head。
- G061 terminal：2373 teacher均一一分配；CAL assigned query 3cm/5cm coverage `25.7%/34.6%`，相对centroid paired distance `-0.1701 m [-0.2062,-0.1327]`，解锁G062。
- G062 round 3：只训练一个contact-conditioned base-SE3 head；在原G052 CAL set上用G055同一metric逐group比较G053，translation与rotation均不退化才允许IK/physical。
- G062 terminal：6000-step loss `2.462->0.467`、reload 0；CAL translation/rotation `0.2212m/1.673rad`，相对base-only paired差为`-0.00268m [-0.02813,0.02580]`和`-0.1455rad [-0.2922,0.0018]`，均未过gate。标记shared contact-conditioned SE3 scoped negative，不做IK/physical或事后sweep；A030/G061的contact proposal正信号保留。

## Latest Protocol Correction: A6-O192C fresh-world isolation

- **状态**：IMPLEMENTED / READY；覆盖下方 O171--O191 的旧跨 arm live decision。
- 根因：`run_a6_o126c_zero_contact_live_probe.py` 曾对一个 target 只创建一次 `ViewPcdCapturer/DemoWorld`，随后按 arm 仅写回 qpos/qvel。O185 checkpoint 在 O188 排第三得到 `3/8`，在 O191 排第一得到 `5/8`，target 1/2 progress 出现约 `0.049 -> 0.401`、`0.284 -> 0.401` 的执行位置依赖。
- 修复：每个 arm 独立创建 world，重新设置 drive/friction/target joint 和完整 start state，arm 后立即 close；artifact 记录 `world_reset_mode=independent_world_per_arm`、creation index 与 configured/closed 标记。协议回归测试通过。
- 证据处理：D180、O181/O182/O185/O189 training 与 O183/O186/O190 CAL 保留；旧 live run 只保留首 arm 原始事实，所有第二及后续 arm 和 paired 结论撤销。
- 下一步：O192C 在 target 1/2 用同一 O185 checkpoint first/last 做 order audit；通过后 O193C fresh-world 重跑 O127、seed1、seed2、repeat-last 的 8-target fixed-budget 比较。无任意绝对性能 gate。

### A6-O192C fresh-world order audit

- **状态**：DONE；`results/a6_o192c_fresh_world_order_audit_v1/summary.json`
- target 1/2 均使用四个独立 world；同一 O185 residual checkpoint 放在 first/last。协议 checks 全部通过。
- 两个 target 的 first/last 均 exact trace match：target 1 为 496 calls、final progress `0.401031662`；target 2 为 524 calls、`0.401137947`；final progress/contact/calls/termination 与逐 call progress 差异全为 0。
- **Decision**：world-order bug 已修复；放行 O193C，不把该协议通过误解为 residual 性能提升。

### A6-O193C fresh-world fixed-budget result

- **状态**：DONE；`results/a6_o193c_fresh_world_fixed_budget_live8_v1/summary.json`
- 32/32 rows 的 independent-world、fixed budget、correct start、zero-oracle、opening-stop checks 全通过。
- O127 / random residual seed1 / seed2 / repeat-last 的 mean progress 为 `0.24232 / 0.31365 / 0.28965 / 0.00069`，task success 为 `2/8 / 5/8 / 3/8 / 0/8`。
- seed1/seed2 相对 O127 paired progress 为 `+0.07133 [-0.01491,0.18111]` 与 `+0.04733 [-0.00608,0.10978]`；两支 aggregate contact 均值未下降。O127-vs-repeat progress CI 完全大于 0，确认真实拟合信号。
- **Decision**：residual 继续，但不挑 seed1 直接冻结。O194C 只把有放回随机 sampler 改为逐 epoch 无放回均衡覆盖，训练两个 seed；随后 O195/O196 判断稳定性。

### A6-O194C through O196C equal-exposure sampling

- 两个 O194C seed 的 prefix exposure 为 187/188、recovery exposure 严格 1500/row；parity/freeze/reload checks 全通过。O195C balanced/random CAL 差异约 `1e-6`。
- O196C valid fresh-world：balanced seed1/2 mean progress `0.27639/0.26134`、task 均 `3/8`；random seed1 为 `0.31365 / 5/8`。balanced-minus-random progress `-0.03726/-0.05231`，task 均 `-2`。
- 随机 seed1-vs-seed2 exposure 差与 TRAIN live call 的相关为 `-0.018`，没有晚期偏置证据，不授权 late-state weighting。
- **Decision**：equal-exposure sampler scoped negative，停止该调参。最后测试一次数学可解释且零额外推理成本的 50/50 residual-head weight average（O197--O199）。

### A6-O197C through O199C residual-head weight average

- O197C 证明两个 source 的 frozen baseline 完全相同，只平均 linear head；CPU fixed-batch output-mean parity `5.07e-7`、reload 0。首次 GPU parity 的 `2.98e-4` 来自 TF32 运算顺序，已修正为 CPU algebraic check，不属于模型失败。
- O198C consumer/finite checks 通过，average CAL 位于两个 seed 之间。
- O199C valid fresh-world：average mean progress `0.26895`、task `4/8`、contact `0.59112`。相对 seed1 progress `-0.04470 [-0.13212,0.00315]`、task `-1`；相对 seed2 progress `-0.02070 [-0.04660,-0.00200]`、task `+1`。
- **Decision**：weight average 不选择，停止 residual sampler/seed fusion。O185 seed1 作为 A5_CAL live8 provisional operation 候选（`5/8`），O127 为 fallback；下一主线进入 G000C 抓取 trajectory-vs-qpose 合同。

## Previous Protocol Correction: A6-R170C

- **状态**：DONE；覆盖下方所有旧 live decision。
- 旧 producer 把 `operation_start_joint_command_qpos` 写成实际 robot qpos，旧 live runner 又把它当 last command。当前 35 个 CAL source 中该 bad field 与 robot qpos 完全相同，而 `logged_operation_start_joint_command_qpos` 与 operation 首行 raw command 完全相同；bad finger 约 `0.004--0.017`，正确 command finger 为 0。
- O131、O135/O136、O140/O141-live、O146/O147-live、O148、O149、O153-live、O156-live 的科学结论均标记 `INVALID_LIVE_PROTOCOL`。D160--O163 因错误 last command 进入 recovery label/training，整条链标记 invalid。
- O127/O128 checkpoints、D040/D042 offline、O129 CAL、O151/O154 train 与 O152/O155 CAL 保留。下一步只能是 R170C audit -> O171C 3-call sanity -> corrected live8；G000C 暂停。

### A6-R170C / O171C corrected result

- R170C 对 8 个 CAL live target 和 16 个 TRAIN recovery target 共 24 条 source 完成审计；所有 selected command 与 logged/raw/repaired 完全一致，finger 为 0，旧字段逐条等于 robot qpos。
- O171C corrected live8 所有 lineage/oracle/finite 检查通过。MLP mean progress `0.22942`，8/8 正向、task `2/8`；repeat-last mean `0.00034`、task `0/8`。MLP-minus-repeat paired progress `+0.22908 [0.13684,0.32046]`，证明明确拟合信号。
- Parallel mean progress `0.17909`，但 task `0/8`；MLP-minus-parallel CI 跨 0，暂以 task success 将 MLP 作为优先候选，不作最终 decoder 宣称。
- MLP contact 相对 repeat-last 为 `-0.17515 [-0.37517,-0.00949]`。下一步先 corrected 重评已有 additive/perturb/geometry checkpoint（O172C），不立即增加训练预算。

### A6-O172C corrected checkpoint screen

- 6-arm、8-target corrected rollout 全部通过 lineage/budget/oracle checks。聚合初版曾因 repeat-last outcome 漂移 `4.86e-5` 被错误的 `1e-7` gate 标记失败；已改为只以输入/预算/lineage 作协议 gate，outcome repeatability 仅记录诊断，未重跑 rollout。
- Additive coverage 相对 O127 progress `-0.04463 [-0.09986,0.00605]`、contact `-0.21450 [-0.42248,-0.03431]`，task `1/8 vs 2/8`；不继续该数据扩展。
- Perturb 1x/3x 的 progress/contact CI 均跨 0，task 都是 `2/8`；不继续 common-offset augmentation。
- Geometry residual 相对 baseline progress `+0.00367 [-0.00370,0.01203]`、contact `-0.00035 [-0.04807,0.03619]`，task 相同；未证明收益，保留更简单 O127。
- Trace attribution：接触丢失发生在 call 41/44/74/119，另有平台约在 call 325；旧 D160 仅采前 4 calls，覆盖设计无效。下一轮 D180C 用 16 TRAIN full-horizon rollout、每 target 8 个 log-spaced state，并比较 time/progress-aligned teacher。

### A6-D180C through O184C full-horizon recovery

- D180C 在 148 s 内跑完 16 TRAIN targets / 4798 policy calls，保存 128 rows；112 行 progress anchor 不同于 time anchor，中位绝对偏移 50.5 raw steps，最大 5135。所有 correct-command、TRAIN-only、prefix exact、oracle-boundary checks 通过。
- O181/O182 matched 训练有效，但 recovery raw MAE 仍为 `0.0391/0.0475`，prefix MAE 也升至 `0.00107/0.00131`。O183 CAL 相对 baseline 分别退化 `+0.00120` 和 `+0.00944`。
- O184 time-aligned mean progress `0.25649` vs baseline `0.22942`，paired `+0.02707 [-0.01796,0.10855]`、task 都为 `2/8`；contact paired `-0.19254 [-0.37394,-0.05324]`。它将 target 1 从 `0.010` 提到 `0.314`，但损伤其余 contact。
- Progress-aligned paired progress `-0.13035 [-0.26589,-0.01755]`，task `-1`；只用 object progress 匹配 canonical teacher 会忽略机器人构型兼容性，停止该分支。
- 下一步 O185C 冻结 O127、加 zero-init linear recovery residual，以 prefix distillation 约束基础行为；不做 arbitrary weight sweep。

### A6-O185C through O187C recovery residual

- O185C step0 output parity/reload error 都为 0，O127 参数全部冻结，只训练 zero-init linear residual。O186 CAL residual 相比 uniform recovery 改善 `-0.00063`，但仍比 baseline 差 `+0.00057`。
- O187 residual mean progress `0.25850` vs baseline `0.22942`，paired `+0.02909 [-0.00923,0.07562]`，task 均 `2/8`；contact paired `-0.03390` 且 CI 跨 0。相对 uniform recovery，progress 等价而 contact `+0.15864 [0.01085,0.34228]`。
- Residual isolation 是比全模型 recovery 更好的 tradeoff，但尚无 task-success 提升。下一步不训练，先删除 source-horizon oracle：O188 所有 target 使用统一 650-call/5200-step cap，opening angle only stop。

### A6-O188C fixed-budget deployment screen

- 所有 target/arm 使用统一 650 calls / 5200 physics-step cap；只用 opening angle early stop，不读取 demonstration horizon。全部 8x4 rollout 与 aggregate checks 通过。
- O127 task `2/8`、mean progress `0.24232`；uniform recovery `2/8`、`0.26635`；residual `3/8`、`0.25937`。Residual 新增 target 5 success，90 calls 到达 0.4。
- Residual-minus-O127 progress `+0.01705 [-0.02838,0.07705]`、task `+1`、contact `-0.08260` 且 CI 跨 0。该信号 promising 但来自单 seed；下一步只复现 residual sampling seed，不改变架构/data/loss/budget。

## Round 0: Grounding

- **状态**：DONE
- **结论**：A6-R100、A6-A000C、A6-R110、A6-D020C 已通过；O200F 的 command-delta fixed64 是正向拟合信号；O201F 不能作为科学负结论；dirty IDs `70/251/349` 已从 A6 派生数据排除。
- **计划入口**：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)、[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md)
- **当前基础 artifact**：`results/a6_d020_clean_sample_normalizer_v1/`、`results/a6_r100_data_clean_contract_v1/`、`results/a6_r110_gate_reinterpretation_v1/`

## Round 1: Clean Command-Delta Normalizer

- **状态**：DONE
- **Hypothesis**：使用 clean A5_TRAIN 的合法 anchor/horizon 统计独立 command-delta normalizer，可以消除 absolute-action std 对表示的错误缩放，并保持 absolute reconstruction/parity。
- **Run**：`A6-D021C` / `a6_d021c_command_delta_normalizer_v1`
- **Code**：`run_a6_d021c_command_delta_normalizer.py`
- **Data**：只读 D020C `sample_index.jsonl` 和 clean TRAIN trajectory；禁止读取 CAL、MECH_DEV、SAME_TEST、TARGET_TEST、outcome。
- **Validation**：finite mean/std；轨迹计数与 D020C exact；fixed64 shape；delta/absolute roundtrip max error；forbidden-field audit；atomic JSON summaries。
- **Decision**：D021C 通过，进入 DYN64 operation decoder screen；D030C companion audit 也已通过。
- **Expected output**：`results/a6_d021c_command_delta_normalizer_v1/summary.json`、`normalizer.json`、`fixed_batch_command_delta.npz`。
- **Supervisor**：`supervisor_config_d021c.json`；监控状态写入 `monitor_state_d021c.json` 和 `NEXT_CHECK_D021C.md`。

### Companion audit: A6-D030C

- **状态**：DONE
- **Result**：`results/a6_d030c_clean_dyn64_hash_audit_v1/summary.json`
- **Decision**：64 targets、1024 frames、source/output hashes、clean membership、finite 和 held-out boundary 全部通过；无需重新 materialize。

## Round 2: DYN64 Operation Decoder Screen

- **状态**：IN_PROGRESS
- **Candidates**：MLP、parallel-query、causal，全部 command-delta，matched input/seed/budget/checkpoint rule。
- **Decision**：用 same-sample baseline-relative MAE、endpoint/per-horizon error、replay and operation-only metrics；只做 paired evidence，不使用任意 absolute improvement gate。首个实施对象为 matched MLP command-delta sanity。

### A6-D040C / A6-O100C completed

- D040C first ran a 2-target probe and then all 64 targets (1024 rows). Every input is finite, shape-checked, clean TRAIN-only, and delta reconstruction max error is `7.45e-9`; forbidden-field audit is clean.
- O100C evaluated the frozen 6k command-delta MLP checkpoint on those 1024 rows. Raw absolute MAE is `0.010396` versus repeat-last `0.010511` (relative ratio `0.9890`); endpoint MAE is `0.020174`. Checkpoint reload and all input checks passed.
- Interpretation: this is a valid, weak offline generalization result, not a contract failure. It does not justify selecting MLP or stopping the operation screen. Implement matched O110C/O120C next.

### A6-O110C / A6-O120C completed

- O110C parallel-query: fixed64 normalized delta MAE `0.03013`; DYN64 raw MAE `0.008267`, repeat-last ratio `0.7865`.
- O120C causal: teacher-forced DYN64 raw MAE `0.001237` (ratio `0.1177`), but deployable autoregressive raw MAE `0.010858` (ratio `1.0330`). The gap is a training/inference exposure issue; teacher-forced metrics are diagnostic only.
- All three arms used seed `20260805`, 6,000 optimizer steps, the same D040C inputs and D021C normalizer. Parallel-query is the only arm with a clear deployable offline improvement. Next run is O130C predicted-command replay, not another contract audit.

### A6-O130C probe stopped before full screen

- The 2-target x 3-arm probe executed each model's first 32 predicted commands from the operation start. SAPIEN replay, contact telemetry, and artifact writing all passed; each rollout retained contact but opened only about 10% because the operation itself requires substantially more than 32 steps.
- This is a protocol limitation, not a model negative. The probe is terminal infrastructure evidence only (`scientific_screen_authorized=false`); do not run the 64-target short-horizon version.
- The correct protocol is the Architecture 2/3 live operation evaluator: initialize operation state once, predict a short chunk, execute its fixed prefix, recapture current SAPIEN observation/state/context, and repeat until metadata opening stop or max policy calls. `replay_action` remains valid for GT/full-action parity, but is not the learned-chunk execution loop. The single 32-step O130 probe is infrastructure evidence only.

### A6-R120C planning/implementation audit

- O100C is invalid implementation evidence: its reused O200F checkpoint was trained with the D020 absolute-action normalizer, while O100C evaluation used the D021C command-delta normalizer.
- O110C/O120C are valid finite fixed-fit runs but only diagnostic: D040C contains A5_TRAIN rows, overlaps the fixed64 anchors, and builds part of context from recorded trajectory `contact_feedback`. They are not CAL generalization or deployment results.
- Before any O130 live rollout, build A6-D041C from clean A5_CAL and explicitly separate `recorded_current_observation` from `live_sapiens_observation`. Retrain the MLP with D021C from scratch; do not reuse O200F.

### A6-O100RC corrected sanity

- Re-trained MLP from scratch with D021C command-delta normalizer. Fixed64 normalized MAE `0.012657`, raw delta MAE `0.0002923`, repeat-last raw delta MAE `0.0090673`, reload error `0`.
- This repairs the representation mismatch but remains a fixed-batch implementation check. It does not authorize DYN64/CAL or live claims.

### A6-D041C / O121C and R121C semantic audit

- D041C materialized 35 target-disjoint A5_CAL targets and 280 rows; split, point cloud, state, label, finite, and overlap checks pass.
- O121C paired statistics ran, but R121C found the 33D collection `contact_feedback` contains object progress/open ratio and controller-internal telemetry. D040/D041 also use raw anchors against an operation-relative log.
- Therefore D041C point/state/label evidence remains valid, while its context is invalid. O100RC/O110C/O120C/O121C model metrics are invalid input-context evidence and must not be used for architecture selection.
- Next: D042C zeros context dimensions 0:34 and preserves only 9D task metadata, then retrain all three arms from scratch.

### A6-D042C through O136C: deployable closed loop and mixed-intervention correction

- D042C zeroed context dimensions 0:34 while preserving task metadata, point/state/label arrays. TRAIN1024 MLP/PAR both beat repeat-last on frozen A5_CAL and in source-horizon live8; O131C mean progress was `0.1689`/`0.1664`, with `2/8`/`1/8` task success.
- D043C built 194 targets x 8 anchors. O134C offline CAL improved to MLP `0.002384` and PAR `0.002663`, but O136C live mean progress was `0.1562`/`0.1096`; MLP-vs-TRAIN1024 CI crossed zero and PAR significantly regressed in progress/contact.
- Intervention audit found D043C was not pure target expansion: D042C used 16 anchors/target, D043C used 8, and only 22/64 overlapping targets retained the same trajectory. O132C-O136C remain valid results for that deployable data recipe, but are diagnostic-only for target-coverage causality.
- Decision: D044C must preserve the complete D042C 1024-row prefix and append only the 130 new-target rows from D043C. Retrain matched MLP/PAR and repeat the unchanged CAL/live8 comparison before considering exposure-robustness training.

### A6-D044C through O141C: corrected additive target coverage

- D044C passed with the exact D042C 1024-row prefix plus 1040 rows from 130 disjoint new TRAIN targets. All arrays in the prefix are byte-value identical; total is 194 targets/2064 rows with zero CAL overlap and zero contact context.
- O139C additive checkpoints significantly improve frozen target-paired CAL error over TRAIN1024: MLP delta `-0.001416 [-0.001826,-0.001038]`, PAR delta `-0.000693 [-0.000970,-0.000417]`.
- O141C live8 did not improve: MLP mean progress `0.1346` and PAR `0.1140`; additive-minus-TRAIN1024 CIs cross zero for both. MLP/PAR remain above repeat-last, but task success falls from `2/8`/`1/8` to `1/8`/`0/8`.
- Decision: adding target diversity improves offline interpolation/generalization but does not solve closed-loop exposure/contact loss. Keep O127C/O128C as the current live baseline. Run O142C horizon attribution before one focused exposure-robustness training change.

### A6-O142C horizon attribution

- Additive MLP prefix8 raw MAE is `0.000630` versus TRAIN1024 `0.001011`; first-difference MAE is `0.000342` versus `0.000433`. Its target-paired prefix improvement CI is fully below zero.
- Since the commands actually executed by the live loop are already more accurate offline, reweighting the first 8 steps is not supported as the next intervention. The remaining evidence points to state distribution shift and contact loss after self-generated actions.
- Decision: return to the O127C TRAIN1024 live baseline and test only consistent 7D arm state/command perturbation at TRAIN-derived `1x` and `3x` scales. Preserve absolute action labels exactly and compare both variants within the same live simulation protocol.

### A6-O143C through O147C: consistent perturbation scoped negative

- Both 1x/3x runs preserve absolute action targets to `2.38e-7`, keep qvel/finger unchanged, and complete matched 6k-step TRAIN1024 fits. O145C clean and shifted CAL comparisons are statistically indistinguishable from O127C.
- O146C exactly reproduces the O131C baseline per-target progress (`baseline_repeatability` CI `[0,0]`). Perturb 1x mean progress is `0.17264` versus baseline `0.16888`; paired delta `+0.00376 [-0.00650,0.01925]`. Contact delta also crosses zero and task success remains `2/8`.
- Perturb 3x mean progress is `0.15903`, includes one wrong-way target, and provides no paired benefit. This rules out simply enlarging the same common-offset augmentation.
- Decision: retain O127C as the current operation policy. The next cheapest causal test is replanning frequency: K4 first, then K2 only if K4 signals improvement, with total physics steps held fixed. If frequency fails, move to deployable kinematics-aware state or TRAIN-only simulator recovery supervision.

### A6-O148C: feedback-frequency K4

- K4 and K8 used the same frozen O127C MLP, same eight CAL targets, same stop rule, and identical maximum physics steps per target. K8 reproduced O131C exactly.
- K4 mean progress was `0.17018` versus K8 `0.16888`; paired progress delta `+0.00130 [-0.00581,0.01077]`, with task success unchanged at `2/8`.
- K4 contact fraction was `0.6995` versus K8 `0.6125`; paired contact delta `+0.0870 [0.0238,0.1754]`. Observation time increased from `72.4 s` to `135.5 s` across the eight MLP rollouts, as expected from doubled policy calls.
- Decision: K4 is not a task-success improvement, but its contact-retention signal justifies the preplanned K2 test. If K2 does not convert retention into progress, stop frequency optimization.

### A6-O149C: feedback-frequency K2

- K2 retained the same maximum physics-step budget and passed all protocol/oracle checks. Mean progress was `0.16095`, task success `1/8`, versus K8 `0.16888` and `2/8`.
- K2 minus K8 progress was `-0.00793 [-0.03965,0.01815]`; contact delta was `+0.0645 [-0.0930,0.2156]`. The K4 contact improvement did not grow into a reliable K2 effect, and higher observation cost was substantial.
- Decision: stop replanning-frequency optimization. Keep K8 as the default operation policy because K4 has no task/progress gain and K2 regresses task success. The next operation revision must target state representation or recovery supervision, not call frequency.

## Round 4: Deployable FK/Visible-Target Relative State

- **状态**：IN_PROGRESS
- **Hypothesis**：当前 PointNet scene feature 与 81D robot state 只在 late fusion 相遇，模型需要自己近似 FK 和跨模态相对几何；显式提供 `hand_xyz - visible_target_centroid_xyz` 与 target visible fraction，可能降低自执行后接触丢失。
- **Single variable**：O127C 的 state width 从 81 改为 85；模型、command-delta、数据行、seed、6000 steps、batch64、ZERO affordance、K8 live loop 和 evaluator 全部不变。
- **Deployment boundary**：机器人 hand position 来自当前 qpos FK；target centroid/fraction 来自当前 live point cloud + target mask。禁止读取 object qpos/progress，opening angle 仍只给 evaluator stop/metric。
- **Runs**：D150C input contract -> O151C matched train -> O152C frozen CAL -> O153C source-horizon live8。
- **Decision**：不设 absolute gate；paired progress/contact/task 支持后才增加 seed，否则保留 O127C 并转向 TRAIN-only simulator recovery supervision。

### A6-D150C

- 1024 TRAIN / 280 CAL rows，state shape 85；旧 81D prefix exact、zero-contact exact、TRAIN/CAL target disjoint、feature normalizer TRAIN-only、forbidden object-progress read 为零。
- TRAIN feature mean `[-0.18445,-0.04629,0.06364,0.24919]`，std `[0.09588,0.23268,0.20586,0.16856]`；21 个 TRAIN row 的 target mask 为空，按合同退回全物体点云质心并记录 visible fraction 0；CAL 无空 mask。
- Decision: authorize O151C matched training.

### A6-O151C through O153C: naive concat is confounded

- O151C completed 6000 steps in 34.1 s, raw TRAIN MAE `0.0008892`, reload error `0`. O152C geometry-minus-baseline target-paired CAL MAE was `-0.000053 [-0.000264,0.000153]`; zeroing the geometry channels changed output error, so the consumer used them.
- O153C baseline exactly reproduced O131C. Naive geometry achieved `1/8` task success versus baseline `2/8`; paired progress delta was `-0.02371 [-0.10559,0.03503]`, contact delta `-0.0110 [-0.2127,0.1568]`.
- Isolation audit found the 85D append changed `LayerNorm(81)` to `LayerNorm(85)` and changed downstream initialization. This result is valid only for naive concat and cannot reject geometry information itself.
- Decision: run O154C with the original 81D encoder untouched and a separate zero-init `Linear(4,256)` residual. Require common initial weights and step-0 output parity before training; then reuse unchanged CAL/live8 as O155C/O156C.

### A6-O154C through O156C: corrected geometry residual

- O154C passed common initialization/output parity, kept the 81D state encoder, and learned a nonzero geometry residual. TRAIN raw MAE was `0.0008712` versus O127C `0.0008735`.
- O155C CAL geometry-minus-baseline target-paired MAE was `-0.000054 [-0.000124,0.000014]`; offline improvement remains inconclusive.
- O156C baseline reproduced O131C exactly. Residual mean progress `0.17118` vs `0.16888`, paired delta `+0.00231 [-0.00914,0.01336]`; contact delta `+0.07852 [-0.00252,0.19441]`; task success equal `2/8`; one residual target went wrong-way.
- Decision: do not promote geometry or run a second seed. The contact signal without progress motivates TRAIN-only simulator recovery supervision; keep O127C K8 as default.

## Round 5: TRAIN-only Simulator Recovery Supervision

- **状态**：DONE / scoped negative recipe
- **Hypothesis**：live degradation is caused by states after the policy's own commands that are absent from teacher-forced TRAIN rows. Add a small, explicitly TRAIN-only set of live states generated by frozen O127C and label them with the corresponding canonical future command chunk.
- **Data contract**：16 clean A5_TRAIN targets, 4 K8 calls each, 64 recovery rows; D042C 1024-row prefix exact; no CAL/MECH_DEV/final reads; no object qpos/progress/contact in model input.
- **Runs**：D160C generation -> O161C matched 1088-row MLP -> O162C CAL -> O163C live8.
- **Decision**：paired live evidence only; no fixed improvement gate, no test-set teacher labels, no stop-rule change.

### A6-D160C through O163C: unweighted recovery recipe

- D160C generated 16 TRAIN targets x 4 O127C live calls. All 64 rows are TRAIN-only, zero-contact, oracle-free at model input, and the D042C 1024-row prefix remains exact in the 1088-row additive dataset. Arm-state deviation mean/median/max was `0.0326/0.0251/0.1134`.
- O161C fit was finite/reload-exact, but recovery64 raw MAE remained `0.00404` versus prefix1024 `0.00123`; canonical recovery deltas are substantially larger under the D021C teacher-state normalizer.
- O162C CAL regressed: recovery-minus-baseline target-paired MAE `+0.000658 [+0.000381,+0.000968]`, endpoint `+0.001185 [+0.000675,+0.001715]`.
- O163C improved mean progress `0.16888 -> 0.18791` and contact `0.61250 -> 0.76565`, but paired CIs crossed zero and task success fell `2/8 -> 1/8`. It helps the formerly stuck target 3 while damaging a baseline-success target.
- Decision: reject only the fixed-time, unweighted canonical-label recipe. Keep O127C K8 as the frozen operation baseline and move to A6-G000C grasp-route comparison; do not sweep recovery weights/clips without a new revision.

## Round 3: Grasp Interface Diagnostic

- **状态**：BLOCKED；等待 corrected O171C 后重新选择 operation baseline
- **Candidates**：direct complete grasp qpath vs terminal open qpose + current-state online planner。
- **Decision**：先 GT interface pair，再 learned fixed-batch/DYN64；affordance ZERO/PRED/GT 作为独立 intervention，不与 operation 基础结果混淆。

## Long-Run Handoff

## Execution Policy

- 预计总运行时间不超过 20 分钟、且不需要 GPU 多卡/长 SAPIEN rollout 的实验：直接以前台命令在当前对话中运行，按 progress artifact 做有界轮询，terminal 后立即解析并分析，不交给 tmux。
- 预计超过 20 分钟、包含多 seed/GPU/SAPIEN 长 rollout 或需要跨 turn 等待的实验：放入项目目录下的 tmux supervisor。启动前在对话中说明 run id、命令、输出目录、预计耗时和 tmux 查看命令。
- 后台实验必须写 `commands.json`、`run_state.json`、`queue_state.json`、progress/heartbeat 和 terminal `summary.json`。用户查看方式固定为 `tmux attach -t <session>`；只读检查命令和下一次检查时间写入 `NEXT_CHECK*.md`。
- 后台运行期间不启动重复进程；下一次对话检查只读 artifact、PID、日志和 ETA。完成后由当前对话解析结果、更新 tracker、决定下一 round。

当 round 运行时间超过当前对话的 practical wait window：

1. 在 result root 写 `commands.json`、`run_state.json`、`queue_state.json` 和 progress/heartbeat；
2. 用 `long-experiment-supervisor` 做只读监控；
3. 记录当前阶段、完成单元、ETA、下一次检查时间；
4. 不启动重复进程、不修改运行中的科学合同。

## Round 6: G000C Clean Grasp Label Set

- **状态**：DONE；`results/a6_g000c_grasp_label_set_contract_v2/summary.json`
- **Hypothesis**：使用已验证的 A5 phase/three-state 合同，在 clean A6 TRAIN/CAL 上物理分离完整 qpath、open terminal qpose 和 closed operation state，能够为 G-TRAJ/G-QPOSE 提供同 group、同 K、无 outcome 排序的公平 teacher set。
- **Implementation**：一个 clean `sample_id` 对应一个 observation group；按 collection trajectory index 顺序最多取 K4，少于 K 不复制；qpath 以 7D 累计弧长重采样到 L64。trajectory/qpose/closed-state 分别写入独立 NPZ，qpose manifest 不含 stored qpath/path 字段。
- **Result**：TRAIN/CAL `557/102` groups、`26,597/4,919` source trajectories、`2,099/387` teachers；67 groups 少于 K。relative path/qpose roundtrip 最大误差均 `1.1921e-7`，open-vs-closed L2 最小 `0.00617`；所有实现与 forbidden-field checks 通过。
- **Repair audit**：v1 误把 float32 roundtrip 要求为 bitwise exact；v2 使用 `<=1e-6`。v1/v2 标签 NPZ hash 完全一致，故只修 gate、不改数据。
- **Decision**：解锁 G005C；先做 GT-TRAJ vs GT-QPOSE+current-state single-segment planner exact-paired 接口上限，不进入 learned grasp training。

## Round 7: Replay-pass Teacher Repair and Candidate-Set Ceiling

- **状态**：G000C v3 DONE；G005 top-1 pilot SUPERSEDED；candidate-set ceiling DONE。
- **发现**：G000C v2 直接取 collection index 前 K 条，没有消费已有 replay quality gate。审计显示 513/2486 selected teacher replay-fail，263 groups 混入失败 teacher，27 groups 没有 replay-pass trajectory。
- **修复**：G000C v3 只保留 `fixed_gate_pass`，保持原 index 顺序取 K4；零合法 teacher group 排除并写 exclusion artifact。v3 通过 exact join、label、finite、roundtrip 和 forbidden-field checks。
- **结果**：531/101 groups，1991/382 selected teachers；all 2373 selected teachers replay-pass。旧 v1/v2 G005 和 physical pilot 不再引用。
- **G005 v4**：qpose-only single-segment planner 在 8/8 CAL targets 成功，terminal max error `2.38e-7`。
- **Top-1 pilot（仅诊断）**：GT-TRAJ 8/8 strict grasp、2/8 task、mean progress `0.18468`；GT-QPOSE slot0 3/8 strict grasp、2/8 task、mean progress `0.09857`。两者均接 O185，说明 grasp-to-operation handoff 是主要瓶颈，但 slot0 不能代表 K4 candidate set。
- **下一步**：对每个 group 的全部 valid K4 candidates 做 planner/physical candidate-level screen；用 planner success -> shortest joint-space path -> slot tie-break 的无 outcome 规则选 route-level candidate，完成后才启动 G010/G020 learned grasp training。
- **Planner v7**：31/31 valid qpose candidates 独立规划成功，8/8 groups 有 route-level candidate；max terminal error `1.19e-7`。跨 target/candidate 一次性 batch 的 v5 success 会受 batch composition 影响，已 supersede。
- **Physical long run**：2-group pilot 中 GT-TRAJ 8/8 candidate strict grasp、GT-QPOSE 0/8；为避免小样本结论，扩大到 8 targets。单 candidate 实测约 50--60 s，转入 tmux `a6_g005c_candidates`，read-only supervisor `a6_g005c_monitor`；GT-TRAJ v2 从 partial rows resume，完成后自动运行 GT-QPOSE v2。
- **GT-TRAJ full candidate result**：31 candidates 中 30 strict grasp、8 task；shortest-path no-outcome selector 在 8/8 groups strict grasp、2/8 task。direct trajectory consumer ceiling 成立，完整 task 的剩余瓶颈是真实 grasp-to-operation handoff。
- **GT-QPOSE full candidate result**：31 candidates 中 8 strict grasp、7 task；同一 shortest `(joint_space_length,candidate_index)` no-outcome selector 在 8 groups 上为 1/8 strict grasp、3/8 task。summary `selected` 已由现有 rows 离线重聚合写入 `results/a6_g005c_gt_qpose_candidate_physical_pilot_v2/summary.json`；selector 只读取 group/cost/index，outcome mutation invariance 通过。qpose route 不因该 heuristic selector 的低 strict rate 取消，G020 继续。
- **G005 decision**：GT-TRAJ/G-QPOSE 两条路线均解锁 matched learned fit；G010/G020 必须共用 observation join、K4/presence、seed/steps/optimizer/exposure 与 physical selector，不能根据 teacher outcome 选择 learned slot。

## Round 8: G006 observation join and G010/G020 matched grasp fit

- **G006C**：DONE；`results/a6_g006c_grasp_observation_label_contract_v1/summary.json`。632 groups joined exactly to primary `joint_bestview_dual.zarr` rows by `source_replay_id` (TRAIN 531, CAL 101). Inputs are XYZ plus current 7D arm qpos only; the Zarr fourth channel is affordance and is deliberately excluded, so it is not mislabelled as target-link mask. No future path, closed state, progress, contact log or outcome is in model input.
- **G010/G020 sanity**：DONE；both 50-step fixed-batch checks passed with finite decreasing loss, split counts, zero-affordance/no-oracle audit and reload max error `0`. A temporary failure from dropout-mode reload and a broadcasting warning was repaired before pass; no failed sanity metric is used scientifically.
- **Next**：launch matched 6000-step G010 direct `KxL64x7` trajectory and G020 `Kx7` terminal-qpose fits, same encoder/seed/optimizer/effective batch. Physical screens remain blocked until both terminal checkpoints and parity audit are available.
- **Renderer recovery**：单进程在约 15 个 fresh render world 后稳定失败；显式 close 不足。最终用每 candidate 独立 Python 进程 + rows resume 完成 GT-TRAJ，GT-QPOSE 同协议运行中。

## Round 9: Contact-Local Mode Generation and Affordance Semantics

- **状态**：G063/G064 DONE；G065 SCOPED_NEGATIVE / ORACLE_ONLY；A031 DONE；A032 formal RUNNING。
- **Hypothesis**：给定contact后的orientation/IK监督是多模态的；离散contact-local mode加连续residual可生成优于单值SE3的候选，但必须由合法的outcome-blind score选中。Initial/mix060 affordance只在同预算contact screen优于updated后才值得进入下游utility。
- **A031**：运行`run_a6_a031c_affordance_lineage_audit.py` probe/full。557 TRAIN primary、102 CAL primary、4899 TRAIN augmentation全部exact join；initial/updated/mix060 finite且1024点，mix060公式最大误差0。Decision：解锁A032，不替换A030 updated producer。
- **G063/G064**：G063的CAL M8相对M1 translation/rotation为`-0.42238 m [-0.44916,-0.39489]`与`-0.43663 rad [-0.63005,-0.24573]`；至少一个/两个合法IK为`93.98%/90.05%`。G064物化2373个mode/residual、IK-set和8cm pregrasp supervision，SE(3)重构与lineage/no-outcome合同通过。
- **G065 implementation repair**：`sapien`环境不支持`torch.flatnonzero`，在训练前改为等价的`torch.nonzero(..., as_tuple=True)[0]`。两支50-step sanity均通过finite/loss/reload/no-outcome gate。
- **G065 commands**：classifier与set-residual均使用seed`20260806`、6000 steps、batch64、AdamW `1e-4/1e-6`，在GPU0/1并行完成；formal耗时`76.65/77.20 s`。
- **G065 result**：classifier selected为`0.29315 m / 1.86826 rad / 0%`；set selected为101-group口径`0.25664 m / 1.81123 rad / 1.49%`，相对G062 translation `+0.03547 m [+0.01471,+0.05707]`。set best-of-8为`0.17218 m / 0.57954 rad / 3.47%`，相对G062 translation/rotation `-0.04899 m [-0.06667,-0.03135]`和`-1.09371 rad [-1.29318,-0.88923]`。
- **G065 decision**：`mode_scoring_analysis.json`确认`oracle_generation_supported=yes`、`selector_supported=no`。G066/G067/physical保持阻断；不把oracle候选当deployable top-1，不启动Diffusion或更大backbone。
- **A032 sanity**：新增matched INITIAL/MIX060 runner。两支50-step均通过loss下降、deterministic eval、reload exact、A031 lineage和no-outcome checks；sanity point metrics不用于语义选择。
- **A032 launch**：formal命令固定Point-M2AE、7000 steps、microbatch48、accumulation2、workers4、seed`20260806`。tmux `a6_a032_initial`使用GPU0，`a6_a032_mix060`使用GPU1；supervisor为`a032c_supervisor_config.json`，完成标志为各自`full/summary.json`，combined artifact为`results/a6_a032c_affordance_semantics_screen_v1/summary.json`。

## Round 10: G066 Selector Attribution and Direct-QPose Pivot

- **A032 terminal**：INITIAL contact mean distance `0.1061m`优于updated `0.1220m`，但3cm/5cm coverage为`41.10%/50.26%`，低于updated `45.55%/51.05%`；MIX060为`0.1304m/40.58%/46.60%`，三项均退化。保留A030 updated producer。
- **G066 S0/S1**：TRAIN-only risk calibrator实现/reload通过。S1相对S0小幅缩小oracle gap，但相对G062 translation仍退化`+0.02584m [+0.00770,+0.04401]`，未过门。
- **G066 S2**：101 CAL groups按8个不重叠shard完成；多GPU非零设备必须使用`CUDA_VISIBLE_DEVICES`映射到进程内`cuda:0`，否则SAPIEN/Warp到CuRobo切换触发CUDA illegal memory access。修复只改变设备映射，不改变数据或selector。
- **G066 terminal**：1942 legal IK、341 planner successes；S2 selected IK/planner coverage `97.64%/69.37%`，但pose为`0.2938m/2.163rad/0.74%`，相对G062退化`+0.07261m [+0.04807,+0.09670]`和`+0.48979rad [+0.28401,+0.69427]`。G067/physical停止。
- **Failure attribution**：full pose oracle只有`34.82%`有合法IK、`7.59%`有planner path；planner-feasible oracle相对full oracle退化`+0.10449m/+1.26870rad`。结论：主要瓶颈是predicted SE(3)候选realizability，不是继续调path-length selector。
- **G070 direct-qpose**：新增K8 relative-qpose set head，使用G064合法IK集合做TRAIN-only min-over-set supervision。50-step sanity通过；6000-step loss `0.8653 -> 0.1926`、reload 0，但CAL best-of-8 qpose-set L1为`0.5953rad`，没有`<=0.03`样本，初步负向。
- **G070 planner handoff**：2-group probe合同通过，any/selected planner coverage均`50%`。full 101-group screen在tmux `a6_g070_planner`运行；read-only supervisor `g070c_supervisor_config.json`。
- **G070 terminal**：101-group full在`728.4s`完成，103个candidate planner success；按valid query slot统计，any/selected planner coverage均`25.65%`。planner-selected qpose-set L1 `0.6014rad`，未优于best-of-8 oracle `0.5953rad`。结论为SCOPED_NEGATIVE：直接qpose-set未解决跨观测泛化与可执行候选覆盖，停止G070及本revision。

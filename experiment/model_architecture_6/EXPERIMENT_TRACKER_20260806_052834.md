# Architecture 6 Experiment Tracker

- **修订日期**：2026-08-06
- **状态**：`REOPENED_ACTIVE`
- **权威计划**：`EXPERIMENT_PLAN.md`
- **当前训练状态**：Round 1 `A6-D021C` CPU 预处理运行中；未启动 GPU operation training

状态语义：

- `DONE`：terminal artifact 与实现不变量通过。
- `READY`：当前可实现/执行，不依赖未完成科学结果。
- `BLOCKED`：等待表中明确依赖。
- `RUNNING`：存在真实 PID 或非 terminal artifact。
- `SCOPED_NEGATIVE`：实现有效，只否定该模型/表示/配置。
- `INVALID_IMPLEMENTATION`：实现或 normalizer 错误，禁止作科学引用。
- `SUPERSEDED`：被本 planning revision 取代，不得恢复为当前队列。
- `CONDITIONAL`：仅在计划中的证据触发条件出现后发布新 revision。

## Current Evidence

### Latest analysis

`a6_o201f_mlp_state_start_delta_fixed64_v1`：`INVALID_IMPLEMENTATION`。terminal metrics 为 normalized MAE `1.513186`、raw MAE `0.0135267`、zero-repeat `0`、delta/absolute parity `3.50e-7`、reload `0`；因复用了 absolute-action normalizer，不能作为 state-start-delta 科学负结论。`A6-O200F` 的 command-delta reconstructed MAE `0.0006511` 与 repeat-last `0.010378` 及 parity/reload 证据仍有效，旧 arbitrary performance gate 已由 revision `20260806T060210Z-c3831dc1` 撤销。

决策：保留 dirty IDs `70/251/349` 的 quarantine，不再执行 recovery；remaining work 为 `A6-D021C` command-delta TRAIN-only normalizer、`A6-D030C` clean DYN64 hash audit，随后才允许 operation DYN64。

| Run ID | Purpose | Status | Evidence / Decision |
|---|---|---|---|
| `A6-R100` | dirty sample clean contract | DONE | 816 -> 813 samples；排除 replay IDs 70/251/349；raw collection 保留；所有 A6 派生 manifest 同步排除 |
| `A6-A000C` | clean affordance membership | DONE | TRAIN 557/557、CAL 102/102 exact join；零 overlap/join failure；A010C 解锁 |
| `A6-R110` | old gate reinterpretation | DONE | command-delta 晋级；absolute negative 限定范围；旧 O201F 因 normalizer 错误失效 |
| `A6-D020C` | clean operation index/fixed64/absolute normalizer | DONE | TRAIN 557 samples、26,597 trajectories、36,700,596 command rows、64 chunks；hash/mask/heldout checks 通过 |
| `A6-D030` | original DYN64 materialization | DONE | 64 targets、1024 frames、source hash 与两 worker artifact 完整；需 D030C 确认不含排除样本后复用 |
| `A6-O200F` | command-delta fixed64 MLP | DONE | reconstructed MAE 0.000651，repeat-last 0.010378，parity/reload 通过；旧 multiplicative gate 已退休 |
| `A6-O010S` | MLP absolute fixed64 | SCOPED_NEGATIVE | 仅说明当前 absolute-action MLP 配置不优于 repeat-last |
| `A6-O020S` | parallel absolute fixed64 | SCOPED_NEGATIVE | 仅说明当前 absolute-action parallel 配置不优于 repeat-last |
| `A6-O030S` | causal absolute fixed64 | SCOPED_NEGATIVE | 当前 causal AR 与 teacher-forced 差距大；不影响其他 decoder |
| `A6-O201F` | state-start-delta fixed64 | INVALID_IMPLEMENTATION | 使用 absolute-action normalizer；禁止作为 representation 负结论 |
| `A6-A000RR/RRR/RRRR` | missing-row recovery | SUPERSEDED | 三个 dirty samples 已从 clean contract 排除；禁止重跑 recovery |
| old global termination | Architecture 6 unsuccessful | SUPERSEDED | 基于错误门槛、normalizer 和旧 membership；不再控制队列 |

## Phase 0: Active Preprocessing

| Run ID | Purpose | Inputs | Required Checks | Depends On | Status | Next |
|---|---|---|---|---|---|---|
| `A6-D021C` | command-delta TRAIN-only normalizer | D020C sample index + clean TRAIN actions | finite；valid-mask exact；delta/absolute roundtrip；zero CAL/heldout read；hash persisted | D020C | RUNNING | O100C/O110C/O120C |
| `A6-D030C` | clean membership/hash audit of D030 | D030 manifest + R100 exclusion manifest + D020C source hashes | excluded samples absent；64 targets unchanged；source/output hashes exact | R100,D020C,D030 | READY | O100C/O110C/O120C |
| `A6-A010C` | affordance clean fixed8 fit | A000C clean TRAIN primary rows | finite grad/output；mask/shape；reload；zero heldout read | A000C | READY | A020C |

`A6-D021C`、`A6-D030C` 与 `A6-A010C` 互相独立。资源允许时并行，禁止人为串行。

## Phase 1: Operation Architecture Comparison

| Run ID | Decoder / Representation | Data | Comparison | Depends On | Status |
|---|---|---|---|---|---|
| `A6-O100C` | MLP command-delta | fixed64 -> DYN64 TRAIN/CAL | repeat-last、TRAIN mean delta、paired offline/replay/live metrics | D021C,D030C | BLOCKED |
| `A6-O110C` | parallel-query command-delta | same | same samples/config/budget/seed/checkpoint rule | D021C,D030C | BLOCKED |
| `A6-O120C` | causal command-delta | same | autoregressive metrics + teacher-forcing diagnostic | D021C,D030C | BLOCKED |
| `A6-O130C` | predicted-command replay | all three terminal arms | exact-paired replay error、endpoint、failure strata | O100C,O110C,O120C | BLOCKED |
| `A6-O140C` | operation-only live screen | all valid arms | progress、wrong-way、contact retention、task、latency | O130C | BLOCKED |
| `A6-O150C` | architecture decision audit | existing artifacts only | paired CI + frozen engineering tie-break；no new rollout | O140C | BLOCKED |

三支 decoder 必须全部形成 terminal artifact 后才允许 `O150C`。任一 arm 的 scoped negative 不阻塞其余 arm。

## Phase 2: Operation Strategy And Formal Selection

| Run ID | Purpose | Trigger / Variable | Depends On | Status |
|---|---|---|---|---|
| `A6-O200C` | sampling comparison | raw-uniform vs motion-balanced | O150C | BLOCKED |
| `A6-O210C` | endpoint weighting | one-variable comparison | O150C | BLOCKED |
| `A6-O220C` | affordance intervention | ZERO/PRED/GT；frozen sensitivity + matched-train utility | O150C,A030C | BLOCKED |
| `A6-O230C` | scheduled sampling | only when offline/replay good but live degrades | O140C evidence | CONDITIONAL |
| `A6-O300C` | formal multi-seed TRAIN | frozen operation candidates | strategy decision | BLOCKED |
| `A6-O310C` | A5_CAL exact-paired selection | all O300C seeds/checkpoints | O300C | BLOCKED |
| `A6-O320C` | operation result/claim audit | no new rollout | O310C | BLOCKED |

## Phase 3: Affordance

| Run ID | Purpose | Split | Depends On | Status |
|---|---|---|---|---|
| `A6-A020C` | clean full train, multiple seeds | A5_TRAIN | A010C | BLOCKED |
| `A6-A030C` | checkpoint/consumer freeze | A5_CAL + live DYN observation | A020C | BLOCKED |
| `A6-A040C` | affordance result/claim audit | existing artifacts | A030C | BLOCKED |

Affordance branch 不阻塞 ZERO operation 或 ZERO grasp。GT affordance 仅为诊断上限。

## Phase 4: Grasp Representation Comparison

| Run ID | Purpose | Route / Metric | Depends On | Status |
|---|---|---|---|---|
| `A6-G000C` | qpath/qpose label-set contract | qpath、open terminal qpose、closed operation state 分离 | O320C | BLOCKED |
| `A6-G005C` | interface oracle pair | GT-TRAJ vs GT-QPOSE+online planner；exact-paired | G000C | BLOCKED |
| `A6-G010C` | direct trajectory fixed-batch fit | learned complete qpath | G005C | BLOCKED |
| `A6-G020C` | qpose fixed-batch fit | learned open terminal qpose + online planner | G005C | BLOCKED |
| `A6-G030C` | parity audit | same encoder/data/candidates/seed/exposure/executor | G010C,G020C | BLOCKED |
| `A6-G100C` | DYN64 offline comparison | route validity、path/qpose error、planner coverage | G030C | BLOCKED |
| `A6-G110C` | physical comparison | reach、contact、retention、transition、task、latency | G100C | BLOCKED |
| `A6-G120C` | route decision audit | paired CI + engineering tie-break | G110C | BLOCKED |
| `A6-G130C` | grasp affordance intervention | ZERO/PRED/GT on surviving routes | G120C,A030C | BLOCKED |
| `A6-G200C` | formal multi-seed grasp train | frozen candidates | G130C | BLOCKED |
| `A6-G210C` | A5_CAL grasp selection | exact-paired full-stage metrics | G200C | BLOCKED |
| `A6-G220C` | grasp result/claim audit | no new rollout | G210C | BLOCKED |

## Phase 5: Joint Optimization

| Run ID | Purpose | Comparison | Depends On | Status |
|---|---|---|---|---|
| `A6-J000C` | frozen deployable pipeline | frozen operation + frozen grasp + deployable PRED/ZERO affordance | O320C,G220C,A040C | BLOCKED |
| `A6-J100C` | supervised joint training | frozen pipeline vs shared-encoder joint model | J000C | BLOCKED |
| `A6-J200C` | bounded task fine-tune | only after written TRAIN-only error attribution | J100C evidence | CONDITIONAL |
| `A6-J210C` | complete-system selection | read A5_MECH_DEV once | J100C + J200C if triggered | BLOCKED |
| `A6-J220C` | deployment audit | zero oracle fields；hash/config/stop rule frozen | J210C | BLOCKED |

## Phase 6: Final

| Run ID | Purpose | Split | Depends On | Status |
|---|---|---|---|---|
| `A6-F000C` | same-target robustness | SAME_TEST once | J220C | BLOCKED |
| `A6-F010C` | target-disjoint generalization | A5_TARGET_TEST once | J220C | BLOCKED |
| `A6-F020C` | final aggregation | no new rollout | F000C,F010C | BLOCKED |

## Runtime Control Checklist

- [ ] pending planning revision 已由 primary worker 读取并 ack
- [ ] `active_contract.json` 对应当前 run，不再指向 O201F
- [ ] `actual_launch_config.json` 与 active contract 完全一致
- [ ] `preflight_contract_audit.json` status 为 passed 且 contract hash 一致
- [ ] `loop_config.json` 的 iteration、completion、dashboard、resource mode 与真实 run 一致
- [ ] GPU launch 前 GPUALIVE 已同步停止
- [ ] 训练启动后确认真实 PID 或 terminal artifact，再设置 `waiting_external`
- [ ] terminal 后先分析全部同级 arm，再改变队列
- [ ] 任意 negative 只按其 scope 更新，不自动全局停止

## Immediate Authorization

本修订只授权 planning/control 恢复与 `A6-D021C`、`A6-D030C`、`A6-A010C`。当前对话不启动训练。三支 operation GPU run 只有在 D021C/D030C 和各自 preflight 通过后才可由 primary worker 并行启动。

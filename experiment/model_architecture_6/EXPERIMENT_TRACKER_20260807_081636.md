# Architecture 6 Experiment Tracker

- **修订日期**：2026-08-07
- **状态**：`GRASP_FACTOR_REVISION_EXECUTING / G065C_ORACLE_GENERATION_POSITIVE_SELECTOR_NEGATIVE / G066C_SCORING_READY / A032C_RUNNING`
- **权威计划**：`EXPERIMENT_PLAN.md`
- **当前训练状态**：A032C INITIAL/MIX060 formal 正在GPU0/1并行；tmux `a6_a032_initial` / `a6_a032_mix060`；完成标志为各自`full/summary.json`
- **最新抓取状态（2026-08-07）**：G063/G064证明多mode与multi-IK监督可构造；G065 best-of-8 oracle显著优于G062但selected top-1退化；授权G066 outcome-blind scoring诊断，physical仍阻断

状态语义：

- `DONE`：terminal artifact 与实现不变量通过。
- `READY`：当前可实现/执行，不依赖未完成科学结果。
- `BLOCKED`：等待表中明确依赖。
- `RUNNING`：存在真实 PID 或非 terminal artifact。
- `SCOPED_NEGATIVE`：实现有效，只否定该模型/表示/配置。
- `INVALID_IMPLEMENTATION`：实现或 normalizer 错误，禁止作科学引用。
- `INVALID_LIVE_PROTOCOL`：checkpoint/offline 可保留，但该 live rollout 与其闭环结论禁止引用。
- `INVALID_SHARED_WORLD_ORDER`：多个 arm 共用一个 SAPIEN world；首 arm 原始事实可单独保留，第二及后续 arm 和所有 paired 结论禁止引用。
- `INVALID_LIVE_DEPENDENCY`：本身可能含有效离线部分，但依赖无效 live 结果的决策撤销。
- `PARTIAL_INVALID`：artifact 中明确列出的子结果保留，其余结论撤销。
- `SUPERSEDED`：被本 planning revision 取代，不得恢复为当前队列。
- `CONDITIONAL`：仅在计划中的证据触发条件出现后发布新 revision。

## Current Evidence

### Grasp factor revision execution (2026-08-07)

`A6-A031C`通过initial/current-updated/mix060 exact lineage：TRAIN primary 557、CAL primary 102、TRAIN augmentation 4899，mix060公式最大误差0。该结果只解锁matched A032 screen，不替换A030 updated producer。

`A6-G063C/G064C`通过contact-local mode与IK-set合同。G063 CAL M8相对M1的translation/rotation为`-0.42238 m [-0.44916,-0.39489]`与`-0.43663 rad [-0.63005,-0.24573]`；CAL至少一个/两个合法IK为`93.98%/90.05%`。G064物化2373 labels、8-mode residual、IK-set和pregrasp keypose，lineage/no-outcome检查全通过。

`A6-G065C`两支6000-step matched fit均实现通过但promotion失败。classifier top-1为`0.29315 m / 1.86826 rad / 0%`；set-residual selected top-1为`0.25664 m / 1.81123 rad / 1.49%`（101-group口径），相对G062 translation退化`+0.03547 m [+0.01471,+0.05707]`。同一set的best-of-8 oracle为`0.17218 m / 0.57954 rad / 3.47%`，相对G062 translation/rotation改善`-0.04899 m [-0.06667,-0.03135]`与`-1.09371 rad [-1.29318,-0.88923]`。结论限定为candidate generation有oracle信号、mode selector失败；只解锁G066 scoring诊断，G067/physical保持阻断。

Planning微调后，G065不解锁physical，但其oracle generation信号解锁`A6-G066C` scoring/realization诊断。G066固定同一候选池并比较`S0 mode-logit / S1 pose-likelihood+uncertainty / S2 IK-FK-planner rank`；只有selected top-1缩小oracle gap且相对G062 pose CI不退化才进入G067。A032继续按原配置运行，不因revision重启。

### Latest analysis

`A6-R120C` planning/implementation audit found two invalid comparison assumptions: O100C evaluated an O200F checkpoint trained with the D020 absolute-action std using the D021C command-delta std; and D040C/O110C/O120C use only A5_TRAIN recorded observations (with logged contact feedback), not A5_CAL or live observations. Old O100/O110/O120 numbers are therefore not deployment claims; O100 is invalid implementation evidence, O110/O120 are diagnostic-only.

`A6-O100RC` corrected MLP fixed64 fit now passes with D021C from-scratch training (`fixed normalized delta MAE 0.012657`, raw delta MAE `0.0002923`, reload error `0`). This is still only a fixed-batch implementation check; it does not replace the missing A5_CAL/live screen.

`A6-R121C` input-semantic audit invalidates the old context: collection `contact_feedback[:,33]` includes object progress, target open ratio, and controller-internal telemetry; D040/D041 also index it with raw anchors although the log is operation-relative. Preserve point/state/label/split artifacts, but invalidate all old model comparisons that consumed this context. Baseline must use zero contact/availability and task metadata only.

`A6-O131C` valid live8 evidence: MLP `2/8`, parallel `1/8`, repeat-last `0/8`; paired mean progress gains are `+0.1696 [0.0693,0.2840]` and `+0.1671 [0.0715,0.2692]`. MLP vs parallel CI overlaps zero. Expand TRAIN target coverage; contact retention is the current bottleneck.

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
| `A6-D021C` | command-delta TRAIN-only normalizer | D020C sample index + clean TRAIN actions | finite；valid-mask exact；delta/absolute roundtrip；zero CAL/heldout read；hash persisted | D020C | DONE | `results/a6_d021c_command_delta_normalizer_v1/summary.json`；delta rows 12,774,558 |
| `A6-D030C` | clean membership/hash audit of D030 | D030 manifest + R100 exclusion manifest + D020C source hashes | excluded samples absent；64 targets unchanged；source/output hashes exact | R100,D020C,D030 | DONE | `results/a6_d030c_clean_dyn64_hash_audit_v1/summary.json`；可直接复用 D030 |
| `A6-D040C` | DYN64 command-delta input adapter | D030 materialization + clean trajectory + D021C normalizer | 2-target probe；64 targets/1024 rows；state/context/label shapes；delta roundtrip；forbidden-field audit | D021C,D030C | DIAGNOSTIC_ONLY | 全部为 A5_TRAIN；部分 context 来自 recorded contact_feedback，不支持部署结论 |
| `A6-D041C` | clean A5_CAL recorded-current-observation input | A5_CAL trajectories + clean metadata + D021C normalizer | CAL membership；target-disjoint pairing；recorded-vs-live source label；forbidden future/outcome audit | D021C,D030C | PARTIAL_VALID | 35 targets/280 rows split/point/state/label valid；旧 context invalid |
| `A6-D042C` | zero-contact deployable-schema fixed64/CAL inputs | O000BR2 fixed input + D041C point/state/label | context[0:34] exact zero；metadata tail exact；point/state/label hash parity | D041C | READY | 完成后从头重训三支 decoder |
| `A6-D043C` | TRAIN194 mixed recipe | clean A5_TRAIN, 1 trajectory x 8 anchors per target | 194 targets；1552 rows；zero CAL overlap；zero contact | D042C | MIXED_INTERVENTION | 同时改变 anchors 和 trajectory；不得称为 pure coverage |
| `A6-D044C` | additive TRAIN194 coverage | D042C exact 1024-row prefix + D043C 130 new targets | 194 targets；2064 rows；prefix exact；zero CAL/contact | D042C,D043C | DONE | all checks passed；`results/a6_d044c_additive_train194_zero_contact_input_v1/full/summary.json` |
| `A6-A010C` | affordance clean fixed8 fit | A000C clean TRAIN primary rows | finite grad/output；mask/shape；reload；zero heldout read | A000C | READY | A020C |

`A6-D021C`、`A6-D030C` 与 `A6-A010C` 互相独立。资源允许时并行，禁止人为串行。

## Phase 1: Operation Architecture Comparison

| Run ID | Decoder / Representation | Data | Comparison | Depends On | Status |
|---|---|---|---|---|---|
| `A6-O100C` | MLP command-delta | fixed64 -> DYN64 TRAIN/CAL | repeat-last、TRAIN mean delta、paired offline/replay/live metrics | D021C,D030C,D040C | INVALID_IMPLEMENTATION | O200F checkpoint 使用 absolute-action std；撤销原 O100 数字 |
| `A6-O100RC` | corrected MLP command-delta fixed64 | D021C + fixed64 | finite、delta parity、reload | D021C | INVALID_INPUT_CONTEXT | normalizer 修复有效，但消费旧 telemetry context |
| `A6-O110C` | parallel-query command-delta | same | same samples/config/budget/seed/checkpoint rule | D021C,D030C,D040C | INVALID_INPUT_CONTEXT | 消费含 oracle/internal fields 的 recorded telemetry |
| `A6-O120C` | causal command-delta | same | autoregressive metrics + teacher-forcing diagnostic | D021C,D030C,D040C | INVALID_INPUT_CONTEXT | 消费含 oracle/internal fields 的 recorded telemetry |
| `A6-O121C` | A5_CAL recorded-observation paired screen | corrected checkpoints + D041C | paired target CI | D041C,O100RC,O110C,O120C | INVALID_INPUT_CONTEXT | split/statistics code valid，但三支均消费旧 telemetry context；数值禁止引用 |
| `A6-O131C` | TRAIN1024 live8 aggregate | MLP/PAR/repeat-last | source-horizon exact-paired live progress/contact/task | O127C,O128C | INVALID_LIVE_PROTOCOL | 错误 operation-start last command；数值禁止引用 |
| `A6-O132C` | MLP D043 mixed-recipe training | D043C；D021C normalizer；6k steps | train fit；finite；zero contact | D043C | DIAGNOSTIC_ONLY | raw delta MAE `0.0010514`；不能隔离 coverage |
| `A6-O133C` | parallel D043 mixed-recipe training | D043C；D021C normalizer；6k steps | train fit；finite；zero contact | D043C | DIAGNOSTIC_ONLY | raw delta MAE `0.0018326`；不能隔离 coverage |
| `A6-O134C` | D043 frozen A5_CAL screen | O132C/O133C + unchanged D042C CAL | repeat-last；paired target CI；vs O129C | O132C,O133C | DIAGNOSTIC_ONLY | offline improved；MLP `0.002384`；PAR `0.002663` |
| `A6-O135C` | D043 source-horizon live8 | MLP/PAR/repeat-last；same 8 CAL targets | task；progress；contact retention；zero model oracle | O134C | INVALID_LIVE_PROTOCOL | checkpoint/offline 保留，live 数值禁止引用 |
| `A6-O136C` | D043 live8 aggregate | O135C vs O131C exact target pairing | paired progress/contact mixed-recipe effect | O135C | INVALID_LIVE_PROTOCOL | 两侧均依赖错误 start command |
| `A6-O130C` | predicted-command replay | all three terminal arms | 复用 Arch2/3 live operation evaluator；每个 chunk 后重新观测 | O100C,O110C,O120C | BLOCKED | 2-target short-horizon probe infrastructure passed，但单 chunk 不能代表完整任务；不得扩展旧 probe |
| `A6-O137C` | MLP additive TRAIN194 | D044C；same seed/budget | train fit + frozen CAL | D044C | DONE | raw delta MAE `0.0011165`；42.4 s |
| `A6-O138C` | parallel additive TRAIN194 | D044C；same seed/budget | train fit + frozen CAL | D044C | DONE | raw delta MAE `0.0021463`；60.4 s |
| `A6-O139C` | additive TRAIN194 CAL | O137C/O138C | paired target CI vs O129C | O137C,O138C | DONE | MLP/PAR paired improvement CIs fully below zero |
| `A6-O140C` | additive TRAIN194 live8 | MLP/PAR/repeat-last | same source horizons and stop | O139C | INVALID_LIVE_PROTOCOL | O137/O138 train 与 O139 CAL 保留 |
| `A6-O141C` | additive coverage decision | O140C vs O131C exact target pairing | progress/contact/task；no hard threshold | O140C | INVALID_LIVE_DEPENDENCY | offline coverage 结论保留，闭环结论撤销 |
| `A6-O142C` | horizon/exposure attribution | O127/128 vs O132/133 vs O137/138 on same CAL | prefix8/suffix24/difference/per-horizon error | O141C | PARTIAL_INVALID | offline horizon 统计保留，因旧 live 对照而不能归因 exposure |
| `A6-O143C` | MLP consistent perturb 1x | D042C TRAIN1024；50% arm offset | clean/perturbed fit；absolute target invariant | O142C | DONE | clean raw `0.000909`；invariance `2.38e-7` |
| `A6-O144C` | MLP consistent perturb 3x | same except scale 3 | same | O142C | DONE | clean raw `0.000975`；invariance `2.38e-7` |
| `A6-O145C` | perturb CAL matrix | baseline/1x/3x x clean/1x/3x contexts | full/prefix target-paired error | O143C,O144C | DONE | all perturb-vs-baseline CIs cross zero |
| `A6-O146C` | perturb live8 | baseline/1x/3x/repeat-last | same source horizons；progress/contact/task | O145C | INVALID_LIVE_PROTOCOL | O143/O144 train 与 O145 CAL 保留 |
| `A6-O147C` | perturb decision | O146C exact target pairing | paired effects；no hard threshold | O146C | INVALID_LIVE_DEPENDENCY | 不得据此否定 perturb |
| `A6-O148C` | replan prefix K4 | frozen O127C；same total physics steps | progress/contact/task/latency vs K8 | O147C | INVALID_LIVE_PROTOCOL | 不得引用旧 contact/progress |
| `A6-O149C` | replan prefix K2 | frozen O127C；same total physics steps | same | O148C signal | INVALID_LIVE_PROTOCOL | 不得引用旧 contact/progress/task |
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
| `A6-A020C` | clean current-updated full train, multiple seeds | A5_TRAIN | A010C | DONE；3 seeds fixed-last |
| `A6-A030C` | current-updated checkpoint/consumer freeze | A5_CAL initial observation | A020C | DONE / POSITIVE；dynamic live尚未claim |
| `A6-A040C` | old affordance result/claim audit | existing artifacts | A030C | SUPERSEDED；由A031-A034下游utility审计取代 |

Affordance branch 不阻塞 ZERO operation 或 ZERO grasp。GT affordance 仅为诊断上限。

## Phase 4: Grasp Representation Comparison

| Run ID | Purpose | Route / Metric | Depends On | Status |
|---|---|---|---|---|
| `A6-G000C-v2` | qpath/qpose label-set contract | 未过滤 replay-fail teacher | O199C operation decision | SUPERSEDED；513/2486 selected teacher replay-fail |
| `A6-G000C-v3` | qpath/qpose label-set contract | replay-pass filter + zero-group exclusion | O199C operation decision | DONE；2373 teachers，531/101 groups |
| `A6-G005C-top1` | interface oracle pair | slot0 GT-TRAJ vs GT-QPOSE+online planner | G000C-v2 | SUPERSEDED；teacher source invalid |
| `A6-G005C-candidate-set` | interface oracle pair | all valid K4 candidates；planner/reach/contact/task | G000C-v3 | DONE；GT-TRAJ candidate 30/31 strict、8/31 task、route 8/8 strict/2/8 task；GT-QPOSE candidate 8/31 strict、7/31 task、route 1/8 strict/3/8 task；selector outcome-invariant |
| `A6-G010C` | direct trajectory fixed-batch fit | learned complete qpath | G005C | READY；先 G006 observation/label join + fixed-batch sanity |
| `A6-G020C` | qpose fixed-batch fit | learned open terminal qpose + online planner | G005C | READY；与 G010 共用 G006 contract |
| `A6-G030C` | parity audit | same encoder/data/candidates/seed/exposure/executor | G010C,G020C | BLOCKED |
| `A6-G100C` | DYN64 offline comparison | route validity、path/qpose error、planner coverage | G030C | BLOCKED |
| `A6-G110C` | physical comparison | reach、contact、retention、transition、task、latency | G100C | BLOCKED |
| `A6-G120C` | route decision audit | paired CI + engineering tie-break | G110C | BLOCKED |
| `A6-G130C` | old grasp affordance intervention | ZERO/PRED/GT on surviving routes | G120C,A030C | SUPERSEDED；由A031-A034/G067语义干预取代 |
| `A6-G200C` | formal multi-seed grasp train | frozen candidates | G130C | BLOCKED |
| `A6-G210C` | A5_CAL grasp selection | exact-paired full-stage metrics | G200C | BLOCKED |
| `A6-G220C` | grasp result/claim audit | no new rollout | G210C | BLOCKED |

G006C observation-label join is terminal pass (`632` groups, `531/101` TRAIN/CAL; ZERO-affordance XYZ+qpos schema). Both G010C and G020C 50-step sanity runs passed finite/loss-decrease/reload/no-oracle checks; formal 6000-step fits are the current authorized runs.

## Phase 5: Joint Optimization

| Run ID | Purpose | Comparison | Depends On | Status |
|---|---|---|---|---|
| `A6-J000C` | frozen deployable pipeline | winning affordance/contact + G066 grasp + O185 operation | G068,A033/A034 | BLOCKED |
| `A6-J050C` | shared Stage-1 geometry representation | independent affordance/contact/mode/dual-space keypose heads；operation frozen | J000C | BLOCKED |
| `A6-J100C` | supervised joint comparison | modular vs shared + parameter-matched control | J050C | BLOCKED |
| `A6-J150C` | transition-aware global optimization | post-grasp state ranking/handoff；O185 frozen | G069,J100C | BLOCKED |
| `A6-J200C` | bounded task fine-tune | only after written TRAIN-only handoff error attribution | J150C evidence | CONDITIONAL |
| `A6-J210C` | complete-system selection | read A5_MECH_DEV once | J150C + J200C if triggered | BLOCKED |
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

`A6-D021C/D030C/D040C` 与三支 operation decoder 离线 screen 已完成。MLP 仅改善约 1.1%，parallel-query 改善约 21.3%，causal teacher-forced 很强但 autoregressive 比 repeat-last 差约 3.3%。O130 短 horizon probe 证明底层执行接口可用，但 learned chunk 必须接入 Arch2/3 的 live observation loop；下一步实现 online adapter，再做 exact-paired full replay。

## A6-R150C Focused Operation Revision

以下条目覆盖上面的历史 Immediate Authorization；O130 不得重新启动。

| Run ID | Purpose | Status | Evidence / Next |
|---|---|---|---|
| `A6-D150C` | 81D robot history 后追加 4D FK/visible-target relative feature；TRAIN1024 + CAL280 | DONE | `results/a6_d150c_fk_target_relative_inputs_v1/summary.json`；所有合同检查通过 |
| `A6-O151C` | matched 85D MLP，ZERO contact，D021C，seed 20260805，6000 steps/batch64 | READY | 只改变 state representation；O127C checkpoint 不覆盖 |
| `A6-O152C` | A5_CAL exact-paired baseline/geometry/zero-feature offline screen | READY | 不设绝对 MAE gate；仅作可学习性与 sensitivity 诊断 |
| `A6-O153C` | K8 source-horizon live8 baseline/geometry/repeat-last | READY | 同 O131C target/calls/stop；terminal 后按 paired progress/contact/task 决策 |

### A6-R150C decision rules

- 只有实现不变量、split/hash、forbidden-field、reload/parity 失败才是 implementation/data-contract failure；修复同一配置后重跑。
- geometry-vs-O127C 的 paired CI 跨零时不宣称优胜，O127C K8 继续作为默认；task/progress 下降时只记录 scoped closed-loop negative。
- 只有 paired progress CI 下界大于零且 task 不退化，才授权第二 seed；否则下一方向是 TRAIN-only simulator recovery supervision。
- 不重新做 replay/底层合同，不重跑 dirty-row recovery，不加入固定百分比性能门。

### A6-R154C isolation correction

| Run ID | Purpose | Status | Evidence / Next |
|---|---|---|---|
| `A6-O151C` | naive 85D concat MLP train | CONFOUNDED | finite/reload/train valid，但 `LayerNorm(81)->LayerNorm(85)` 改变了原状态归一化 |
| `A6-O152C` | naive concat CAL | CONFOUNDED | geometry-minus-baseline target-paired MAE `-0.000053 [-0.000264,0.000153]`，不可区分 |
| `A6-O153C` | naive concat live8 | INVALID_LIVE_PROTOCOL | training/CAL 保留；live start command 错误 |
| `A6-O154C` | zero-init geometry residual，81D state encoder 不变 | READY | 必须先过 common-init/output parity，再 matched 6k training |
| `A6-O155C` | corrected residual CAL | READY | 同 O152C rows/evaluator |
| `A6-O156C` | corrected residual live8 | READY | 同 O153C target/calls/K8/evaluator |

历史上 O153C 的 baseline 逐 target exact 复现 O131C，只证明同一错误协议可重复；该句不再构成 live protocol 有效性证据。

### A6-R160C Recovery Supervision Revision

| Run ID | Purpose | Status | Evidence / Next |
|---|---|---|---|
| `A6-O156C` | corrected geometry residual live8 | INVALID_LIVE_PROTOCOL | training/CAL 保留；live start command 错误 |
| `A6-D160C` | 16 TRAIN targets x 4 frozen-O127C live calls，64 recovery rows | INVALID_START_PROTOCOL | 错误 finger last-command 污染 recovery delta label |
| `A6-O161C` | D042C 1024 prefix + D160C 64 rows，原 MLP matched train | INVALID_DESCENDANT | checkpoint 依赖无效 D160C |
| `A6-O162C` | unchanged A5_CAL offline screen | INVALID_DESCENDANT | 评测本身可运行，但被评 checkpoint 无效 |
| `A6-O163C` | unchanged A5_CAL live8 | INVALID_DESCENDANT + INVALID_LIVE_PROTOCOL | 不得引用旧 progress/contact/task |

Recovery branch decision: only implementation/data-contract failures trigger repair/re-run; no absolute performance gate. Paired progress/contact/task support is required before a second seed or more recovery rows. No TRAIN-only recovery result can be used to claim deployment until the same frozen live8 screen is complete.

旧 Terminal decision 已撤销；O185 seed1 已冻结为 provisional operation 候选，O127 作为 fallback，G000C 可以进入。

### A6-R170C Corrected Live Protocol

| Run ID | Purpose | Status | Evidence / Next |
|---|---|---|---|
| `A6-R170C` | live8 CAL 8 + recovery TRAIN 16 start-command audit | DONE | 24/24 全部检查通过 |
| `A6-O171C-sanity` | corrected target0 3-call MLP/PAR/repeat-last | INVALID_SHARED_WORLD_ORDER | start lineage 有效；跨 arm outcome 无效 |
| `A6-O171C` | corrected source-horizon live8 | PARTIAL_INVALID | 首 arm O127 rows 保留；MLP-vs-PAR/repeat paired 结论撤销 |
| `A6-O172C` | corrected valid-checkpoint intervention screen | PARTIAL_INVALID | 首 arm baseline 保留；所有后续 intervention 排名撤销 |
| `A6-D180C` | full-horizon TRAIN recovery state/teacher alignment | DONE | 128 rows；full spread；correct command；oracle label-only |
| `A6-O181C` | time-aligned recovery train | DONE | checkpoint/finite/reload/prefix 证据有效；旧 live outcome 撤销 |
| `A6-O182C` | progress-aligned recovery train | DONE | 训练/离线实现有效；旧 live negative 撤销，是否复测由 fresh-world 证据决定 |
| `A6-O183C` | frozen CAL alignment screen | DONE | time/progress 均回归；仅诊断 |
| `A6-O184C` | corrected recovery live8 | PARTIAL_INVALID | 首 arm baseline 保留；后续 time/progress live 结论撤销 |
| `A6-O185C` | frozen O127 + zero-init recovery residual | DONE | parity/freeze/optimizer/reload 全通过 |
| `A6-O186C` | residual CAL screen | DONE | vs uniform MAE `-0.00063`；vs baseline `+0.00057` |
| `A6-O187C` | residual corrected source-horizon live8 | PARTIAL_INVALID | 首 arm baseline 保留；residual/uniform paired 结论撤销 |
| `A6-O188C` | fixed 650-call deployment live8 | PARTIAL_INVALID | 首 arm O127 `2/8` 保留；其余 arm 和 paired 结论撤销 |
| `A6-O189C` | residual second seed train | DONE | implementation checks pass；metrics 与 seed1 接近 |
| `A6-O190C` | residual seed CAL | DONE | seed1/seed2 CAL 几乎相同 |
| `A6-O191C` | residual seed fixed-budget live8 | INVALID_SHARED_WORLD_ORDER | 已 terminal；首 arm seed1 `5/8` 仅作单臂事实，seed2/control comparison 无效 |
| `A6-O192C` | fresh-world first/last checkpoint order audit | DONE | target 1/2；lineage pass；trace/final progress/contact 差异全为 0 |
| `A6-O193C` | fresh-world O127/seed1/seed2/repeat fixed-budget live8 | DONE | `2/8` vs `5/8` vs `3/8` vs `0/8`；所有 independent-world checks pass |
| `A6-O194C` | two epoch-balanced residual train seeds | DONE | exact exposure/parity/freeze/reload checks pass |
| `A6-O195C` | balanced residual CAL audit | DONE | consumer checks pass；balanced/random CAL essentially equivalent |
| `A6-O196C` | baseline/random/balanced1/balanced2 fresh-world live8 | SCOPED_NEGATIVE | balanced both `3/8` vs random seed1 `5/8`；stop balanced sampler |
| `A6-O197C` | 50/50 random-seed residual head weight average | DONE | exact lineage；output-mean parity `5.07e-7`；single-model inference |
| `A6-O198C` | weight-average CAL audit | DONE | checkpoint/consumer/finite checks pass |
| `A6-O199C` | baseline/seed1/seed2/average fresh-world live8 | SCOPED_NEGATIVE | average `4/8`，不及 seed1 `5/8` 且 progress 低于 seed2 |
| `A6-G040C` | primary camera whole-object alignment materialization | INVALID_GATE | 632 rows 完整，但 64 rows 的非目标活动 link 不可 exact 复现；不得据此否定 target mask |
| `A6-G040R` | exact target-link membership repair | DONE | 632 mask；64 rows 重算；target alignment max `9.77e-6`；非空/二值通过 |
| `A6-G041C` | base-frame input + target-mask exact join | DONE | 531 TRAIN / 101 CAL；group/hash/split/no-oracle 全通过 |
| `A6-G042C` | concat-mask qpose target vs zero control | SCOPED_NEGATIVE | target `0.652161` vs zero `0.611153` endpoint MAE |
| `A6-G044C` | raw-unit paired CAL | DONE | target-minus-zero `+0.041007 [0.010256,0.071644]`；禁止同编码 traj/physical |
| `A6-G045C` | dual-pool target/null qpose | DONE | target/null 两支 6000-step finite/reload/matched checks 通过 |
| `A6-G047C` | dual-pool raw-unit paired CAL | NO_SIGNAL | target `0.620677` vs null `0.624526`；CI 跨零；不训练 G046 |
| `A6-G048C` | target-local shape+centroid qpose | DONE | target/null 两支 6000-step finite/reload/matched checks 通过 |
| `A6-G050C` | target-local raw-unit paired CAL | SCOPED_NEGATIVE | target `0.658488` vs null `0.626526`；停止 mask encoder |
| `A6-G051C` | qpose vs hand-SE3 label diagnostic | DONE | 632/632 IK；qpose等价解 L2 p90 `4.846`；支持 deployable base-SE3 route |
| `A6-G052C` | grouped base-frame hand SE3 labels | READY | K4 exact join；translation+rotation6D；无 link pose input |
| `A6-G053C` | base-only SE3 proposal model | BLOCKED | waits G052；matched 6k |
| `A6-G054C` | target-local SE3 proposal model | BLOCKED | waits G052；matched 6k |
| `A6-G055C` | raw pose error + IK CAL | BLOCKED | waits G053/G054 |
| `A6-G052C` | grouped base-frame SE3 contract | DONE | 2373 labels；rotation/lineage/no-link-pose-input pass |
| `A6-G053C/G054C` | base-only / target-local SE3 fit | DONE | 6k finite/reload；真实 GPU1/2 并行 |
| `A6-G055C` | raw pose error + IK CAL | SCOPED_NEGATIVE | pose error大；IK reach不是抓取正确性；禁止 physical |
| `A6-G058C` | contact-frame compactness diagnostic | DONE | 94.9% local-z；约8cm retreat；98.1% affordance coverage |
| `A6-G059C` | normal+hinge geometry-frame ceiling | READY | CPU/meta only；决定是否训练 contact proposal |
| `A6-G059C` | normal+hinge geometry-frame ceiling | SCOPED_NEGATIVE | fixed/oracle 3cm12deg `5.0%/10.5%`；不做physical |
| `A6-G060C` | GT affordance top-K contact coverage | READY | contact label与affordance consumer上限 |
| `A6-G060C` | GT affordance top-K contact coverage | DONE / POSITIVE | CAL top4-NMS 3cm/5cm `57.9%/70.7%` |
| `A6-A010C-v1` | fixed8 random-init learnability | INVALID_RELOAD_GATE | model stochastic forward未reset seed；科学fit保留，gate撤销 |
| `A6-A010C-v2` | fixed8 random-init learnability | DONE | loss `2.991->0.0595`；MAE `0.0427->0.00155`；seeded reload 0 |
| `A6-A020C` | clean uniform-combined 3-seed full train | DONE | seeds `20260806/07/08`均7000-step terminal pass；三个fixed `last.pth` reload exact |
| `A6-A030C` | fixed-last CAL + PRED top-K contact consumer | DONE / POSITIVE | ensemble IoU/AUPRC `0.4598/0.6509`；top4-NMS 3cm/5cm `47.6%/51.6%`；paired distance CI全小于0 |
| `A6-G061C` | PRED contact-query K4 supervision contract | DONE / POSITIVE | 2373 assignments；CAL 3cm/5cm `25.7%/34.6%`；paired distance CI全小于0 |
| `A6-G062C` | contact-conditioned base-SE3 fit | SCOPED_NEGATIVE | translation `0.2212m` vs `0.2239m`、rotation `1.673` vs `1.819rad`；两个paired CI均跨0；禁止physical |

## Grasp Factor Revision Queue (2026-08-07)

当前权威affordance语义：`A6-A020C/A030C = current updated`。Initial和`updated_mix_060`尚未进入A6 clean producer；mix060的独立实验只支持可学习性，不支持下游动作收益。

| Run ID | Milestone | Purpose / Comparison | Split | Priority | Status / Gate |
|---|---|---|---|---|---|
| `A6-G063C` | M0 | contact-local orientation mode + IK-equivalent best-of-M诊断 | TRAIN构造/CAL一次评测 | MUST | DONE / POSITIVE；M8 pose与multi-IK oracle通过 |
| `A6-A031C` | M0 | initial/updated/mix060与A000 clean membership exact lineage | TRAIN/CAL audit | MUST | DONE；三语义lineage/公式/覆盖通过，A030仍为updated |
| `A6-G064C` | M1 | mode/residual、IK-set、sparse keypose supervision contract | TRAIN/CAL labels | MUST | DONE；2373 labels，8-mode/IK/keypose合同通过 |
| `A6-G065C` | M1 | G062复现 vs mode+residual vs +set/FK consistency | TRAIN/CAL | MUST | SCOPED_NEGATIVE / ORACLE_ONLY；best-of-8正向但selected top-1退化 |
| `A6-G066C` | M2 | fixed-candidate S0 mode-logit vs S1 calibrated likelihood vs S2 IK/FK/planner selector | TRAIN calibration/CAL一次评测 | MUST | READY；仅scoring/realization诊断，physical未解锁 |
| `A6-A032C` | M2 | matched one-seed PRED_INITIAL/PRED_MIX060 screen；UPDATED复用A030 | TRAIN/CAL | MUST | RUNNING；sanity双通过，formal GPU0/1并行 |
| `A6-G067C` | M2 | predicted-contact full proposal；相同总candidate预算 | CAL | MUST | BLOCKED on G066 selected top-1 gate |
| `A6-A033C` | M2 | UPDATED/INITIAL/MIX060/[INITIAL,MIX060] consumer utility | TRAIN/CAL | MUST | BLOCKED on A032 and G066 winning selector；四condition同wave并行 |
| `A6-A033U` | appendix | ensemble mean+uncertainty consumer | TRAIN/CAL | NICE | CONDITIONAL；仅双通道已有utility且matched多seed producer可用时触发 |
| `A6-A034C` | M3 | affordance winning condition formal 3-seed freeze | TRAIN/CAL | CONDITIONAL | 仅A033下游paired evidence触发；否则保留A030 |
| `A6-G068C` | M3 | fresh-world grasp-only physical screen | CAL live8 | MUST | BLOCKED on G067 + affordance freeze；strict grasp先非零再补seed |
| `A6-G069C` | M4 | grasp-only selector vs transition-aware selector + O185 | CAL live8 | MUST | BLOCKED on G068 stable strict grasp |
| `A6-J000C` | M4 | frozen modular deployable baseline | CAL live | MUST | BLOCKED on G068,A033/A034 |
| `A6-J050C` | M4 | shared Stage-1 independent multi-head representation | TRAIN/CAL | MUST | BLOCKED on J000 |
| `A6-J100C` | M4 | modular vs shared + parameter-matched control | CAL | MUST | BLOCKED on J050 |
| `A6-J150C` | M4 | transition-aware global handoff optimization | CAL live | MUST | BLOCKED on G069,J100 |
| `A6-J200C` | M4 | bounded joint fine-tune/DAgger | TRAIN only | CONDITIONAL | 仅J150有明确合法handoff residual时新revision |

多GPU policy：当前A032占GPU0/1；G066 S0/S1/S2及parity可使用其余GPU并行。A032/G066 terminal后，A033四affordance condition与G067四proposal condition组成最多8卡的下一wave。禁止为占卡重训已有updated baseline、提前跑额外seed或增加mix070/mix080。

停止规则：G066若不能缩小selected-vs-oracle gap并保持相对G062 pose不退化，则G067/physical继续阻断；G067在相同candidate预算下无selected top-1改善则不physical；G068 strict grasp仍为0则不接O185；A033若mix/双通道无下游收益则A030 updated保持权威producer；J050/J100若牺牲contact或strict grasp则回退J000模块化系统。

明确失效的旧 live runs：O131、O135/O136、O140/O141-live、O146/O147-live、O148、O149、O153-live、O156-live、O163。O127/O128 checkpoints 与对应 offline/CAL 结果仍可用于 O171C。

O171--O191 的 live runner 对同一 target 只创建一个 `ViewPcdCapturer/DemoWorld`，多个 arm 仅重设 qpos/qvel。O185 checkpoint 在 O188 排第三为 `3/8`、在 O191 排第一为 `5/8`，target 1/2 差异达到约 `0.35/0.12` progress，说明 arm 顺序污染足以改变结论。当前不能再引用任何旧 paired live 比较。

保留范围：D180 数据、O181/O182/O185/O189 checkpoint、O183/O186/O190 CAL，以及每个旧 runner 的首 arm 原始事实。O188 首 arm O127 `2/8`、O191 首 arm seed1 `5/8` 都需在同一 fresh-world 协议中重评后才能比较。

O192C 不使用任意数值硬阈值：协议 gate 是每 arm 独立创建/configure/close world 和相同 checkpoint lineage；first/last trace 差异用于判断是否还存在其他全局状态。只有该差异不再呈现旧 run 的大幅顺序效应，才进入 O193C。

O193C 使用统一 650 calls/K8/opening-angle stop，现场重跑 O127、两个 residual seed 和 repeat-last；按 paired progress/contact/task 判断方向，不设预知性能提升 gate。

O192C terminal artifact：`results/a6_o192c_fresh_world_order_audit_v1/summary.json`。两个 target 的同一 O185 checkpoint first/last 均 exact trace match，证明 fresh-world 修复消除了已发现的 arm-order 污染；O193C 可执行。

O193C terminal artifact：`results/a6_o193c_fresh_world_fixed_budget_live8_v1/summary.json`。baseline/seed1/seed2/repeat task 为 `2/5/3/0`；residual 两 seed 相对 baseline progress 均值为 `+0.0713/+0.0473`，contact 均值没有退化。方向 promising，但 seed gap 要求先执行 O194--O196 的 equal-exposure sampling 稳定性实验。

O196C terminal artifact：`results/a6_o196c_balanced_recovery_residual_live8_v1/summary.json`。balanced 两 seed 都是 `3/8`，相对 random seed1 task 均 `-2`；均衡采样缩小 seed gap 但牺牲性能，标记 scoped negative。随机 exposure 与 snapshot live call 的相关诊断为 `-0.018`，不支持 late-state weighting。下一步只执行 O197--O199 的线性 head weight average。

O199C terminal artifact：`results/a6_o199c_recovery_residual_weight_average_live8_v1/summary.json`。average 为 `0.26895 / 4/8`，没有超过 seed1 的 `0.31365 / 5/8`，并且 progress 明确低于 seed2。停止 sampler/seed fusion；冻结 O185 seed1 为 provisional operation 候选，O127 为 fallback，G000C READY。

## G000C Grasp Label Contract

`results/a6_g000c_grasp_label_set_contract_v2/summary.json` terminal pass：TRAIN/CAL 为 `557/102` observation groups、`26,597/4,919` source trajectories、`2,099/387` selected K4 teachers。所有 endpoint、relative roundtrip、open-vs-closed、presence、reload、split 和 forbidden-field checks 通过；qpose sidecar 物理排除 stored qpath。v1 的 bitwise float gate 已由 `<=1e-6` parity 修复，v1/v2 标签 NPZ hash 相同。下一步只执行 G005C GT interface pair。

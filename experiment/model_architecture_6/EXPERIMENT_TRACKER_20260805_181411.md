# Architecture 6 Experiment Tracker

- **Canonical plan**：`EXPERIMENT_PLAN.md`
**状态语义**：`READY` 可立即执行；`TODO` 依赖满足后自动变为 READY；`BLOCKED` 当前依赖未满足；`CONDITIONAL` 仅在 plan 指定信号出现时运行；`DONE` 有 terminal artifact；`REJECTED` 合同有效但该分支未通过 gate。

## Reused Evidence

| Evidence | Status | Artifact | Architecture 6 decision |
|---|---|---|---|
| Canonical qpos executor | DONE | `force_admittance_collect/replay_contract.py` | direct-qpos 主线冻结，不重建 collection controller |
| Shared replay C020 | DONE | `model_architecture_2/results/canonical_shared_replay/C020/validation_summary.json` | 20/20，零错误 |
| Shared replay C021 | DONE | `model_architecture_2/results/canonical_shared_replay/C021/validation_summary.json` | 57/57，12 targets，零错误 |
| Split lineage | DONE | `model_architecture_5/results/a5_c000sr_split_lineage_v1/` | 复用 source partitions 和 SAME_TEST semantics |
| Terminal observation mask | DONE | `model_architecture_5/results/a5_c010f_terminal_mask_v1/` | 复用 2-row mask；command 不变 |
| Label/data contract | DONE | `model_architecture_5/results/a5_c020_label_contract_v1/` | 39,377 trajectories；split counts 冻结 |
| Dynamic observation DYN8 | DONE | `model_architecture_5/results/a5_c030_dyn8_observation_v3/` | producer 可复用 |
| Dynamic observation DYN64 | DONE | `model_architecture_5/results/a5_c031_dyn64_resource_v1/` | 2 render workers |
| Data prefreeze | DONE | `model_architecture_5/results/a5_c040p_prefreeze_v2/` | 复用 hashes；删除 C011 dependency |
| Old-split affordance compatibility reference | DONE | `jointTrain_new/runs/stage1_bestview_1024_multiview_updated/best.pth` | 仅load/shape smoke；禁止正式forward，PRED由A000-A030重训 |
| Architecture 1 affordance intervention | DONE | `model_architecture/EXPERIMENT_RESULTS.md`、E230/E240 summaries | condition会改变action但未改善task；必须做ZERO/PRED/GT和joint non-regression |
| Stage1 updated optimization | DONE | `jointTrain_new/experiment/stage1_optimize/refine-logs/EXPERIMENT_RESULTS.md` | replay-balanced -0.0246；A020固定uniform combined且不做常规sweep |
| Architecture 2 path/planner ceiling | DONE | exact P010/QF010 markers（plan E19） | stored qpath 100%；GT SE(3)+fresh planner 91.67%；不等价于terminal-qpose-only，后者由G005新测 |
| Architecture 3 three-state/qpose-qpath evidence | DONE | GQ015R summary、`model_architecture_3/EXPERIMENT_PLAN.md` A-E表 | open target与closed state分离；旧qpose arm含GT pregrasp，只作新公平比较动机 |
| Architecture 4 qpath control | DONE | `a4_m0_a003r_pathfix_v1/A4-003/summary.json` | corrected stored/candidate qpath为68/68和66/68 |
| Architecture 4 candidate/live components | DONE | live observation、target mask、SE3/set loss、Curobo/selector summaries | 组件代码可复用；不继承旧learned winner |
| Architecture 4 alternative fits | DONE | `a4_b12_alternative_matched_fit_s1_terminal_recovery_v1` | RPAR/KP/DP3 train fit完成；online架构排名未建立 |

## M0: Data and Adapter

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-D000 | evidence/path freeze | E1-E28定向阅读 + evidence lock + `path_config.py` entries | metadata only | review逐项写可信/禁止外推claim与symbols；hashes persisted；zero new full replay | MUST | reused evidence | DONE | terminal summary passed；remaining D010 adapter roundtrip + A000 membership/provenance |
| A6-D010 | exact adapter roundtrip | authoritative operation adapter | 2 fixed-gate trajectories | normalize/decode/stitch max error <=1e-6；2/2 replay parity | MUST | D000 | DONE | max error 1.192e-7；source/decoded success与parity均2/2；不是 executor requalification |
| A6-D020 | sample/normalizer freeze | 16-anchor H32/K8 index | DYN8/A5_TRAIN stats | 64 fixed chunks；mask/split/hash exact；shared normalizer | MUST | D010 | DONE | 559 samples；26,605 trajectories；36,714,651 rows；64 chunks；all gates pass |
| A6-D030 | DYN64 materialization | live 1024-point keyframes | 64 train targets | finite；2 workers；restartable；source hash exact | MUST | D010 | DONE | probe2/2 8.09s；64 targets/1,024 frames；all gates pass |

## M0A: Split-Safe Affordance Baseline

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-A000 | affordance membership/provenance | A5 sample IDs -> Stage1 primary/aug manifest | A5_TRAIN/A5_CAL metadata | 559/102 sources；zero overlap/unjoined/duplicate/final-label read；init provenance | MUST | D000 | BLOCKED | v1 consumer join bug inflated unjoined to29；correct lineage exposes 2 missing TRAIN primary rows；A010 locked |
| A6-A000R | corrective metadata audit | accepted_samples source index -> Zarr source_replay_id | same frozen metadata | remove false joins；persist exact missing rows；zero old replay_split/content read | MUST | A000 v1 | DONE | terminal data_contract_failure：TRAIN 557/559，CAL 102/102；missing TRAIN IDs 70/251；random init required |
| A6-A010 | fixed-batch fit | Point-M2AE dual updated | 8 A5_TRAIN sources | 2k；MAE<=0.02；AUPRC>=0.95；finite grad；reload<=1e-6 | MUST | A000 | BLOCKED | no E11 supervised init |
| A6-A020 | split-safe full train | fixed uniform-combined protocol, 3 seeds | A5_TRAIN only | 100x70 steps；fixed last；zero CAL/heldout read；all seeds | MUST | A010 | BLOCKED | 3 independent GPUs |
| A6-A030 | calibration/consumer freeze | three A020 last checkpoints | A5_CAL primary + DYN8 live | finite metrics；[B,1024] point alignment；reload/live repeat<=1e-6；no selection | MUST | A020 | BLOCKED | PRED source for O220/G130 |

## M1: Fixed-Batch Operation Fit

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-O000A | shared fit contract audit | label/mask/normalizer + plan-vs-runner config/input | fixed64 metadata only | exact labels/masks/normalizer；enumerate all config/input drift；zero training | MUST | O010/O020/O030 invalid | DONE | labels/masks/normalizer/unique inputs pass；runner/input drift confirmed；A6-INPUT-v1.1 applied |
| A6-O000B | shared input materialization sanity | qpos/qvel/command/contact/task + XYZ/mask/ZERO | fixed64 TRAIN only | exact schema；finite；causal；URDF metadata；forbidden fields zero | MUST | O000A + revision | BLOCKED | v1 implementation crash：non-target joint missing limit incorrectly rejected；no terminal science |
| A6-O000BR | parser repair input sanity | same frozen A6-INPUT-v1.1 | same fixed64 | target joint remains strict；non-target missing limit allowed | MUST | O000B crash | BLOCKED | v2 implementation crash：short contact_feedback not masked；no terminal science |
| A6-O000BR2 | contact-mask repair input sanity | same frozen A6-INPUT-v1.1 | same fixed64 | short/missing contact -> zero + availability0 | MUST | O000BR crash | DONE | 64行/全部shape/finite通过；45 absent严格zero+mask0，19 available与mask一致；forbidden reads零 |
| A6-O010 | memorization | O-MLP-ABS | fixed 64 chunks | 2k steps；MAE <=1e-3；loss >=100x decrease；reload <=1e-6 | MUST | D020,D030,O000A repair | BLOCKED | v1 invalid：lr/loss/shared encoder/input contract drift；not scientific negative |
| A6-O010R | corrected memorization | O-MLP-ABS + shared A6-INPUT-v1.1 encoder | fixed 64 chunks | hidden256/dropout0.1；AdamW 1e-4/1e-6；effective batch64；normalized L1；2k；same gate | MUST | O000BR2 | REJECTED | valid training-fit negative：final 0.025114、26.94x，差于repeat-last 0.010378；grad/reload/pilot/contracts均通过；仅拒绝该MLP配置 |
| A6-O020 | memorization | O-PAR-ABS | same 64 chunks | same gate | MUST | D020,D030,O000A repair | BLOCKED | v1 invalid：lr/loss/hidden/dropout/shared input drift；not scientific negative |
| A6-O020R | corrected memorization | O-PAR-ABS + shared A6-INPUT-v1.1 encoder | same fixed 64 chunks | same hidden256/dropout0.1/AdamW/L1/seed/batch/2k gate；parallel learned queries | MUST | O000BR2,O010R terminal | REJECTED | valid training-fit negative：final 0.030015、24.35x；比repeat-last差189%、比O010R差19.5%；contracts/parity均通过 |
| A6-O030 | memorization | O-CAUSAL-ABS | same 64 chunks | same gate | MUST | D020,D030,O000A repair | BLOCKED | v1 invalid：lr/loss/hidden/dropout/shared input drift；not scientific negative |
| A6-O030R | corrected memorization | O-CAUSAL-ABS + shared A6-INPUT-v1.1 encoder | same fixed 64 chunks | teacher forcing train/autoregressive eval；其他hidden256/dropout0.1/AdamW/L1/seed/batch/2k gate相同 | MUST | O000BR2,O020R terminal | REJECTED | valid training-fit negative：AR final 0.117449、7.29x、比repeat-last差1032%；teacher train loss 0.016597需同语义audit |
| A6-O000C | M1 all-branch fit attribution | reload O010R/O020R/O030R checkpoints | same fixed64 TRAIN only | same-semantic MAE；causal TF-vs-AR；per-horizon/repeat structure；embedding separability；zero training/replay | MUST | all corrected M1 terminal | DONE | encoder 64 unique/rank63；all models worse repeat；causal TF 0.007464 vs AR 0.117343=15.72x；A6-FIT-v1.2 |
| A6-O010S | revised-budget memorization | O-MLP-ABS scratch under A6-FIT-v1.2 | same fixed64 | 6k；2k reproduction；MAE<=1e-3；100x；reload<=1e-6；same scientific config | MUST | O000C + revision | REJECTED | 2k parity error0；6k final0.014016/48.27x，仍比repeat-last差35.1%；valid scoped training-fit negative |
| A6-O020S | revised-budget memorization | O-PAR-ABS scratch under A6-FIT-v1.2 | same fixed64 | same 6k/reproduction/gates | MUST | O010S terminal | REJECTED | 2k parity error0；6k final0.028086/26.02x，仅较2k改善6.4%；比repeat-last差170.6% |
| A6-O030S | revised-budget memorization | O-CAUSAL-ABS scratch under A6-FIT-v1.2 | same fixed64 | same 6k/reproduction/gates；TF train/AR eval | MUST | O020S terminal | RUNNING | 当前唯一GPU授权；复用O030R pilot；scratch 2k parity + fixed6k AR gate + TF metric |

## M2: DYN64 Operation Screen

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-O100 | 6k offline fit | O-MLP-ABS | DYN64 train/CAL screen | first/endpoint MAE >=50% better than repeat baseline；p95 reported | MUST | O010,D030 | BLOCKED | one seed |
| A6-O110 | 6k offline fit | O-PAR-ABS | same | same gate | MUST | O020,D030 | BLOCKED | one seed |
| A6-O120 | 6k offline fit | O-CAUSAL-ABS | same | same gate | MUST | O030,D030 | BLOCKED | one seed |
| A6-O130 | predicted trajectory replay | all passing candidates | fixed train/CAL episodes | >=90% of exact-replay success；endpoint/p95 finite | MUST | O100/O110/O120 passing arms | BLOCKED | recorded observations only |
| A6-O140 | live operation-only screen | all passing candidates | exact-paired train/CAL | train target-macro >=80%；wrong-way <=5%；latency | MUST | O130 | BLOCKED | select decoder, not final claim |

## M3: Representation and Training Strategy

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-O200 | action representation | winner ABS vs start-delta | DYN64 train/CAL | same model/data/steps；offline+closed-loop exact-paired | MUST | O140 | BLOCKED | one-variable comparison |
| A6-O210 | sampling ablation | motion-balanced vs raw-uniform | DYN64 train/CAL | task/endpoint gain；quantify hold shortcut | MUST | O140 | BLOCKED | same winner |
| A6-O220 | affordance intervention | matched-train ZERO/A030-PRED/GT + frozen-PRED sensitivity | DYN64 train/CAL | task, progress, offline error；utility vs consumer sensitivity分开 | MUST | O200,O210,A030 + O230/O240 if triggered | BLOCKED | GT diagnostic only |
| A6-O225 | affordance x decoder interaction | runner-up decoder under A030-PRED | DYN64 train/CAL | trigger abs(PRED-ZERO)>=5pp；same seed/budget；rerank decoder | CONDITIONAL | O220 trigger | BLOCKED | zero MECHDEV/final read |
| A6-O230 | endpoint loss | winner + endpoint weighting | DYN64 train/CAL | run only if offline good but predicted replay fails | CONDITIONAL | O130 failure pattern | BLOCKED | no sweep |
| A6-O240 | scheduled sampling | winner + frozen schedule | DYN64 train/CAL | run only if replay passes but live fails | CONDITIONAL | O140 failure pattern | BLOCKED | no DAgger yet |

## M4: Formal Operation Selection

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-O300 | full training | frozen operation candidates, 3 seeds | A5_TRAIN | fixed 6k/config；all seeds; no best seed | MUST | M3 decision + O225 if triggered | BLOCKED | single GPU per job |
| A6-O310 | calibration audit | same checkpoints | A5_CAL | artifact completeness；train/CAL gap；candidate freeze | MUST | O300 | BLOCKED | no outcome-driven retraining |
| A6-O320 | intermediate method selection | frozen candidates, 3 seeds | A5_CAL | exact-paired；>=5pp and CI lower >0 for CAL superiority | MUST | O310 | BLOCKED | choose module；zero MECHDEV/final read |
| A6-O330 | result/claim audit | frozen artifacts | no new rollouts | pairing/hash/claim limits | MUST | O320 | BLOCKED | unlock grasp |

## M5: Grasp and Affordance

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-G000 | qpath/qpose label-set contract | same-initial-state K=4 groups + terminal sidecar | A5_TRAIN + fixed CAL metadata | trajectory lineage；qpose==qpath[-1]；open/closed separation；producer/consumer split | MUST | O330 | BLOCKED | no outcome/test read |
| A6-G005 | interface oracle pair | GT-TRAJ vs GT-QPOSE+online planner | fixed CAL screen | GT-TRAJ path contract100%/task>=90%；GT-QPOSE planner contract>=90%/task>=80% or <=15pp gap；L64 exact-paired | MUST | G000 | BLOCKED | qpose consumer zero stored pregrasp/qpath |
| A6-G010 | fixed-batch fit | ZERO G-TRAJ Kx64x7 + presence | fixed 8 observation groups | 2k；path/endpoint MAE<=1e-3；100x；presence100%；grad/reload | MUST | G005 trajectory route pass | BLOCKED | representation isolated |
| A6-G020 | fixed-batch fit | ZERO G-QPOSE Kx7 + presence | same groups | 2k；qpose MAE<=1e-3；100x；presence100%；grad/reload；single-segment consumer | MUST | G005 qpose route pass | BLOCKED | zero GT pregrasp/qpath |
| A6-G030 | comparison parity audit | G-TRAJ vs G-QPOSE | DYN64 plan | same encoder/ZERO/K/exposure/seed/close/settle/op；params/resources | MUST | G010,G020 or surviving arm | BLOCKED | freezes commands |
| A6-G100 | matched fit | both routes or G005 survivor | DYN64 train/CAL | offline fit, validity/planner coverage, no outcome checkpoint select | MUST | G030 | BLOCKED | single seed |
| A6-G110 | physical representation screen | surviving route(s) + frozen operation | same episodes | path tracking/planning/reach/contact/retention/task/latency | MUST | G100 | BLOCKED | exact-paired only if both survive |
| A6-G120 | representation audit | TRAJ vs QPOSE or single-route freeze | existing G110 outcomes | >=5pp+CI/tie-break；single route no comparison claim | MUST | G110 | BLOCKED | freeze one interface |
| A6-G130 | affordance intervention | matched-train ZERO/A030-PRED/GT + frozen-PRED sensitivity | DYN64/CAL | >=5pp and CI lower >0 to claim utility；sensitivity separate | MUST | G120,A030 | BLOCKED | task decides routing |
| A6-G135 | affordance x representation interaction | nonwinner route under A030-PRED | DYN64/CAL | trigger abs(PRED-ZERO)>=5pp and both routes survived；rerank representation | CONDITIONAL | G130 trigger | BLOCKED | zero MECHDEV/final read |
| A6-G140 | joint-affordance fit | G-JOINT-AFF on winner interface | A5_TRAIN fixed/full sanity | affordance+grasp fit；gradients；reload；operation unchanged | CONDITIONAL | G130 utility/headroom + G135 if triggered | BLOCKED | otherwise NOT_APPLICABLE_BY_GATE |
| A6-G150 | frozen vs joint-aff physical screen | G-WIN-PRED vs G-JOINT-AFF | DYN64/CAL | reach/contact/retention/task；operation non-regression | CONDITIONAL | G140 pass | BLOCKED | exact-paired |
| A6-G200 | formal grasp train | passing candidates, 3 seeds | A5_TRAIN/CAL | fixed configs, no best seed | MUST | G130 + G135/G150 if triggered | BLOCKED | no operation changes |
| A6-G210 | intermediate grasp selection | frozen candidates, 3 seeds | A5_CAL | exact-paired full-stage metrics | MUST | G200 | BLOCKED | choose module；zero MECHDEV/final read |
| A6-G220 | result/claim audit | frozen artifacts | no new rollouts | representation+affordance pairing/hash/claim limit | MUST | G210 | BLOCKED | unlock integration |

## M6: Joint Optimization

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-J000 | deployable pipeline baseline | frozen operation/grasp with independent O/G affordance gates | A5_TRAIN/CAL screen | complete task + stage metrics + forbidden fields | MUST | G220 | BLOCKED | frozen modules |
| A6-J100 | supervised joint training, 3 seeds | shared action+grasp；per-head affordance gates frozen | A5_TRAIN/CAL | full task gain；operation-only regression <=2pp；no best seed | MUST | J000 | BLOCKED | joint aff loss only if G-JOINT-AFF promoted |
| A6-J200 | bounded task fine-tune | J-TASK-FT | A5_TRAIN only | conditional task gain；zero heldout reads | CONDITIONAL | J100 attribution | BLOCKED | no automatic RL/DAgger |
| A6-J210 | complete-system selection | all frozen complete-system candidates, 3 seeds | A5_MECH_DEV once | joint >=5pp and CI lower >0；operation non-regression | MUST | J100 + J200 if triggered | BLOCKED | only MECHDEV read；choose one system |
| A6-J220 | deployment audit | unique winner | no new outcomes | zero oracle fields；hash/config complete；zero E11 ancestor | MUST | J210 | BLOCKED | unlock final |

## M7: Final Evaluation

| Run ID | Purpose | System / Variant | Split | Metrics / Gate | Priority | Depends On | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A6-F000 | same-target robustness | unique frozen winner | A5_SAME_TEST once | complete task + available stage/operation diagnostics, CI | MUST | J220 | BLOCKED | no source trajectories |
| A6-F010 | target-disjoint generalization | same winner | A5_TARGET_TEST once | complete task + recorded-start operation-only, CI | MUST | J220 | BLOCKED | no retraining afterward |
| A6-F020 | final aggregation | frozen artifacts | no new rollout | claim table, failure strata, resource table | MUST | F000,F010 | BLOCKED | terminal Architecture 6 report |

## Conditional Complexity

| Run ID | Trigger | Variant | Status | Rule |
|---|---|---|---|---|
| A6-X010 | stable conditional action multimodality + mode averaging | Diffusion qpos chunk | CONDITIONAL | cannot block deterministic winner |
| A6-X020 | independent need for Cartesian representation | SE(3)+IK system | CONDITIONAL | local GT adapter gate only |
| A6-X030 | scheduled sampling fails and legal train-only teacher exists | DAgger | CONDITIONAL | requires new planning revision |
| A6-X040 | frozen encoder proven bottleneck | larger 3D backbone | CONDITIONAL | matched decoder/data/budget |

## Current Authorization

O010S/O020S exactly reproduce their 2k predecessors. O020S reaches only 0.028086/26.02x at 6k, improves 6.4% from 2k and remains 170.6% worse than repeat-last; it is a valid final scoped PAR negative. O030S is the sole READY GPU run. After O030S, perform M1 revised-budget review before any DYN64. Physics, heldout and conditional complexity remain blocked.

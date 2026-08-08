# Architecture 6 Historical Evidence Review

- Run: `A6-D000`
- Scope: E1-E28 only; metadata/indexing review, CPU-only.
- Scientific status: no new replay, oracle, controller reconstruction, training, or full replay was run.
- Hash policy: file hashes are SHA-256; directory hashes are the sorted-file manifest digest recorded in `source_hashes.sha256`.

| ID | Read path(s) | Reusable code symbol(s) | Trusted claim | Explicit non-claim / routing limit |
|---|---|---|---|---|
| E1 | `../../../force_admittance_collect/replay_contract.py`; `replay_quality_gate.py` | `replay_contract`; `replay_quality_gate` | `joint_command_qpos_repaired`, absolute 9D qpos, one waypoint per physics step, frozen drive/friction/finger settings | Does not justify torque, Cartesian, or controller reconstruction |
| E2 | `../model_architecture_2/results/canonical_shared_replay/C020/validation_summary.json` | shared replay validator | 20/20 fixed-gate replay passed with zero execution errors | Historical replay is not a new A6 adapter qualification |
| E3 | `../model_architecture_2/results/canonical_shared_replay/C021/validation_summary.json` | shared replay validator | 57/57 across 12 targets passed | No A6 split/generalization claim |
| E4 | `../model_architecture_5/results/a5_c000sr_split_lineage_v1/` | split lineage/manifest readers | 816 accepted samples, 283 targets, 248 shapes; no source target/shape leakage; SAME_TEST evaluator-only | No re-splitting or importing unlisted old conclusions |
| E5 | `../model_architecture_5/results/a5_c010f_terminal_mask_v1/` | terminal observation mask consumer | Only two operation terminal observation rows are masked; command rows and operation start remain | Must not alter command labels or delete trajectories |
| E6 | `../model_architecture_5/results/a5_c020_label_contract_v1/summary.json` | label/phase/source lineage contract | 39,377/39,377 trajectories valid; open grasp goal is separate from closed operation state | No new label semantics inferred |
| E7 | same E6 artifact | split membership fields | A5_TRAIN 26,605; CAL 4,919; MECH_DEV 4,091; TARGET_TEST 3,762 | No trajectory-level random reshuffle |
| E8 | `../model_architecture_5/results/a5_c030_dyn8_observation_v3/` | dynamic observation producer | State restore, target mask, 1024-point render and repeat render are consistent | A6 adapter still needs its own shape/hash sanity |
| E9 | `../model_architecture_5/results/a5_c031_dyn64_resource_v1/summary.json` | materialization worker layout | 64 targets and all strata generated; 2 workers is the recorded recommendation | No new worker benchmark or >2-worker default |
| E10 | `../model_architecture_5/results/a5_c040p_prefreeze_v2/` | prefreeze field/hash readers | Split, mask, label, render and hashes can be frozen together | The old C011 dependency is explicitly discarded |
| E11 | `../../runs/stage1_bestview_1024_multiview_updated/best.pth` | `load_stage1_model` interface | A loadable 1024-point updated-affordance compatibility reference exists | Old split; load/shape smoke only, never formal A6 forward or selection |
| E12 | `../stage1_optimize/refine-logs/EXPERIMENT_RESULTS.md` | Stage1 uniform-combined protocol | Updated uniform baseline 0.4833 +/- 0.0036; replay-balanced was -0.0246; ordinary sweeps missed +0.008 | A020 reuses fixed protocol; no new sweep |
| E13 | `../stage1_optimize/AFFORDANCE_DATA_QUALITY.md` | Stage1 label/context analysis | Geometry-only labels have irreducible cross-replay conflicts | Affordance is a geometry prior; outcome context belongs in grasp head |
| E14 | `../model_architecture_3/results/aq040_pred_affordance_offline_audit/summary.json` | predicted/GT/zero affordance adapter | Frozen predicted interface runs and top-64 positive coverage is 1.0 | No utility claim; A6 must measure physical ZERO/PRED/GT intervention |
| E15 | `../../../force_admittance_collect/run_collection.py` | `grasp_plan_qpath` and frozen grasp-plan support | Dataset contains traceable executable grasp-qpath teacher | `q_path_terminal_open` and `q_operation_start_closed` remain distinct |
| E16 | `../model_architecture_3/gq130_residual_model.py`; `run_gq130_train_smoke.py` | residual/ranking model and smoke entry point | Candidate residual/ranking and affordance conditioning are reusable starting code | No old promotion/negative result is inherited |
| E17 | Architecture 1 conclusion/results and E230 summaries | DP3 affordance conditioning/evaluators | Global affordance condition changes action, but old GT map/contact did not improve online physics | Do not make global heatmap concatenation the sole route |
| E18 | Architecture 1 E240 summary and intervention scripts | shared auxiliary affordance heads | Spatial signal can be learned while success fell 12.5pp and grasp 37.5pp | Offline affordance fit is not action/task improvement; keep frozen control |
| E19 | planned-qpath plan; P010; QF010 markers | stored qpath and fresh planner contracts | Stored qpath + GT operation reached 100%; GT SE(3)+fresh planner reached 91.67% with full planner/frame/execution coverage | Not a terminal-qpose-only result; G005 must remeasure that interface |
| E20 | v2 evaluator, validation suite, planned-path test | `planned_qpath_schedule` and contract tests | Scheduler, frame/terminal tolerance and exact-pair selection are reusable | Do not copy old queue/routing or oracle evaluator branch |
| E21 | `../model_architecture_3/results/gq015r_three_state_label_audit/summary.json` | three-state label audit | 27,872/27,872 valid; open terminal differs from closed operation start (mean/median 0.01893/0.01752) | qpose teacher must be open terminal |
| E22 | Architecture 3 plan and v2 `oracle_joint_goal_planner` | qpose/qpath comparison branch | Old predicted-qpose arm had GT pregrasp leakage; candidate qpath was 66/68 | 57/68 and 66/68 are not A6 baselines or winners |
| E23 | `../model_architecture_4/results/a4_m0_a003r_pathfix_v1/A4-003/summary.json` | qpath executor layout | Corrected stored qpath 68/68 and candidate qpath 66/68; four-worker executor available | Learned A6 comparisons still require fresh exact pairing |
| E24 | A4 live observation and target-mask summaries | live observation/mask producer | 1024-point live observation, mask, state sensitivity and zero read contracts pass | A6 batch adapter needs its own sanity |
| E25 | A4 candidate head/token/representation/loss and sanity summaries | candidate-token encoder, 6D rotation codec, set matching | Fixed K=4 SE(3) candidate path is learnable and consumable | Fixed-four-label loss cannot be reused for variable qpath/qpose labels |
| E26 | A4 Curobo coverage, selector and summaries | Curobo telemetry; `select_shortest_joint_path` | 32/32 candidates planner-consumable; selector 8/8 deterministic | Planner success is not contact, retention, or task success |
| E27 | A4 RPAR/KP/DP3 strategy files and fit summary | strategy heads and training skeletons | All three matched fits had finite gradients, strict reload and zero heldout read | No online architecture ranking or checkpoint performance inherited |
| E28 | v2 `stage_fsm.py`; A4 FSM/progress summaries | FSM/progress producer/consumer | Observable FSM matched phase oracle 204/204 on the old contract; live progress components passed | No new-split or object-replacement performance claim |

## Decision and remaining work

Decision: `A6-D000` passes as an evidence/path freeze. The only newly authorized runs are `A6-D010` and CPU-only `A6-A000`, subject to the plan gates. All later runs remain blocked by their declared dependencies. Missing/renamed source paths are not scientific failures; they are resolved by the repository-root path convention above.

Remaining work: perform D010 adapter roundtrip and A000 membership/provenance audit; do not launch replay, training, GPU validation, or full materialization in D000.

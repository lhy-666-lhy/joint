# Architecture 6 Grasp Bottleneck and Idea Discovery Report

**Date**: 2026-08-07  
**Scope**: `jointTrain_new/experiment/model_architecture_6`  
**Pipeline**: repository audit -> literature survey -> first-principles diagnosis -> sequential idea generation -> novelty pre-check  
**Validation status**: degraded before external jury; the configured Codex/manual reviewer tools are unavailable in this runtime, so rankings below are evidence-based internal recommendations, not cross-model verdicts. No pilot was launched.

## Executive conclusion

The dominant grasp-stage bottleneck is not simply "generating a grasp pose" and it is no longer mainly "finding where to grasp." Architecture 6 has learned a useful contact-location signal, but has not learned how to convert a contact into one of several physically equivalent and kinematically realizable hand-orientation/IK modes. The more precise bottleneck is:

> **contact-anchored, multimodal grasp realization under sparse and incomplete teacher sets, followed by a stable grasp-to-operation handoff.**

The current baseline is not a per-point grasp-pose regressor. `GraspProposalBase` globally pools the point cloud and robot state, appends four learned global queries, and directly regresses four complete trajectories, terminal joint poses, or SE(3) poses. The later G062 model is contact-conditioned, but still maps each frozen contact query to one deterministic translation offset and one rotation-6D output. Thus only the latest branch is point/contact anchored, and even it is not a multimodal per-point action distribution in the Where2Act/Contact-GraspNet sense.

The user's two proposed directions have different answers:

1. **Stage-1 joint affordance + pose + qpose prediction has not been tried in that form.** A020 predicts only per-point affordance. The older Stage-1 co-output experiment jointly predicts initial and updated affordance, not grasp pose or robot qpose. Architecture 6 grasp heads are downstream global proposal heads. Moving grasp realization to Stage 1 is promising only if it preserves multimodality and kinematic equivalence; adding a single qpose L1 head is likely to repeat the current failure.
2. **Direct trajectory fitting has already been tried.** G010 predicts `K x 64 x 7` joint trajectories; G020 predicts `K x 7` terminal qposes followed by online planning; G052-G055 regress task-space hand SE(3). Their failure rules out the current monolithic deterministic/set-regression formulations, not all trajectory learning. A contact/mode-conditioned generative trajectory or sparse phase-keypose model remains untested.

## What was actually tried

| Route | Representation and conditioning | Result | Interpretation |
|---|---|---|---|
| A6-G010C | Global point/state encoder -> K=4 full `64 x 7` joint trajectories | Training loss `1.217 -> 0.343`; learned physical strict grasp/task `0/8, 0/8` | Direct joint-trajectory fitting was tried; offline fit did not transfer physically |
| A6-G020C | Global point/state encoder -> K=4 terminal joint qposes -> online planner | Training loss `0.789 -> 0.228`; planner coverage `8/8`; physical strict grasp/task `0/8, 0/8` | Planner availability is not the main failure; predicted goal modes are wrong |
| G031-G038 | GT affordance, frame changes, normalization, multi-view/same-target diagnostics | Same-target is easier; target-disjoint remains poor | There is a generalization/observability gap, not just an optimizer failure |
| G042-G050 | Target-mask concat, dual-pool, target-local geometry | Target mask no better than null/base controls | A stronger point encoder alone does not repair the output representation |
| G052-G055 | Base-frame task-space SE(3), translation + rotation-6D, K4 set matching | `0.224 m`, `1.819 rad`, 3 cm/12 deg `0.50%` | Switching from qpose to deterministic SE(3) does not resolve multimodality |
| G058-G059 | Contact frame from surface normal + hinge axis | fixed/oracle 3 cm/12 deg `5.0%/10.5%` | Simple local geometry does not uniquely identify hand orientation |
| A030/G060 | Predicted/GT affordance -> top-4 NMS contact proposals | PRED 3 cm/5 cm `47.64%/51.57%`; GT `57.85%/70.68%` | "Where" is partially learnable and retains meaningful headroom |
| G061-G062 | Frozen predicted contacts -> deterministic contact-conditioned SE(3) | `0.221 m`, `1.673 rad`, 3 cm/12 deg `1.24%`; paired CIs cross zero | Contact helps weakly, but a single shared rotation mode remains inadequate |
| G110 | Learned grasp routes + frozen O185 operation | both learned routes strict/task `0/8, 0/8` | Nearest fully learned proxy is 0%; it still uses recorded observation, not full live perception |

GT route ceilings also matter. With valid teachers, GT direct trajectories achieve route-level strict grasp `8/8`, whereas GT terminal-qpose plus a fixed shortest-path selector reaches only `1/8` strict despite planner success. This shows that trajectory shape and candidate selection matter; endpoint reachability alone is insufficient.

## First-principles failure diagnosis

### 1. The output is not identifiable from the input as a single target

A partial single-view point cloud can localize a handle or movable part, but it often cannot uniquely determine gripper roll, approach side, collision-free pregrasp, or IK elbow branch. Symmetry and occlusion produce several physically equivalent explanations. G059's oracle geometry frame reaches only `10.5%` at 3 cm/12 deg, directly showing that normal + hinge geometry is not enough.

Consequence: a deterministic rotation regressor is asked to predict a value that is not a function of the observation. It will average modes or lock onto dataset frequency rather than generate a valid mode.

### 2. Joint-space labels contain artificial disagreement

G051 reconstructs the same hand SE(3) with IK, yet the IK qpose differs from the stored teacher by median/p90/max `0.620/4.846/7.364 rad`. Therefore joint-space L1 penalizes equivalent kinematic solutions as large errors. K=4 Hungarian matching reduces permutation sensitivity but does not enumerate the valid IK equivalence class.

Consequence: directly adding a Stage-1 qpose regression head would inject contradictory gradients unless supervision is set-valued, branch-aware, or evaluated through FK/task constraints.

### 3. The teacher set is sparse, trajectory-specific, and incomplete

Training has 531 observation groups and 2,373 retained trajectory-specific teachers, capped at four teachers per group. These are successful collected paths, not an exhaustive set of valid contacts, orientations, IK branches, or approach paths. A prediction can be physically valid yet be far from every recorded teacher. Presence means a stored proposal exists, not that a proposed grasp will succeed.

Consequence: conventional supervised regression confuses "not recorded" with "invalid." The issue is especially severe for G010: each sample emits up to `4 x 64 x 7 = 1,792` continuous trajectory values from only 531 training groups.

### 4. Current factorization spends capacity on the wrong uncertainty

The global heads jointly infer contact, hand center, orientation, IK branch, and path shape from one pooled scene token. A030 proves contact can be separated and learned. G062 anchors translation to contact, but still uses one shared orientation head and no explicit IK-mode variable.

Consequence: the model solves an unnecessarily entangled inverse problem. The correct factorization is closer to contact -> local orientation mode -> IK branch -> approach path, with separate scoring and constraints.

### 5. Offline losses do not match physical acceptance

Current losses optimize joint L1, SE(3) translation/geodesic error, or trajectory pointwise error. Physical success depends on pregrasp collision, approach direction, closure geometry, path continuity, contact stability, and post-grasp manipulability. G020's `8/8` planner coverage but `0/8` learned strict grasp is the clearest mismatch.

Consequence: low supervised loss can coexist with useless candidates. Planner/collision/FK/manipulability must enter candidate generation or outcome-blind ranking before physical rollout.

### 6. Grasp and operation are coupled through a state-distribution shift

O185 reaches operation-only `5/8` from its expected starting distribution, but learned grasp + O185 is `0/8`. Even a grasp that closes successfully may leave a different arm pose, contact patch, object state, or manipulability profile than the operation model saw in training.

Consequence: strict grasp is the immediate blocker, but grasp-to-operation compatibility is a second independent gate. Optimizing a generic stable grasp alone is insufficient.

### 7. Static open-loop prediction cannot repair small execution errors

G110 uses recorded current observations, and the grasp approaches are executed without a learned receding-horizon visual/contact correction loop. Small pose error near a narrow handle causes complete contact loss.

Consequence: even a better proposal generator needs either robust approach margins or closed-loop correction. This should follow, not replace, fixing the proposal distribution.

## Literature landscape

No relevant local PDF library was found. The canonical `verify_papers.py` helper and Gemini/reviewer backends were unavailable; exact paper IDs and titles below were instead checked directly against the arXiv API on 2026-08-07.

| Paper | Status | Relevant mechanism | Implication here |
|---|---|---|---|
| Mo et al., *Where2Act*, ICCV 2021, arXiv:2101.02692 | verified via arXiv | predicts localized actionability and action proposals at image/depth pixels | supports moving action prediction to spatial points, but its elementary push/pull action is simpler than robot grasp + IK + trajectory |
| Sundermeyer et al., *Contact-GraspNet*, ICRA 2021, arXiv:2103.14127 | verified via arXiv | roots a distribution of 6-DoF grasps at point-cloud contacts, reducing effective representation dimension | strongest direct precedent for contact-anchored grasp representation |
| Mo et al., *GraspNet*, arXiv:1912.13470 | verified via arXiv | dense grasp labels and analytic grasp evaluation rather than one pose label | supports dense/set-valued supervision and physics-based validity evaluation |
| Fang et al., *AnyGrasp*, T-RO, arXiv:2212.08333 | verified via arXiv | dense 7-DoF grasp proposals and temporal correspondence | supports generating and tracking multiple candidates rather than regressing one global set |
| Urain et al., *SE(3)-DiffusionFields*, arXiv:2209.03855 | verified via arXiv | multimodal SE(3) diffusion cost integrated with collision/joint-limit motion optimization | supports energy/diffusion modeling plus planner guidance |
| *Grasp Diffusion Network*, arXiv:2412.08398 | verified via arXiv | samples multimodal SO(3) x R3 grasp poses from partial point clouds with collision guidance | direct evidence that continuous generative pose models match the multimodal target structure |
| Chi et al., *Diffusion Policy*, RSS 2023, arXiv:2303.04137 | verified via arXiv | multimodal action-sequence generation with receding-horizon execution | supports conditional trajectory generation and closed-loop action chunks, not monolithic supervised regression |
| Zhao et al., *ACT*, RSS 2023, arXiv:2304.13705 | verified via arXiv | generates action chunks to reduce compounding error | supports sparse/chunked trajectory fitting as a separate test from G010 |
| *AnchorDP3*, arXiv:2506.19269 | verified via arXiv | affordance-anchored sparse keypose diffusion with simultaneous joint and end-effector supervision | very close to the user's Stage-1/keypose proposal; makes the generic architecture direction low-novelty, but provides a strong implementation baseline |
| *ClickDiff*, ACM MM 2024, arXiv:2407.19370 | verified via arXiv | predicts contact then generates contact-conditioned grasp distributions | reinforces the contact-first generative factorization; human-hand setting differs from robot articulated manipulation |
| *Sim2Real2*, ICRA 2023, arXiv:2302.10693 | verified via arXiv | active interaction resolves object properties unobservable from one static point cloud | supports active perception only if static ambiguity remains after representation repair |

## Candidate ideas from sequential discovery lenses

All non-duplicate ideas fit the available data/compute envelope. None was eliminated for subjective quality before an external jury because that jury is unavailable.

| Rank | Idea | Core hypothesis | Cheapest decisive test | Risk |
|---|---|---|---|---|
| 1 | Contact-Anchored Mode-and-IK Generator | given a predicted contact, discrete local orientation modes plus explicit IK-branch enumeration are learnable even when a single rotation/qpose is not | TRAIN-only orientation codebook; predict top-M modes at G061 contacts; enumerate IK and report best-of-M pose/planner coverage on frozen CAL | medium |
| 2 | Set-Valued IK-Equivalent Supervision | current labels are too sparse; augmenting every teacher with FK-equivalent IK solutions removes false joint-space penalties | generate legal IK alternatives for existing TRAIN teachers; compare original-vs-augmented qpose CAL min-set error with identical model | low-medium |
| 3 | Dual-Space Sparse Phase-Keypose Head | jointly supervising contact, pregrasp/grasp SE(3), and qpose through FK consistency is easier than dense trajectory fitting | predict 2-3 phase keyposes with contact anchor; compare against G010/G020 at matched parameter/sample budget | medium |
| 4 | Planner-Guided SE(3) Energy/Diffusion | a generative SE(3) distribution plus collision/reachability guidance preserves modes that deterministic rotation-6D averages away | train a small conditional score/MDN on contact-local poses; sample 32 candidates; evaluate best-of-N and outcome-blind planner ranking | medium-high |
| 5 | Transition-Aware Grasp Ranking | grasps should be selected for operation manipulability, not only closure | label TRAIN candidates with outcome-blind post-grasp Jacobian margin, joint-limit margin, and operation-direction compatibility; test whether ranking improves GT candidate route ceiling | medium |
| 6 | Contact-and-Mode-Conditioned Trajectory Generator | G010 failed because it globally regressed whole paths; conditioning on contact and IK mode makes trajectory distribution much simpler | reuse selected contact/mode endpoints, train residual phase trajectory or short action chunks, compare planner/physical screen | medium-high |
| 7 | Retrieval plus Residual Grasp Realization | sparse 531-group data may favor retrieving a TRAIN local-geometry prototype and predicting a small residual | contact-local feature nearest-neighbor baseline plus learned residual; compare to G062 and base SE(3) | low |
| 8 | SE(3)-Equivariant Contact-Local Encoder | current PointNet/global frames waste data learning rotations and translations | frozen contact-local canonical frame plus equivariant/local encoder, holding output and loss fixed | medium |
| 9 | Closed-Loop Grasp Servo | receding-horizon updates can tolerate residual proposal error near contact | oracle/GT-pose perturbation sweep first; only if moderate perturbations are recoverable, train short visual action chunks | medium-high |
| 10 | Active View/Probe Disambiguation | some orientation/kinematic modes are fundamentally unobservable from one view | measure pose-mode entropy reduction using existing alternate views or one safe probe before any new policy | high |

## Novelty pre-check for the top directions

### Idea 1: Contact-Anchored Mode-and-IK Generator

- **Novelty of generic contact-anchored grasp generation**: low; Contact-GraspNet and modern grasp diffusion methods already establish this.
- **Novelty of articulated-task contact + orientation-mode + IK-branch factorization**: medium. The differentiator must be explicit set-valued robot kinematics and downstream articulated-operation compatibility, not merely applying a grasp network.
- **Recommendation**: proceed as the highest-value engineering/scientific direction, but frame the contribution around diagnosing and resolving false supervision from SE(3)/IK equivalence.

### Idea 2: Set-Valued IK-Equivalent Supervision

- **Novelty of multiple IK solutions**: low as robotics knowledge.
- **Novelty of treating IK equivalence as missing-label completion for learned articulated grasp proposals**: medium, subject to deeper search.
- **Recommendation**: run first as a cheap diagnostic/ablation. It can validate the central causal claim before building a generative model.

### Idea 3: Dual-Space Sparse Phase-Keypose Head

- **Novelty**: low in generic form because AnchorDP3 already predicts affordance-anchored keyposes with joint and end-effector supervision.
- **Potential differentiator**: set-valued IK consistency and grasp-to-operation transition constraints on articulated objects.
- **Recommendation**: use AnchorDP3 as a baseline/design reference, not as the paper's nominal novelty.

## Recommended method thesis

The best focused direction is not "put every output into Stage 1." It is:

> **Predict task-relevant contact in Stage 1, then generate a set of contact-local orientation modes; realize each mode through an IK-equivalent candidate set and rank candidates using outcome-blind planning and grasp-to-operation compatibility constraints.**

Implementation should have four explicit stages:

1. Reuse A030's per-point affordance and top-N contact proposal.
2. At each contact, predict a categorical orientation-mode distribution plus a small continuous residual, rather than one rotation-6D vector.
3. Enumerate multiple IK solutions for each task-space mode; train with min-over-valid/set loss and FK consistency instead of direct single-qpose L1.
4. Rank surviving candidates by collision, path feasibility, joint margin, and post-grasp operation manipulability; then execute a short approach path, optionally with closed-loop correction.

This retains the good part of the user's Where2Act analogy (spatially anchored action proposals), avoids the demonstrated failure of global monolithic regression, and makes each failure source measurable.

## Minimal experiment sequence before implementation-scale training

1. **M0 label audit: orientation and IK mode entropy.** Cluster TRAIN teacher orientations in a contact-local frame, enumerate IK alternatives, and report modes/contact, IK solutions/mode, and CAL best-of-M oracle curves. Go only if M=4/8 materially lifts the pose/IK ceiling over G062.
2. **M1 fixed-contact mode classifier.** Use G061 frozen contacts and TRAIN-only codebook labels. Compare single regression, M-mode classification+residual, and min-over-set loss with identical encoder/budget. Gate on paired CAL pose error and planner coverage, not training loss.
3. **M2 predicted-contact full proposal.** Combine A030 contacts with the M1 generator, sample/rank candidates using only legal geometry/kinematics, and require a substantial offline best-of-N and selected top-1 gain before physical rollout.
4. **M3 physical grasp-only screen.** Fresh-world fixed targets, strict grasp first. Do not attach O185 until grasp is non-zero and stable across seeds.
5. **M4 transition-aware ranking and handoff.** Once strict grasp passes, compare generic grasp ranking against post-grasp manipulability-aware ranking with the same O185 operation consumer.

Direct trajectory learning should be revisited only after M1/M2 identify a valid contact/orientation/IK mode. The next trajectory model should predict sparse phase keyposes or short residual chunks conditioned on that mode; repeating global `K x 64 x 7` regression is not justified.

## Decision and limitations

- **Strongest current causal conclusion**: contact localization is promising; deterministic global or contact-conditioned pose regression is the wrong output model for the available sparse multimodal labels.
- **Highest-priority next action**: M0 set-valued orientation/IK diagnostic, followed by M1 contact-local mode classification if the oracle curve is positive.
- **Do not do next**: another PointNet/mask encoder tweak, another single rotation-6D seed sweep, or another monolithic full-trajectory regressor.
- **Pipeline limitation**: external triage, external novelty jury, research-review, and reviewer-driven refinement could not be performed because their required MCP backends are absent. Consequently no reviewer score, "novelty confirmed" claim, pilot result, `FINAL_PROPOSAL.md`, or replacement Architecture 6 experiment plan is emitted.

## Primary repository evidence

- `EXPERIMENT_PLAN.md`: frozen contracts, terminal outcomes, and current decision gates.
- `a6_grasp_models.py`: global K-query regressors and deterministic contact-conditioned SE(3) implementation.
- `results/a6_g010c_direct_traj_fixed_batch_v1/summary.json`: full joint-trajectory fit.
- `results/a6_g020c_qpose_fixed_batch_v1/summary.json`: terminal-qpose fit.
- `results/a6_g055c_se3_offline_v1/summary.json`: SE(3) offline accuracy and IK coverage.
- `results/a6_g062c_contact_conditioned_se3_fit_v1/full/summary.json`: contact-conditioned scoped negative.
- `results/a6_g110c_learned_selected_physical_v1/summary.json`: nearest fully learned physical proxy.
- `../stage1_initial_updated_co_output/train_co_output.py`: older Stage-1 co-output scope is affordance-only.

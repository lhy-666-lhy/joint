# Architecture 6 Grasp-Factor Results

**Date**: 2026-08-07
**Plan**: `EXPERIMENT_PLAN.md`
**Scope**: A031-A032 affordance semantics, G063-G066 contact-mode/IK selection, G070 direct-qpose alternative

## Completed Runs

| Run | Result | Key evidence | Decision |
|---|---|---|---|
| `A6-A032C` | NO_REPLACEMENT | INITIAL distance improves `-0.0160m` but 3cm/5cm coverage drops `-4.45/-0.79pp`; MIX060 regresses all contact metrics | retain A030 updated |
| `A6-G065C` | ORACLE_ONLY | best-of-8 `0.1722m/0.5795rad/3.47%`; learned selected top-1 regresses | test bounded selectors only |
| `A6-G066C-S1` | SCOPED_NEGATIVE | versus G062 translation `+0.02584m [0.00770,0.04401]` | calibrated risk is insufficient |
| `A6-G066C-S2` | SCOPED_NEGATIVE | `0.2938m/2.1630rad/0.74%`; versus G062 `+0.07261m/+0.48979rad`, both CIs strictly positive | stop G067 and physical |
| `A6-G066C` failure analysis | DIAGNOSTIC | full oracle legal-IK/planner coverage `34.82%/7.59%`; planner-feasible oracle loses `+0.1045m/+1.2687rad` | bottleneck is candidate realizability |
| `A6-G070C` | SCOPED_NEGATIVE | TRAIN loss `0.8653 -> 0.1926`, CAL qpose best-of-8 L1 `0.5953rad`, zero `<=0.03` proxy hits | direct qpose set does not generalize |
| `A6-G070C-PLANNER` | SCOPED_NEGATIVE | any/selected planner coverage `25.65%`; planner-selected qpose L1 `0.6014rad` | stop direct-qpose branch |

## Result-to-Claim

- Supported: contact-local multimodality and IK equivalence exist in the labels; increasing candidate count improves an oracle upper bound.
- Supported: current learned scores, IK/FK/path heuristics, and direct qpose regression do not convert that oracle into a deployable top-1 candidate.
- Not supported: INITIAL or MIX060 replacing A030 updated; G066/G070 improving G062; any grasp physical or full-task improvement.
- Anti-claim ruled out: the failure is not only a weak mode classifier. Even the full-oracle candidate usually lacks a legal IK/planner realization.

## Failure Interpretation

The common failure is observation-to-executable-candidate generalization. G065 predicts pose candidates with a useful offline oracle but weak IK/planner realizability. G070 removes the SE(3)-to-IK conversion, yet direct qpose-set regression overfits TRAIN and retains only `25.65%` planner coverage on CAL. Post-hoc selector tuning cannot repair a candidate set that rarely contains a correct executable member.

## Stop Decision

- Keep `A6-A030C` updated affordance as the frozen producer.
- Keep G061 contact-query and G063/G064 label/IK contracts as valid mechanism evidence.
- Reject G065 S0/S1/S2 promotion and G070 direct-qpose promotion.
- Do not run G067, A033, G068, physical rollout, extra affordance mix weights, larger backbone, or Diffusion under this revision.
- A new revision is required before further grasp training. It must change the supervision/data formulation, such as executable trajectory/keypose coverage or observation disambiguation, rather than add another post-hoc selector.

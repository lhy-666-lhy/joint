# Architecture 6 Global Termination Review

> **Status: SUPERSEDED.** Revision `20260806T060210Z-c3831dc1` reopened Architecture 6 after confirming an arbitrary fit gate, an invalid state-delta normalizer, and dirty-sample membership drift. This file is historical evidence only; use `EXPERIMENT_PLAN.md` and `EXPERIMENT_TRACKER.md` for current decisions.

- **Outcome**: `unsuccessful`
- **Event ID**: `a6_global_termination_20260805T193129Z`
- **Time**: `2026-08-05T19:31:29Z`
- **Scope**: Architecture 6 frozen contracts only

## Global Objective

The deployable operation/grasp system, frozen-vs-joint comparison, SAME_TEST, and TARGET_TEST objectives were not reached. No MECH_DEV or final split outcome was used to create or select a candidate.

## Terminal Evidence

| Branch | Terminal evidence | Decision |
|---|---|---|
| ABS fixed64 | O010S 0.014016; O020S 0.028086; O030S AR 0.056632; repeat-last 0.010378 | All three valid scoped training-fit failures; budget-only rescue rejected |
| Command-delta fixed64 | O200F MAE 0.000651, decrease 87.55x, parity 4.28e-7, reload 0 | MAE passed but frozen 100x gate failed; branch rejected |
| State-start-delta fixed64 | O201F MAE 1.513186, decrease 22.93x, parity 3.50e-7, reload 0 | Both fit gates failed; branch rejected |
| Affordance membership | TRAIN 557/559 before recovery; IDs 70/251 | Split cannot be shrunk; A010 remains blocked |
| Strict/relaxed recovery | A000RRR and A000RRRR each produced 0/2 primary rows | Valid local data-contract failure; recovery route closed |

## Exhaustion Review

All READY routes authorized by A6-REPAIR-v1.4, A6-CMDDELTA-v1.5, and A6-STATEDELTA-v1.6 are terminal. There is no contract-valid operation candidate for DYN64 and no complete split-safe affordance manifest for A010. Consequently every downstream physics, grasp, integration, and final-test run remains dependency-blocked.

Continuing would require a new scientific contract that changes a frozen gate, representation/model/optimizer, or membership/producer policy. Such a change cannot be selected post hoc inside this loop. The current major credible routes are therefore exhausted under the frozen Architecture 6 scope.

## Claim Limits

This outcome does not show that all architectures, representations, optimizers, camera policies, or datasets are impossible. It shows only that the explicitly frozen Architecture 6 routes did not satisfy their preregistered gates. O200F's low absolute MAE may motivate a separately preregistered future study, but it is not a pass here.

## Termination Decision

Set `loop_status=completed` with outcome `unsuccessful`. The external watcher must independently review this unsuccessful termination before treating it as final.

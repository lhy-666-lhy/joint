# Architecture 6 Logical Audit

## Verdict

The GT replay executor is valid. The previous O130 runner used the wrong entry point for a learned chunk policy. In addition, the first offline comparison contained one representation mismatch and an invalid split/scope claim.

## Findings

1. O100C reused an O200F checkpoint trained with the D020 absolute-action standard deviation, then evaluated it with the D021C command-delta standard deviation. O100C is invalid implementation evidence.
2. O110C/O120C are finite fixed-batch fits, but their DYN64 evaluation is A5_TRAIN recorded-observation data and overlaps fixed64 anchors. They are diagnostic only, not CAL or deployment evidence.
3. D040C reads recorded `contact_feedback`. This is acceptable for teacher-forced diagnosis only. Deployable rollout must construct the same input schema from current SAPIEN observations.
4. `replay_action` remains the correct GT/full-action parity executor. A learned chunk policy must use the Architecture 2/3 live loop: observe, predict, execute prefix, observe again, and stop by metadata-derived opening progress or max policy calls.
5. O100RC repaired the MLP fixed-fit normalizer mismatch. Its artifact is still not a held-out or deployment result.

The next authorized experiment is A6-D041C: clean A5_CAL recorded-current-observation input plus an explicit live-observation adapter contract.

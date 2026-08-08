from copy import deepcopy

from jointTrain_new.experiment.model_architecture_6.a6_g005c_route_selector import (
    select_route_rows,
    selected_indices,
)


def _row(candidate_index, length, strict=False, task=False, progress=0.0, contact=0.0):
    return {
        "group_index": 7,
        "group_id": "group-7",
        "candidate_index": candidate_index,
        "joint_space_length": length,
        "grasp": {"strict_grasp_pass": strict},
        "operation": {
            "task_success": task,
            "final_progress": progress,
            "contact_fraction": contact,
        },
    }


def test_selector_uses_length_then_candidate_index():
    rows = [_row(2, 1.0), _row(1, 1.0), _row(0, 2.0, strict=True, task=True)]
    assert selected_indices(select_route_rows(rows, [7])) == {"7": 1}


def test_selector_is_invariant_to_physical_outcomes():
    rows = [_row(0, 2.0, strict=True, task=True, progress=1.0), _row(1, 1.0)]
    mutated = deepcopy(rows)
    for row in mutated:
        row["grasp"]["strict_grasp_pass"] = not row["grasp"]["strict_grasp_pass"]
        row["operation"].update(task_success=not row["operation"]["task_success"], final_progress=-99.0, contact_fraction=-99.0)
    assert selected_indices(select_route_rows(rows, [7])) == selected_indices(
        select_route_rows(mutated, [7])
    )

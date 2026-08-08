from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from jointTrain_new.experiment.model_architecture_6.a6_joint_goal_planner import (
    extract_trajopt_paths,
    validate_joint_goal_inputs,
)
from jointTrain_new.experiment.model_architecture_6.run_a6_g005c_joint_goal_planner_sanity import (
    file_read_string_literals,
)


class JointGoalPlannerInputTest(unittest.TestCase):
    def test_extract_trajopt_paths_uses_interpolated_solution(self) -> None:
        expected = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
        result = SimpleNamespace(interpolated_solution=SimpleNamespace(position=expected))
        actual = extract_trajopt_paths(result, batch_size=1)
        self.assertEqual(actual.shape, (1, 3, 7))
        np.testing.assert_array_equal(actual[0], expected)

    def test_extract_trajopt_paths_preserves_batch_axis(self) -> None:
        expected = np.arange(2 * 3 * 7, dtype=np.float32).reshape(2, 3, 7)
        result = SimpleNamespace(interpolated_solution=SimpleNamespace(position=expected))
        np.testing.assert_array_equal(
            extract_trajopt_paths(result, batch_size=2), expected
        )

    def test_file_read_audit_ignores_diagnostic_key_names(self) -> None:
        source = '''
        def consumer(root):
            labels = np.load(root / "qpose_labels.npz")
            return {"stored_path_or_pregrasp_read": False, "labels": labels}
        '''
        self.assertEqual(file_read_string_literals(source), {"qpose_labels.npz"})

    def test_file_read_audit_detects_forbidden_sidecar(self) -> None:
        source = '''
        def consumer(root):
            return np.load(root / "traj_labels.npz")
        '''
        self.assertIn("traj_labels.npz", file_read_string_literals(source))

    def test_joint_goal_inputs_broadcast_one_start(self) -> None:
        starts, goals = validate_joint_goal_inputs(
            np.arange(7, dtype=np.float32), np.zeros((3, 7), dtype=np.float32)
        )
        self.assertEqual(starts.shape, (3, 7))
        self.assertEqual(goals.shape, (3, 7))
        np.testing.assert_array_equal(starts[0], starts[2])

    def test_joint_goal_inputs_reject_bad_shapes_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "goal qpos"):
            validate_joint_goal_inputs(np.zeros(7), np.zeros((1, 6)))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_joint_goal_inputs(np.zeros((2, 7)), np.zeros((3, 7)))
        bad = np.zeros((1, 7), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_joint_goal_inputs(np.zeros(7), bad)


if __name__ == "__main__":
    unittest.main()

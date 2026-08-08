from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jointTrain_new.experiment.model_architecture_6.a6_operation_start_contract import (
    load_operation_start,
)


class OperationStartContractTest(unittest.TestCase):
    def test_logged_command_wins_over_mislabeled_robot_qpos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "trajectory.npz"
            robot = np.arange(9, dtype=np.float32) / 10.0
            command = robot.copy()
            command[7:9] = 0.0
            command_rows = np.stack([robot, command, command + 0.01])
            np.savez_compressed(
                path,
                action_phase=np.asarray(["gripper_close", "operation", "operation"]),
                operation_start_index=np.asarray(1, dtype=np.int32),
                actual_joint_qpos=command_rows,
                joint_command_qpos=command_rows,
                operation_start_robot_qpos=robot,
                operation_start_object_qpos=np.asarray([0.2], dtype=np.float32),
                operation_start_joint_command_qpos=robot,
                logged_operation_start_joint_command_qpos=command,
            )

            record = load_operation_start(path)
            selected_command = record["command_qpos"]

            np.testing.assert_array_equal(selected_command, command)
            self.assertEqual(
                record["command_source"], "logged_operation_start_joint_command_qpos"
            )
            self.assertEqual(selected_command[7], 0.0)
            self.assertEqual(selected_command[8], 0.0)


if __name__ == "__main__":
    unittest.main()

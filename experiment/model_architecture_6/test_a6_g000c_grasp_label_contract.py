from __future__ import annotations

import unittest

import numpy as np

from jointTrain_new.experiment.model_architecture_6.run_a6_g000c_grasp_label_contract import (
    resample_joint_path,
    select_teacher_values,
)


class GraspLabelContractTest(unittest.TestCase):
    def test_resample_preserves_endpoints_and_joint_space_line(self) -> None:
        qpath = np.zeros((3, 7), dtype=np.float32)
        qpath[1, 0] = 1.0
        qpath[2, :2] = 1.0
        result = resample_joint_path(qpath, length=5)
        np.testing.assert_array_equal(result[0], qpath[0])
        np.testing.assert_array_equal(result[-1], qpath[-1])
        np.testing.assert_allclose(
            result[:, :2],
            np.asarray([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1]]),
        )

    def test_teacher_selection_keeps_index_order_without_padding(self) -> None:
        self.assertEqual(select_teacher_values(["c", "a", "b"], k=4), ["c", "a", "b"])
        self.assertEqual(select_teacher_values([5, 4, 3, 2, 1], k=4), [5, 4, 3, 2])

    def test_teacher_selection_rejects_duplicate_lineage(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_teacher_values(["same", "same"], k=4)

    def test_resample_rejects_zero_length_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-length"):
            resample_joint_path(np.zeros((2, 7), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

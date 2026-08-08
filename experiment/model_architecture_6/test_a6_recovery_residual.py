from __future__ import annotations

import unittest

import torch

from jointTrain_new.experiment.model_architecture_6.a6_operation_models import (
    OperationMLPAbsolute,
    OperationMLPRecoveryResidual,
)
from jointTrain_new.experiment.model_architecture_6.run_a6_o185c_recovery_residual_train import (
    build_sampling_indices,
)


class RecoveryResidualTest(unittest.TestCase):
    def test_epoch_balanced_sampler_equalizes_row_exposure(self) -> None:
        indices = build_sampling_indices(
            start=1024,
            stop=1152,
            steps=6000,
            batch_size=32,
            generator=torch.Generator().manual_seed(11),
            mode="epoch_balanced_without_replacement",
        )
        exposure = torch.bincount(indices.reshape(-1) - 1024, minlength=128)

        self.assertEqual(indices.shape, (6000, 32))
        self.assertEqual(int(indices.min()), 1024)
        self.assertEqual(int(indices.max()), 1151)
        self.assertEqual(int(exposure.min()), 1500)
        self.assertEqual(int(exposure.max()), 1500)

    def test_zero_initialized_residual_matches_baseline(self) -> None:
        torch.manual_seed(7)
        baseline = OperationMLPAbsolute(dropout=0.0).eval()
        residual = OperationMLPRecoveryResidual(dropout=0.0).eval()
        residual.baseline.load_state_dict(baseline.state_dict(), strict=True)
        inputs = (
            torch.randn(2, 16, 3),
            torch.zeros(2, 16, dtype=torch.bool),
            torch.zeros(2, 16),
            torch.randn(2, 81),
            torch.randn(2, 43),
        )

        with torch.no_grad():
            expected = baseline(*inputs)
            actual = residual(*inputs)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()

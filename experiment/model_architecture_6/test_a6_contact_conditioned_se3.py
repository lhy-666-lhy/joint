from __future__ import annotations

import unittest

import torch

from a6_grasp_models import ContactConditionedSE3, contact_se3_loss


class ContactConditionedSE3Test(unittest.TestCase):
    def test_zero_initialized_offset_and_backward(self) -> None:
        torch.manual_seed(7)
        model = ContactConditionedSE3(hidden_dim=32)
        model.eval()
        points = torch.randn(2, 64, 3)
        state = torch.randn(2, 7)
        affordance = torch.rand(2, 64)
        query = points[:, :4].clone()
        output = model(points, state, affordance, query)
        self.assertEqual(tuple(output["values"].shape), (2, 4, 9))
        self.assertEqual(tuple(output["presence_logits"].shape), (2, 4))
        self.assertTrue(torch.equal(output["values"][..., :3], query))
        target = output["values"].detach().clone()
        target[..., 3:9] = torch.randn(2, 4, 6)
        presence = torch.tensor([[True, True, False, False], [True, True, True, False]])
        loss, _ = contact_se3_loss(output, target, presence)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.rotation_head.weight.grad)


if __name__ == "__main__":
    unittest.main()

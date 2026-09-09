import unittest

import torch

from metrics.pointcloud_metrics import lgan_mmd_cov, one_nn_accuracy


class PointCloudMetricTest(unittest.TestCase):
    def test_lgan_mmd_cov_uses_unique_nearest_references(self):
        distances = torch.tensor(
            [
                [0.1, 2.0, 3.0],
                [0.2, 1.0, 3.0],
                [2.0, 0.1, 3.0],
            ]
        )

        result = lgan_mmd_cov(distances)

        self.assertEqual(result["cov"], 2.0 / 3.0)
        self.assertEqual(result["mmd"], torch.tensor([0.1, 0.1, 3.0]).mean().item())
        self.assertEqual(result["mmd_smp"], torch.tensor([0.1, 0.2, 0.1]).mean().item())

    def test_one_nn_is_one_for_separable_sets(self):
        within = torch.tensor([[0.0, 0.1], [0.1, 0.0]])
        cross = torch.tensor([[10.0, 10.1], [9.9, 10.0]])

        result = one_nn_accuracy(within, cross, within)

        self.assertEqual(result, {"acc": 1.0, "acc_ref": 1.0, "acc_sample": 1.0})

    def test_one_nn_is_zero_for_interleaved_sets(self):
        ref_ref = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        sample_sample = torch.tensor([[0.0, 9.8], [9.8, 0.0]])
        sample_ref = torch.tensor([[0.1, 9.9], [9.9, 0.1]])

        result = one_nn_accuracy(ref_ref, sample_ref, sample_sample)

        self.assertEqual(result, {"acc": 0.0, "acc_ref": 0.0, "acc_sample": 0.0})


if __name__ == "__main__":
    unittest.main()

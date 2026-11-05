"""Tests for kernel-enabled dual SVM prediction path."""
from __future__ import annotations

import unittest

import numpy as np

from kernel_team.kernels import linear_kernel, rbf_kernel
from kernel_team.kernel_dual_svm import (
    decision_scores,
    predict_labels,
    solve_dual_kernel,
)


class TestSolveDualKernel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        # Mildly separable synthetic blobs
        X0 = rng.normal(loc=-1.0, scale=0.6, size=(40, 3))
        X1 = rng.normal(loc=1.0, scale=0.6, size=(40, 3))
        self.X = np.vstack([X0, X1])
        self.y01 = np.array([0] * 40 + [1] * 40, dtype=np.int8)

    def test_linear_kernel_trains_and_predicts(self):
        K = linear_kernel(self.X, self.X)
        model = solve_dual_kernel(K, self.y01, C=1.0, tolerance=0.01)
        self.assertTrue(model["converged"])
        self.assertNotIn("w", model)
        scores = decision_scores(K, model)
        preds = predict_labels(K, model)
        self.assertEqual(preds.shape, (80,))
        self.assertTrue(np.isin(preds, [0, 1]).all())
        self.assertTrue(np.isfinite(scores).all())
        # Should classify most training points correctly on easy data
        acc = np.mean(preds == self.y01)
        self.assertGreater(acc, 0.85)

    def test_rbf_kernel_trains_and_predicts(self):
        K = rbf_kernel(self.X, self.X, gamma=0.5)
        model = solve_dual_kernel(K, self.y01, C=1.0, tolerance=0.01)
        self.assertTrue(np.isfinite(model["b"]))
        self.assertGreater(model["support_vectors"], 0)
        # Holdout-shaped query: fewer rows than training
        Xq = self.X[::5]
        Kq = rbf_kernel(Xq, self.X, gamma=0.5)
        preds = predict_labels(Kq, model)
        self.assertEqual(preds.shape, (len(Xq),))
        self.assertTrue(np.isin(preds, [0, 1]).all())

    def test_rejects_bad_kernel_shape(self):
        K = np.eye(10)
        with self.assertRaises(ValueError):
            solve_dual_kernel(K, self.y01, C=1.0)


if __name__ == "__main__":
    unittest.main()

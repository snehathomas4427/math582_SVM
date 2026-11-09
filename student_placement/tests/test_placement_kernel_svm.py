"""Tests for the placement (-1/+1 label) kernel dual SVM path.

Run from the repository root:
    python -m unittest <folder>.tests.test_placement_kernel_svm -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..kernels import linear_kernel, rbf_kernel
from ..placement_kernel_svm import (
    decision_scores,
    measure,
    predict_labels,
    read_processed_csv,
    solve_dual_kernel,
    stratified_subsample,
)


class TestSolveDualKernelPlacement(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        X0 = rng.normal(loc=-1.0, scale=0.6, size=(40, 3))
        X1 = rng.normal(loc=1.0, scale=0.6, size=(40, 3))
        self.X = np.vstack([X0, X1])
        self.y = np.array([-1] * 40 + [1] * 40, dtype=np.int8)

    def test_linear_kernel_trains_and_predicts(self):
        K = linear_kernel(self.X, self.X)
        model = solve_dual_kernel(K, self.y, C=1.0, tolerance=0.01)
        self.assertTrue(model["converged"])
        self.assertNotIn("w", model)
        preds = predict_labels(K, model)
        self.assertEqual(preds.shape, (80,))
        self.assertTrue(np.isin(preds, [-1, 1]).all())
        self.assertTrue(np.isfinite(decision_scores(K, model)).all())
        self.assertGreater(np.mean(preds == self.y), 0.85)

    def test_dual_feasibility(self):
        K = rbf_kernel(self.X, self.X, gamma=0.5)
        model = solve_dual_kernel(K, self.y, C=1.0, tolerance=0.01)
        a = model["alpha"]
        self.assertTrue(np.all(a >= -1e-12) and np.all(a <= 1.0 + 1e-12))
        self.assertLess(model["equality_residual"], 1e-9)

    def test_rbf_kernel_trains_and_predicts(self):
        K = rbf_kernel(self.X, self.X, gamma=0.5)
        model = solve_dual_kernel(K, self.y, C=1.0, tolerance=0.01)
        self.assertTrue(np.isfinite(model["b"]))
        self.assertGreater(model["support_vectors"], 0)
        Xq = self.X[::5]
        preds = predict_labels(rbf_kernel(Xq, self.X, gamma=0.5), model)
        self.assertEqual(preds.shape, (len(Xq),))

    def test_rejects_bad_kernel_shape(self):
        with self.assertRaises(ValueError):
            solve_dual_kernel(np.eye(10), self.y, C=1.0)

    def test_rejects_single_class(self):
        K = linear_kernel(self.X, self.X)
        with self.assertRaises(ValueError):
            solve_dual_kernel(K, np.ones(80), C=1.0)

    def test_matches_sklearn_if_available(self):
        try:
            from sklearn.svm import SVC
        except ImportError:
            self.skipTest("scikit-learn not installed")
        K = rbf_kernel(self.X, self.X, gamma=0.5)
        model = solve_dual_kernel(K, self.y, C=1.0, tolerance=1e-4)
        ours = predict_labels(K, model)
        ref = SVC(kernel="rbf", gamma=0.5, C=1.0).fit(self.X, self.y).predict(self.X)
        self.assertGreaterEqual(np.mean(ours == ref), 0.98)


class TestMetricsAndIO(unittest.TestCase):
    def test_measure(self):
        y_true = np.array([1, 1, -1, -1, 1])
        y_pred = np.array([1, -1, -1, 1, 1])
        m = measure(y_true, y_pred)
        self.assertEqual(m["confusion_matrix_TN_FP_FN_TP"], [1, 1, 1, 2])
        self.assertAlmostEqual(m["accuracy"], 0.6)
        self.assertAlmostEqual(m["precision_pos"], 2 / 3)
        self.assertAlmostEqual(m["recall_pos"], 2 / 3)

    def test_measure_all_negative_predictions(self):
        m = measure(np.array([1, -1, 1]), np.array([-1, -1, -1]))
        self.assertIsNone(m["precision_pos"])
        self.assertEqual(m["recall_pos"], 0.0)
        self.assertIsNone(m["f1_pos"])

    def test_stratified_subsample_keeps_ratio(self):
        y = np.array([-1] * 700 + [1] * 300)
        idx = stratified_subsample(y, 100, seed=1)
        self.assertEqual(len(idx), 100)
        self.assertAlmostEqual(np.mean(y[idx] == 1), 0.3, delta=0.02)
        np.testing.assert_array_equal(idx, stratified_subsample(y, 100, seed=1))

    def test_read_processed_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text("label,a,b\r\n1,0.5,1.0\r\n-1,-0.5,0.0\r\n")
            X, y, names = read_processed_csv(p)
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(X.shape, (2, 2))
        np.testing.assert_array_equal(y, [1, -1])


if __name__ == "__main__":
    unittest.main()

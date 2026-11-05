"""Unit tests for kernel_team.kernels.

Run from the repository root:
    python -m unittest kernel_team.tests.test_kernels -v
"""
from __future__ import annotations

import unittest

import numpy as np

from kernel_team.kernels import (
    get_kernel,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
)


class TestLinearKernel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(5, 3))
        self.Z = rng.normal(size=(4, 3))

    def test_shape(self):
        K = linear_kernel(self.X, self.Z)
        self.assertEqual(K.shape, (5, 4))

    def test_matches_definition(self):
        K = linear_kernel(self.X, self.Z)
        expected = self.X @ self.Z.T
        np.testing.assert_allclose(K, expected)

    def test_symmetric_when_same_data(self):
        K = linear_kernel(self.X, self.X)
        np.testing.assert_allclose(K, K.T)
        self.assertTrue(np.isfinite(K).all())

    def test_unequal_train_test_sizes(self):
        K = linear_kernel(self.X, self.Z[:2])
        self.assertEqual(K.shape, (5, 2))


class TestPolynomialKernel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.X = rng.normal(size=(6, 2))
        self.Z = rng.normal(size=(3, 2))

    def test_shape(self):
        K = polynomial_kernel(self.X, self.Z, degree=2, gamma=0.5, coef0=1.0)
        self.assertEqual(K.shape, (6, 3))

    def test_matches_definition(self):
        degree, gamma, coef0 = 3, 0.7, 0.5
        K = polynomial_kernel(self.X, self.Z, degree=degree, gamma=gamma, coef0=coef0)
        expected = (gamma * (self.X @ self.Z.T) + coef0) ** degree
        np.testing.assert_allclose(K, expected)

    def test_symmetric_when_same_data(self):
        K = polynomial_kernel(self.X, self.X, degree=2)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        self.assertTrue(np.isfinite(K).all())

    def test_degree_one_related_to_linear(self):
        # With gamma=1, coef0=0, degree=1 -> linear kernel
        K_poly = polynomial_kernel(self.X, self.Z, degree=1, gamma=1.0, coef0=0.0)
        K_lin = linear_kernel(self.X, self.Z)
        np.testing.assert_allclose(K_poly, K_lin)

    def test_default_gamma(self):
        K = polynomial_kernel(self.X, self.Z, degree=2)
        self.assertEqual(K.shape, (6, 3))
        self.assertTrue(np.isfinite(K).all())

    def test_invalid_degree(self):
        with self.assertRaises(ValueError):
            polynomial_kernel(self.X, self.Z, degree=0)


class TestRBFKernel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.X = rng.normal(size=(7, 4))
        self.Z = rng.normal(size=(5, 4))

    def test_shape(self):
        K = rbf_kernel(self.X, self.Z, gamma=0.1)
        self.assertEqual(K.shape, (7, 5))

    def test_values_in_unit_interval(self):
        K = rbf_kernel(self.X, self.Z, gamma=0.5)
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0 + 1e-12))
        self.assertTrue(np.isfinite(K).all())

    def test_diagonal_approximately_one(self):
        K = rbf_kernel(self.X, self.X, gamma=1.0)
        np.testing.assert_allclose(np.diag(K), np.ones(len(self.X)), atol=1e-12)

    def test_symmetric_when_same_data(self):
        K = rbf_kernel(self.X, self.X, gamma=0.2)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_matches_pairwise_definition(self):
        gamma = 0.3
        K = rbf_kernel(self.X, self.Z, gamma=gamma)
        # Explicit pairwise check for a few entries
        for i in range(min(3, len(self.X))):
            for j in range(min(3, len(self.Z))):
                d2 = np.sum((self.X[i] - self.Z[j]) ** 2)
                expected = np.exp(-gamma * d2)
                self.assertAlmostEqual(K[i, j], expected, places=10)

    def test_identical_points_give_one(self):
        x = self.X[:1]
        K = rbf_kernel(x, x, gamma=2.5)
        self.assertAlmostEqual(float(K[0, 0]), 1.0, places=12)

    def test_invalid_gamma(self):
        with self.assertRaises(ValueError):
            rbf_kernel(self.X, self.Z, gamma=-1.0)


class TestGetKernelAndEdgeCases(unittest.TestCase):
    def test_get_kernel_names(self):
        X = np.eye(3)
        for name in ("linear", "polynomial", "poly", "rbf", "gaussian"):
            kn = get_kernel(name, degree=2, gamma=0.5, coef0=1.0)
            K = kn(X, X)
            self.assertEqual(K.shape, (3, 3))
            self.assertTrue(np.isfinite(K).all())

    def test_unknown_kernel(self):
        with self.assertRaises(ValueError):
            get_kernel("sigmoid")

    def test_dimension_mismatch(self):
        X = np.ones((2, 3))
        Z = np.ones((2, 4))
        with self.assertRaises(ValueError):
            linear_kernel(X, Z)
        with self.assertRaises(ValueError):
            polynomial_kernel(X, Z)
        with self.assertRaises(ValueError):
            rbf_kernel(X, Z)

    def test_single_sample(self):
        X = np.array([[1.0, 2.0, 3.0]])
        Z = np.array([[0.5, -1.0, 2.0], [1.0, 2.0, 3.0]])
        self.assertEqual(linear_kernel(X, Z).shape, (1, 2))
        self.assertEqual(rbf_kernel(X, Z, gamma=1.0).shape, (1, 2))
        self.assertAlmostEqual(float(rbf_kernel(X, X, gamma=1.0)[0, 0]), 1.0)


class TestKernelDualConsistency(unittest.TestCase):
    """Sanity: linear kernel dual path agrees with Model-team linear math."""

    def test_linear_kernel_equals_dot_product_gram(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 5))
        K_lin = linear_kernel(X, X)
        K_dot = X @ X.T
        np.testing.assert_allclose(K_lin, K_dot)


if __name__ == "__main__":
    unittest.main()

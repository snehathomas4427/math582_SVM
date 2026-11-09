"""Kernel functions for soft-margin dual SVMs (Kernel Team).

Each kernel K(X, Z) returns a Gram matrix of shape (n_X, n_Z) with
entries K_{ij} = k(x_i, z_j). All operations are vectorized NumPy.

Mathematical definitions (Deisenroth et al., Mathematics for Machine
Learning, Ch. 12):

    Linear:      k(x, z) = x^T z
    Polynomial:  k(x, z) = (gamma * x^T z + c_0)^d
    RBF:         k(x, z) = exp(-gamma * ||x - z||^2)
"""
from __future__ import annotations

import numpy as np


def linear_kernel(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Linear kernel: K(x, z) = x^T z.

    Parameters
    ----------
    X : array, shape (n_x, d)
    Z : array, shape (n_z, d)

    Returns
    -------
    K : array, shape (n_x, n_z)
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if X.ndim != 2 or Z.ndim != 2:
        raise ValueError("X and Z must be 2-D feature matrices")
    if X.shape[1] != Z.shape[1]:
        raise ValueError(
            f"Feature dimensions must match: X has {X.shape[1]}, Z has {Z.shape[1]}"
        )
    return X @ Z.T


def polynomial_kernel(
    X: np.ndarray,
    Z: np.ndarray,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> np.ndarray:
    """Polynomial kernel: K(x, z) = (gamma * x^T z + coef0)^degree.

    Parameters
    ----------
    X : array, shape (n_x, d)
    Z : array, shape (n_z, d)
    degree : positive integer polynomial degree (default 3)
    gamma : scale of the inner product; default 1 / n_features
    coef0 : additive constant c_0 (default 1.0)

    Returns
    -------
    K : array, shape (n_x, n_z)
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if X.ndim != 2 or Z.ndim != 2:
        raise ValueError("X and Z must be 2-D feature matrices")
    if X.shape[1] != Z.shape[1]:
        raise ValueError(
            f"Feature dimensions must match: X has {X.shape[1]}, Z has {Z.shape[1]}"
        )
    if degree < 1:
        raise ValueError("degree must be a positive integer")
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    # (gamma * X Z^T + c_0)^d — vectorized over all pairs
    base = gamma * (X @ Z.T) + coef0
    return np.power(base, degree)


def rbf_kernel(
    X: np.ndarray,
    Z: np.ndarray,
    gamma: float | None = None,
) -> np.ndarray:
    """RBF (Gaussian) kernel: K(x, z) = exp(-gamma * ||x - z||^2).

    Uses the expansion
        ||x - z||^2 = ||x||^2 + ||z||^2 - 2 x^T z
    so the full pairwise distance matrix is formed without Python loops.

    Parameters
    ----------
    X : array, shape (n_x, d)
    Z : array, shape (n_z, d)
    gamma : bandwidth parameter; default 1 / n_features

    Returns
    -------
    K : array, shape (n_x, n_z), values in (0, 1]
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if X.ndim != 2 or Z.ndim != 2:
        raise ValueError("X and Z must be 2-D feature matrices")
    if X.shape[1] != Z.shape[1]:
        raise ValueError(
            f"Feature dimensions must match: X has {X.shape[1]}, Z has {Z.shape[1]}"
        )
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    # Squared Euclidean distances via ||x||^2 + ||z||^2 - 2 x^T z
    X_sq = np.sum(X * X, axis=1, keepdims=True)  # (n_x, 1)
    Z_sq = np.sum(Z * Z, axis=1, keepdims=True)  # (n_z, 1)
    # Broadcast: (n_x, 1) + (1, n_z) - 2 (n_x, n_z)
    sq_dists = X_sq + Z_sq.T - 2.0 * (X @ Z.T)
    # Clamp tiny negatives from floating-point roundoff
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.exp(-gamma * sq_dists)


def get_kernel(name: str, **params):
    """Return a callable kernel(X, Z) -> Gram matrix for a named kernel.

    Parameters
    ----------
    name : {'linear', 'polynomial', 'poly', 'rbf', 'gaussian'}
    **params : forwarded to the underlying kernel (gamma, degree, coef0, ...)

    Returns
    -------
    callable
        Function with signature kernel(X, Z) -> ndarray of shape (len(X), len(Z)).
    """
    key = name.lower().strip()
    if key == "linear":
        return lambda X, Z: linear_kernel(X, Z)
    if key in ("polynomial", "poly"):
        poly_keys = ("degree", "gamma", "coef0")
        poly_params = {k: params[k] for k in poly_keys if k in params}
        return lambda X, Z: polynomial_kernel(X, Z, **poly_params)
    if key in ("rbf", "gaussian"):
        rbf_params = {k: params[k] for k in ("gamma",) if k in params}
        return lambda X, Z: rbf_kernel(X, Z, **rbf_params)
    raise ValueError(
        f"Unknown kernel '{name}'. Choose from: linear, polynomial, rbf."
    )

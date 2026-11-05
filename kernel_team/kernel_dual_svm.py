"""Kernel soft-margin dual SVM built on the Model team's optimizer.

The Model team (epilepsy_linear_dual_svm.py) solves the textbook dual QP
with a two-coordinate working-set method on the linear Gram matrix
K = X @ X.T. This module keeps that optimizer intact in spirit and only
changes what is necessary to accept a *precomputed* kernel matrix:

    Dual:  max_alpha  sum_i alpha_i
                      - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
           s.t.       0 <= alpha_i <= C,   sum_i alpha_i y_i = 0

What changed relative to solve_dual_linear (and why)
----------------------------------------------------
1. Input is a precomputed Gram matrix K instead of feature matrix X.
   Reason: nonlinear kernels never form an explicit feature map, so the
   optimizer must operate on K_ij = k(x_i, x_j) directly.
2. No linear weight vector w is computed.
   Reason: w = Phi(X)^T (alpha * y) lives in the (possibly infinite-
   dimensional) feature space and is not available for RBF/poly kernels.
3. Intercept b is recovered via kernel evaluations on free support vectors:
       b = mean_i [ y_i - sum_j alpha_j y_j K(x_j, x_i) ]
   which reduces to the Model team's formula when K = X @ X.T.
4. Prediction uses f(x) = sum_i alpha_i y_i k(x_i, x) + b, never X @ w + b.

The coordinate-pair update, KKT gap stopping rule, box/equality
feasibility maintenance, and dual objective evaluation are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse Model-team I/O and preprocessing without modifying their file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from epilepsy_linear_dual_svm import (  # noqa: E402
    BasicPreprocessor,
    kkt_gap,
    measure,
    read_csv,
    TARGET,
)


def solve_dual_kernel(K: np.ndarray, y01: np.ndarray, C: float = 1.0,
                      tolerance: float = 0.003, max_steps: int = 150000):
    """Solve the soft-margin dual SVM given a precomputed training kernel.

    This is the Model team's two-coordinate working-set QP solver with the
    sole interface change that K is supplied externally:

        Q_ij = y_i * y_j * K_ij
        residual_i = y_i - sum_j alpha_j y_j K_ij

    Parameters
    ----------
    K : array, shape (n, n)
        Symmetric training Gram matrix K_ij = k(x_i, x_j).
    y01 : array, shape (n,)
        Labels encoded as 0/1 (seizure-negative / seizure-positive).
    C : float
        Soft-margin box constraint upper bound.
    tolerance : float
        KKT gap stopping tolerance (same default as Model team).
    max_steps : int
        Maximum coordinate-pair iterations.

    Returns
    -------
    dict
        Dual solution: alpha, b, diagnostics. No weight vector w.
    """
    if C <= 0 or tolerance <= 0:
        raise ValueError("C and tolerance must be positive")
    K = np.asarray(K, dtype=np.float64)
    y01 = np.asarray(y01)
    n = len(y01)
    if K.shape != (n, n):
        raise ValueError(f"K must be ({n}, {n}); got {K.shape}")
    if not np.isfinite(K).all():
        raise ValueError("Kernel matrix contains non-finite values")

    y = 2 * y01.astype(float) - 1
    if len(np.unique(y)) != 2:
        raise ValueError("Both seizure labels 0 and 1 must be present")

    # --- Model-team optimizer body (Gram matrix supplied, not X@X.T) ---
    diagonal = np.diag(K).copy()
    a = np.zeros(n, dtype=float)
    residual = y.copy()  # residual_i = y_i - sum_j alpha_j y_j K_ij
    converged = False
    for step in range(max_steps):
        eps = 1e-10 * max(C, 1)
        upper = ((y == 1) & (a < C - eps)) | ((y == -1) & (a > eps))
        lower = ((y == 1) & (a > eps)) | ((y == -1) & (a < C - eps))
        upper_scores = np.where(upper, residual, -np.inf)
        lower_scores = np.where(lower, residual, np.inf)
        i = int(np.argmax(upper_scores))
        gap = float(upper_scores[i] - np.min(lower_scores))
        if gap <= tolerance:
            converged = True
            break

        # Exact minimizer along the feasible pair (i, j).
        curvature = np.maximum(diagonal[i] + diagonal - 2 * K[i], 1e-12)
        change = np.maximum(residual[i] - residual, 0)
        decrease = np.where(lower, change * change / curvature, -np.inf)
        j = int(np.argmax(decrease))
        s = y[i] * y[j]

        # alpha_i' = alpha_i + d, alpha_j' = alpha_j - s*d preserves y·alpha.
        d = float(y[i] * (residual[i] - residual[j]) / curvature[j])
        lo, hi = -a[i], C - a[i]
        if s == 1:
            lo, hi = max(lo, a[j] - C), min(hi, a[j])
        else:
            lo, hi = max(lo, -a[j]), min(hi, C - a[j])
        d = float(np.clip(d, lo, hi))
        if abs(d) < 1e-14:
            break
        a[i] += d
        a[j] -= s * d
        residual -= (d * y[i]) * (K[:, i] - K[:, j])

    iterations = step if converged else step + 1
    gap = kkt_gap(a, y, residual, C)

    # Kernel-form intercept (Eq. 12.42 analogue): no w is formed.
    alpha_tol = max(1e-6, min(C * 1e-4, 1e-3))
    free = (a > alpha_tol) & (a < C - alpha_tol)
    Ky = K @ (a * y)  # sum_j alpha_j y_j K(x_j, ·)
    if free.any():
        b = float(np.mean(y[free] - Ky[free]))
    else:
        up = ((y == 1) & (a < C - alpha_tol)) | ((y == -1) & (a > alpha_tol))
        low = ((y == 1) & (a > alpha_tol)) | ((y == -1) & (a < C - alpha_tol))
        b = float((np.max(residual[up]) + np.min(residual[low])) / 2)

    dual_value = float(a.sum() - 0.5 * (a * y) @ Ky)
    return {
        "b": b,
        "alpha": a,
        "y": y,
        "iterations": iterations,
        "converged": converged,
        "kkt_gap": gap,
        "equality_residual": float(abs(y @ a)),
        "support_vectors": int(np.sum(a > alpha_tol)),
        "margin_support_vectors": int(np.sum(free)),
        "dual_objective_max": dual_value,
        "alpha_tol": alpha_tol,
    }


def decision_scores(K_rows: np.ndarray, model: dict) -> np.ndarray:
    """Kernel decision function f(x) = sum_i alpha_i y_i k(x_i, x) + b.

    Parameters
    ----------
    K_rows : array, shape (n_query, n_train)
        Rows of kernel evaluations against the training set:
        K_rows[q, i] = k(x_query_q, x_train_i).
    model : dict
        Output of solve_dual_kernel (needs alpha, y, b).

    Returns
    -------
    scores : array, shape (n_query,)
    """
    return K_rows @ (model["alpha"] * model["y"]) + model["b"]


def predict_labels(K_rows: np.ndarray, model: dict) -> np.ndarray:
    """Predict 0/1 labels from kernel decision scores (sign convention)."""
    return (decision_scores(K_rows, model) >= 0).astype(np.int8)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Binary F1 for the positive (seizure) class."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp + fp == 0 or tp + fn == 0:
        return None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


__all__ = [
    "solve_dual_kernel",
    "decision_scores",
    "predict_labels",
    "f1_score",
    "BasicPreprocessor",
    "measure",
    "read_csv",
    "TARGET",
    "kkt_gap",
]

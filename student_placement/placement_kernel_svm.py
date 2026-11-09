"""Kernel soft-margin dual SVM for the student placement dataset.

Same two-coordinate working-set dual solver as kernel_dual_svm.py (epilepsy),
but self-contained: the placement CSVs are already one-hot encoded and
standardized (train_processed.csv / test_processed.csv), so no preprocessor
or Model-team import is needed. Labels in the CSV are -1/+1.

    Dual:  max_alpha  sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j K_ij
           s.t.       0 <= alpha_i <= C,   sum_i alpha_i y_i = 0

Prediction: f(x) = sum_i alpha_i y_i k(x_i, x) + b   (no weight vector w).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

LABEL = "label"


def read_processed_csv(path: Path):
    """Read a processed placement CSV. Returns X (n,d), y in {-1,+1}, names."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if LABEL not in header:
            raise ValueError(f"'{LABEL}' column not found in {path}")
        li = header.index(LABEL)
        rows = [r for r in reader if r]
    data = np.asarray(rows, dtype=np.float64)
    y = data[:, li].astype(np.int8)
    if not set(np.unique(y)) <= {-1, 1}:
        raise ValueError("Labels must be -1/+1")
    X = np.delete(data, li, axis=1)
    names = [h for i, h in enumerate(header) if i != li]
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite feature values in {path}")
    return X, y, names


def stratified_subsample(y: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Indices of a class-proportional random subsample of size ~n (sorted)."""
    rng = np.random.default_rng(seed)
    if n >= len(y):
        return np.arange(len(y))
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        k = max(1, int(round(n * len(members) / len(y))))
        idx.append(rng.choice(members, size=min(k, len(members)), replace=False))
    return np.sort(np.concatenate(idx))


def kkt_gap(a, y, residual, C, eps_scale: float = 1e-10) -> float:
    """Max violating-pair gap (<= 0 means KKT satisfied)."""
    eps = eps_scale * max(C, 1)
    upper = ((y == 1) & (a < C - eps)) | ((y == -1) & (a > eps))
    lower = ((y == 1) & (a > eps)) | ((y == -1) & (a < C - eps))
    if not upper.any() or not lower.any():
        return 0.0
    return float(np.max(residual[upper]) - np.min(residual[lower]))


def solve_dual_kernel(K: np.ndarray, y_pm1: np.ndarray, C: float = 1.0,
                      tolerance: float = 0.003, max_steps: int = 150000):
    """Solve the soft-margin dual SVM on a precomputed training Gram matrix.

    y_pm1 : labels in {-1,+1}. Returns dict with alpha, b and diagnostics.
    """
    if C <= 0 or tolerance <= 0:
        raise ValueError("C and tolerance must be positive")
    K = np.asarray(K, dtype=np.float64)
    y = np.asarray(y_pm1, dtype=float)
    n = len(y)
    if K.shape != (n, n):
        raise ValueError(f"K must be ({n}, {n}); got {K.shape}")
    if not np.isfinite(K).all():
        raise ValueError("Kernel matrix contains non-finite values")
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes (-1 and +1) must be present")

    diagonal = np.diag(K).copy()
    a = np.zeros(n)
    residual = y.copy()  # y_i - sum_j alpha_j y_j K_ij
    converged = False
    step = 0
    for step in range(max_steps):
        eps = 1e-10 * max(C, 1)
        upper = ((y == 1) & (a < C - eps)) | ((y == -1) & (a > eps))
        lower = ((y == 1) & (a > eps)) | ((y == -1) & (a < C - eps))
        upper_scores = np.where(upper, residual, -np.inf)
        lower_scores = np.where(lower, residual, np.inf)
        i = int(np.argmax(upper_scores))
        if upper_scores[i] - np.min(lower_scores) <= tolerance:
            converged = True
            break

        curvature = np.maximum(diagonal[i] + diagonal - 2 * K[i], 1e-12)
        change = np.maximum(residual[i] - residual, 0)
        decrease = np.where(lower, change * change / curvature, -np.inf)
        j = int(np.argmax(decrease))
        s = y[i] * y[j]

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

    alpha_tol = max(1e-6, min(C * 1e-4, 1e-3))
    free = (a > alpha_tol) & (a < C - alpha_tol)
    Ky = K @ (a * y)
    if free.any():
        b = float(np.mean(y[free] - Ky[free]))
    else:
        up = ((y == 1) & (a < C - alpha_tol)) | ((y == -1) & (a > alpha_tol))
        low = ((y == 1) & (a > alpha_tol)) | ((y == -1) & (a < C - alpha_tol))
        b = float((np.max(residual[up]) + np.min(residual[low])) / 2)

    return {
        "b": b, "alpha": a, "y": y,
        "iterations": iterations, "converged": converged, "kkt_gap": gap,
        "equality_residual": float(abs(y @ a)),
        "support_vectors": int(np.sum(a > alpha_tol)),
        "margin_support_vectors": int(np.sum(free)),
        "dual_objective_max": float(a.sum() - 0.5 * (a * y) @ Ky),
        "alpha_tol": alpha_tol,
    }


def decision_scores(K_rows: np.ndarray, model: dict) -> np.ndarray:
    """f(x) = sum_i alpha_i y_i k(x_i, x) + b; K_rows is (n_query, n_train)."""
    return K_rows @ (model["alpha"] * model["y"]) + model["b"]


def predict_labels(K_rows: np.ndarray, model: dict) -> np.ndarray:
    """Predict labels in {-1,+1}."""
    return np.where(decision_scores(K_rows, model) >= 0, 1, -1).astype(np.int8)


def measure(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision/recall/F1 for the positive (+1) class, specificity."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))
    div = lambda a, b: a / b if b else None
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and precision + recall > 0 else None)
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision_pos": precision, "recall_pos": recall, "f1_pos": f1,
        "specificity_neg": div(tn, tn + fp),
        "confusion_matrix_TN_FP_FN_TP": [tn, fp, fn, tp],
    }

"""MATH 582: basic LINEAR, soft-margin DUAL SVM (model team only).

Mathematics for Machine Learning, Deisenroth et al., Ch. 12, Sec. 12.3:
    min_alpha  0.5 * alpha.T @ Q @ alpha - 1.T @ alpha
    subject to 0 <= alpha_i <= C,   y.T @ alpha = 0
    Q_ij = y_i * y_j * dot(x_i, x_j)

Train using the FULL supplied synthetic dataset (5,000 training rows).
The quadratic program is solved with a two-coordinate working-set method:
each iteration minimizes the exact convex quadratic along a feasible pair.
No sklearn.svm, nonlinear kernels, kernel comparison, or model tuning.

Requirements: numpy; use a standard Python 3.10+ environment.
Place this file beside Training_Data.csv and Testing_Data.csv, then run:
    python epilepsy_linear_dual_svm.py

Optional faster demonstration:
    python epilepsy_linear_dual_svm.py --train-rows 250
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

TARGET = "Had a Seizure Two Weeks After Survey"
ID = "Record ID"
NOMINAL = {"Sex": [1, 2], "Type of Epilepsy": list(range(1, 8))}


def read_csv(file: Path):
    with file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or TARGET not in reader.fieldnames:
            raise ValueError(f"Missing target column in {file.name}")
        features = [c for c in reader.fieldnames if c not in (ID, TARGET)]
        rows = list(reader)
    X = np.array([[float(row[c]) for c in features] for row in rows], dtype=np.float64)
    y01 = np.array([int(row[TARGET]) for row in rows], dtype=np.int8)
    if not np.isfinite(X).all() or not np.isin(y01, [0, 1]).all():
        raise ValueError("Features must be finite numbers and labels must be 0 or 1")
    return X, y01, features


class BasicPreprocessor:
    """One-hot encode two nominal variables; standardize remaining inputs.

    The means/SDs are computed on TRAINING only; test labels are never used.
    No nonlinear kernel or handcrafted nonlinear features are involved.
    """
    def __init__(self, names):
        self.names = names
        self.numeric_positions = [i for i, n in enumerate(names) if n not in NOMINAL]
        self.category_positions = {n: names.index(n) for n in NOMINAL}

    def fit(self, X):
        numeric = X[:, self.numeric_positions]
        self.means = numeric.mean(axis=0)
        self.stds = numeric.std(axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        numeric = (X[:, self.numeric_positions] - self.means) / self.stds
        one_hot = [
            (X[:, self.category_positions[name], None] == np.array(codes)).astype(float)
            for name, codes in NOMINAL.items()
        ]
        return np.column_stack([numeric, *one_hot])


def kkt_gap(alpha, y, residual, C):
    """Largest KKT violation for the box-constrained equality dual."""
    eps = 1e-10 * max(C, 1)
    upper = ((y == 1) & (alpha < C - eps)) | ((y == -1) & (alpha > eps))
    lower = ((y == 1) & (alpha > eps)) | ((y == -1) & (alpha < C - eps))
    return max(0.0, float(np.max(residual[upper]) - np.min(residual[lower])))


def solve_dual_linear(X, y01, C=1.0, tolerance=0.003, max_steps=150000):
    """Solve the textbook's convex QP by repeated exact feasible pair steps.

    Feasibility is maintained: alpha stays in [0,C] and y.T @ alpha == 0.
    Let K = X @ X.T, Q_ij = y_i*y_j*K_ij. Only the LINEAR dot product is used.
    The update of the two alphas is the minimizer along the chosen feasible
    line segment. The KKT gap is used as the stopping certificate.
    """
    if C <= 0 or tolerance <= 0:
        raise ValueError("C and tolerance must be positive")
    y = 2*y01.astype(float) - 1
    if len(np.unique(y)) != 2:
        raise ValueError("Both seizure labels 0 and 1 must be present")
    n = len(y)
    K = X @ X.T
    diagonal = np.diag(K).copy()
    a = np.zeros(n, dtype=float)
    residual = y.copy()  # residual_i = y_i - sum_j alpha_j*y_j*K_ij
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

        # Find a feasible second coordinate with large quadratic decrease.
        curvature = np.maximum(diagonal[i] + diagonal - 2*K[i], 1e-12)
        change = np.maximum(residual[i] - residual, 0)
        decrease = np.where(lower, change * change / curvature, -np.inf)
        j = int(np.argmax(decrease))
        s = y[i]*y[j]

        # alpha_i' = alpha_i + d, alpha_j' = alpha_j - s*d;
        # consequently y_i*alpha_i + y_j*alpha_j is unchanged.
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
        a[j] -= s*d
        residual -= (d*y[i]) * (K[:, i] - K[:, j])

    iterations = step if converged else step + 1
    gap = kkt_gap(a, y, residual, C)
    w = X.T @ (a*y)  # Eq. (12.38)

    # Recover intercept with book Eq. (12.42) at 0 < alpha_i < C.
    alpha_tol = max(1e-6, min(C*1e-4, 1e-3))
    free = (a > alpha_tol) & (a < C - alpha_tol)
    if free.any():
        b = float(np.mean(y[free] - X[free] @ w))
    else:
        # When no strictly interior support vector exists, use the KKT
        # admissible-intercept interval's midpoint.
        up = ((y == 1) & (a < C-alpha_tol)) | ((y == -1) & (a > alpha_tol))
        low = ((y == 1) & (a > alpha_tol)) | ((y == -1) & (a < C-alpha_tol))
        b = float((np.max(residual[up]) + np.min(residual[low]))/2)

    dual_value = float(a.sum() - 0.5 * (a*y) @ (K @ (a*y)))
    return {
        "w": w, "b": b, "alpha": a, "iterations": iterations,
        "converged": converged, "kkt_gap": gap,
        "equality_residual": float(abs(y@a)),
        "support_vectors": int(np.sum(a > alpha_tol)),
        "margin_support_vectors": int(np.sum(free)),
        "dual_objective_max": dual_value,
    }


def measure(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {
        "accuracy": (tn + tp)/len(y_true),
        "recall_seizure_1": tp/(tp + fn) if tp + fn else None,
        "specificity_no_seizure_0": tn/(tn + fp) if tn + fp else None,
        "precision_seizure_1": tp/(tp + fp) if tp + fp else None,
        "confusion_matrix_TN_FP_FN_TP": [tn, fp, fn, tp],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--train-rows", type=int, default=None,
                        help="Optional first N training rows; default all 5000 rows")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Soft-margin penalty; default fixed C=1 (no tuning)")
    parser.add_argument("--tolerance", type=float, default=0.003)
    args = parser.parse_args()
    X_full, y_full, names = read_csv(args.data_dir / "Training_Data.csv")
    X_test, y_test, test_names = read_csv(args.data_dir / "Testing_Data.csv")
    if names != test_names:
        raise ValueError("Training and test feature columns differ")
    if args.train_rows is not None:
        if not 2 <= args.train_rows <= len(y_full):
            parser.error(f"--train-rows must be 2..{len(y_full)}")
        X_full, y_full = X_full[:args.train_rows], y_full[:args.train_rows]
    prep = BasicPreprocessor(names).fit(X_full)
    X_train = prep.transform(X_full)
    X_holdout = prep.transform(X_test)
    model = solve_dual_linear(X_train, y_full, C=args.C, tolerance=args.tolerance)
    y_pred = (X_holdout @ model["w"] + model["b"] >= 0).astype(np.int8)
    # Persist the learned linear classifier and its training-only preprocessing.
    np.savez_compressed(
        args.data_dir / "linear_svm_model.npz",
        weights=model["w"], bias=np.array(model["b"]),
        raw_feature_names=np.array(names),
        numeric_means=prep.means, numeric_stds=prep.stds,
        numeric_positions=np.array(prep.numeric_positions),
    )
    metrics = measure(y_test, y_pred)
    report = {
        "description": "Linear soft-margin dual SVM; synthetic dataset; not clinical",
        "train_rows": len(y_full), "heldout_test_rows": len(y_test),
        "raw_predictors": len(names), "encoded_predictors": X_train.shape[1],
        "C": args.C, "tolerance": args.tolerance,
        "solver": "feasible two-coordinate quadratic working set",
        "converged": model["converged"], "iterations": model["iterations"],
        "kkt_gap": model["kkt_gap"],
        "equality_residual": model["equality_residual"],
        "support_vectors": model["support_vectors"],
        "margin_support_vectors": model["margin_support_vectors"],
        "dual_objective_max": model["dual_objective_max"],
        "intercept": model["b"], "metrics": metrics,
    }
    print(json.dumps(report, indent=2))
    if not model["converged"]:
        print("WARNING: QP did not meet requested KKT tolerance; do not claim convergence.")
    (args.data_dir / "linear_svm_results.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()

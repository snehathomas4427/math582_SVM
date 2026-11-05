# Kernel Team — Soft-Margin Dual SVM with Nonlinear Kernels

**MATH 582 Mini-Project 2 · Synthetic Epilepsy Dataset**  
Educational use only. Results on this synthetic dataset have **no clinical significance**.

## Overview

The Model team implemented a **linear soft-margin dual SVM** with a custom two-coordinate quadratic optimizer (`epilepsy_linear_dual_svm.py`). The Kernel team extends that dual formulation to **nonlinear kernels** by:

1. Implementing vectorized NumPy kernels (linear, polynomial, RBF).
2. Feeding a **precomputed Gram matrix** \(K_{ij}=k(x_i,x_j)\) into the same dual optimizer.
3. Predicting with kernel expansions instead of an explicit weight vector \(w\).

The Model team’s original file is **unchanged**. Kernel-enabled training lives in `kernel_dual_svm.py`.

---

## 1. What a kernel does in an SVM

A soft-margin dual SVM maximizes

\[
\max_{\alpha}\sum_{i=1}^{n}\alpha_i
-\frac12\sum_{i=1}^{n}\sum_{j=1}^{n}
\alpha_i\alpha_j y_i y_j\,k(x_i,x_j)
\]

subject to \(0\le\alpha_i\le C\) and \(\sum_i\alpha_i y_i=0\).

The decision function depends on the data only through inner products in some feature space \(\Phi\):

\[
k(x,z)=\langle\Phi(x),\Phi(z)\rangle.
\]

A **kernel** evaluates that inner product **without constructing \(\Phi(x)\) explicitly**. This is the kernel trick: the optimizer and the predictor both use \(k\) (or the Gram matrix \(K\)) rather than coordinates in \(\Phi\).

---

## 2. Why nonlinear kernels are useful

A linear SVM separates classes with a hyperplane in the original feature space. When class boundaries are curved or interactions matter, a linear separator can underfit.

Nonlinear kernels implicitly map inputs into a higher-dimensional (sometimes infinite-dimensional) space where a **linear** separator in \(\Phi\)-space corresponds to a **nonlinear** boundary in the original space. Polynomial kernels capture multiplicative feature interactions; RBF kernels allow highly local, flexible boundaries controlled by bandwidth \(\gamma\).

Kernels are not automatically better: a poorly chosen \(\gamma\) or degree can overfit or collapse predictions (see RBF \(\gamma=1.0\) below).

---

## 3. Mathematical definitions

| Kernel | Formula | Parameters |
|--------|---------|------------|
| Linear | \(k(x,z)=x^\top z\) | — |
| Polynomial | \(k(x,z)=(\gamma\,x^\top z+c_0)^d\) | \(\gamma>0\), degree \(d\), coef0 \(c_0\) |
| RBF (Gaussian) | \(k(x,z)=\exp(-\gamma\|x-z\|^2)\) | \(\gamma>0\) |

Default \(\gamma=1/d\) when unspecified (\(d=\) number of encoded features).

---

## 4. Implementation (`kernels.py`)

All kernels accept feature matrices `X` (\(n_x\times d\)) and `Z` (\(n_z\times d\)) and return a Gram matrix of shape `(n_x, n_z)` via vectorized NumPy:

- **Linear:** `X @ Z.T`
- **Polynomial:** `(gamma * (X @ Z.T) + coef0) ** degree`
- **RBF:** \(\|x-z\|^2=\|x\|^2+\|z\|^2-2x^\top z\), then `exp(-gamma * sq_dists)`, with tiny negative distances clamped to 0 for numerical stability

Factory: `get_kernel(name, **params)` returns a callable `kernel(X, Z)`.

---

## 5. Passing the kernel matrix to the dual optimizer

`solve_dual_kernel(K, y01, C, ...)` in `kernel_dual_svm.py` is the Model team’s two-coordinate working-set QP solver with one intentional interface change: it takes a **precomputed** training Gram matrix instead of forming \(K=XX^\top\) internally.

```text
K_train = kernel(X_train, X_train)     # (n, n)
model   = solve_dual_kernel(K_train, y_train, C=C)
```

### What changed relative to `solve_dual_linear` (and why)

| Change | Why |
|--------|-----|
| Input is \(K\), not \(X\) | Nonlinear kernels have no explicit \(\Phi(X)\) |
| No weight vector \(w\) | \(w=\Phi(X)^\top(\alpha\odot y)\) is unavailable for RBF/poly |
| Intercept via \(b=\mathrm{mean}_i\bigl[y_i-\sum_j\alpha_j y_j K_{ij}\bigr]\) on free SVs | Same identity as Model Eq. (12.42) when \(K=XX^\top\) |
| Coordinate updates, KKT gap, box/equality feasibility | **Unchanged** |

Verified consistency: on a 300-row subset, linear-kernel dual coefficients, intercept, and holdout predictions match `solve_dual_linear` to numerical precision (prediction disagreement = 0).

---

## 6. Prediction vs. the linear SVM

**Linear (Model team):**

\[
f(x)=w^\top x+b,\qquad w=X^\top(\alpha\odot y).
\]

**Kernel (this team):**

\[
f(x)=\sum_{i=1}^{n}\alpha_i y_i\,k(x_i,x)+b.
\]

In code:

```text
K_test = kernel(X_test, X_train)       # (n_test, n_train)
scores = K_test @ (alpha * y) + b
y_hat  = (scores >= 0).astype(int)
```

Labels in the CSV are \(0/1\); the dual uses \(y\in\{-1,+1\}\) via \(y=2y_{01}-1\), identical to the Model team.

---

## 7. Experiments and parameter choices

**Data / preprocessing (shared with Model team):**

- Target: `Had a Seizure Two Weeks After Survey`
- Train: 5,000 rows (class 0: 2,819; class 1: 2,181; positive rate ≈ 43.6%)
- Test: 5,000 rows (class 0: 2,732; class 1: 2,268; positive rate ≈ 45.4%)
- `BasicPreprocessor`: one-hot `Sex` and `Type of Epilepsy`; standardize remaining numerics using **training** means/SDs only → 33 encoded features
- Same holdout for every configuration

**Primary grid** (\(C=1.0\) fixed to isolate kernel effect):

| Config | Kernel | Parameters |
|--------|--------|------------|
| `linear` | Linear | default |
| `poly_d2` | Polynomial | degree 2, \(\gamma=1/33\), \(c_0=1\) |
| `poly_d3` | Polynomial | degree 3, \(\gamma=1/33\), \(c_0=1\) |
| `rbf_g0.01` | RBF | \(\gamma=0.01\) |
| `rbf_g0.1` | RBF | \(\gamma=0.1\) |
| `rbf_g1.0` | RBF | \(\gamma=1.0\) |

**Follow-up** (around best primary nonlinear = RBF \(\gamma=0.01\)): \(C\in\{0.1,10\}\), and nearby \(\gamma\in\{\approx0.00333,\,0.03\}\) at \(C=1\).

Optimizer tolerance: KKT gap \(0.003\) (Model team default).

---

## 8. Results and comparison

Primary holdout metrics (\(C=1.0\); all configs **converged**):

| Config | Accuracy | Precision (1) | Recall (1) | F1 (1) | # SVs | Time (s) |
|--------|----------|---------------|------------|--------|-------|----------|
| **linear** | **0.7534** | **0.7410** | **0.7015** | **0.7207** | 2892 | 6.9 |
| poly_d2 | 0.7450 | 0.7306 | 0.6936 | 0.7116 | 2916 | 1.1 |
| poly_d3 | 0.7354 | 0.7219 | 0.6777 | 0.6991 | 3006 | 2.0 |
| rbf_g0.01 | 0.7524 | 0.7427 | 0.6949 | 0.7180 | 3048 | 0.5 |
| rbf_g0.1 | 0.7456 | 0.7406 | 0.6759 | 0.7068 | 4246 | 0.7 |
| rbf_g1.0 | 0.5464 | — | 0.0000 | — | 5000 | 1.2 |

Follow-up highlights:

| Config | Accuracy | Recall (1) | F1 (1) |
|--------|----------|------------|--------|
| rbf \(\gamma=0.01\), \(C=0.1\) | 0.7548 | 0.6803 | 0.7157 |
| rbf \(\gamma=0.01\), \(C=10\) | 0.7428 | 0.6900 | 0.7088 |
| rbf \(\gamma\approx0.00333\), \(C=1\) | 0.7546 | 0.6958 | 0.7201 |
| rbf \(\gamma=0.03\), \(C=1\) | 0.7450 | 0.6887 | 0.7102 |

**Class imbalance note.** The positive (seizure) class is a mild minority (~44% train). Overall accuracy can hide poor seizure detection; we therefore emphasize **recall** and **F1 for class 1**. Predicting all negatives on this test set would yield accuracy ≈ 0.546 (exactly what RBF \(\gamma=1.0\) achieved).

Artifacts: `results/kernel_experiment_results.{json,csv}`, `metric_comparison.png`, `best_confusion_matrix.png`, `support_vector_counts.png`, `recall_vs_training_time.png`.

---

## 9. Which kernel performed best?

- **Best primary overall:** **linear** (highest F1 0.7207, highest seizure recall 0.7015, accuracy 0.7534). This exactly reproduces the Model team’s published linear metrics, confirming the kernel path is consistent when \(k(x,z)=x^\top z\).
- **Best primary nonlinear:** **RBF with \(\gamma=0.01\)** (F1 0.7180, accuracy 0.7524)—very close to linear, with far fewer optimizer iterations.
- **Best among follow-ups:** RBF \(\gamma\approx0.00333\), \(C=1\) (F1 0.7201), essentially tying the linear baseline.

**Interpretation.** After one-hot encoding and standardization, the synthetic epilepsy features appear **approximately linearly separable** at this soft-margin budget. Mild nonlinear kernels (poly-2, RBF with small \(\gamma\)) do not beat linear; stronger nonlinearity (poly-3, larger \(\gamma\)) hurts. RBF \(\gamma=1.0\) is pathologically peaked: every training point is a support vector and the classifier predicts **all negatives** on the holdout (recall 0).

---

## 10. Limitations and improvements

- **Computational:** Training Gram matrices are \(n\times n\) (\(5000^2\) doubles ≈ 200 MB). Prediction costs \(O(n_{\mathrm{test}}\cdot n_{\mathrm{SV}}\cdot d)\) kernel evaluations. Sparse \(\alpha\) / reduced SV sets would help.
- **Hyperparameters:** Only a small grid; no nested CV. \(\gamma=1/d\) for polynomials is a conventional default, not tuned.
- **Imbalance:** No class-weighted \(C\) or resampling; soft-margin \(C\) is shared across classes.
- **Optimizer:** Inherits Model-team coordinate descent; large \(C\) or ill-conditioned kernels can need many iterations (linear used 76k steps here).
- **Scope:** Synthetic educational data only—do not claim clinical validity.

Possible improvements: class-weighted penalties, validation-based \(\gamma\)/degree selection, caching kernel columns for prediction, and Nyström / sparse approximations for larger \(n\).

---

## Project report blurb (Kernel Team)

> The Kernel team extended the Model team’s linear soft-margin dual SVM to nonlinear kernels without replacing the custom dual optimizer. Vectorized NumPy implementations of the linear, polynomial, and RBF kernels produce Gram matrices that are passed directly to a kernel-aware copy of the Model team’s two-coordinate QP solver. Predictions use the dual expansion \(f(x)=\sum_i\alpha_i y_i k(x_i,x)+b\) rather than an explicit weight vector. On the synthetic epilepsy holdout with shared Model-team preprocessing and fixed \(C=1\), the linear kernel remained strongest (accuracy 0.7534, seizure recall 0.7015, F1 0.7207). The best nonlinear configuration was RBF with \(\gamma=0.01\) (F1 0.7180). Larger RBF bandwidth \(\gamma=1.0\) collapsed to predicting no seizures. A small follow-up around RBF \(\gamma=0.01\) did not materially beat the linear baseline, suggesting that after encoding and standardization the synthetic decision boundary is nearly linear under this soft-margin regime. Results are educational only and have no clinical interpretation.

---

## Repository layout

```text
kernel_team/
  kernels.py                 # linear, polynomial, RBF kernels
  kernel_dual_svm.py         # dual solver on precomputed K + prediction
  run_epilepsy_experiments.py
  tests/test_kernels.py
  results/                   # metrics, plots, best model
  README.md
```

Unmodified Model-team entry point: `epilepsy_linear_dual_svm.py`.

---

## How to rerun

From the repository root (Python 3.10+, NumPy, Matplotlib):

```bash
# Unit tests
python -m unittest kernel_team.tests.test_kernels -v

# Full kernel comparison (writes kernel_team/results/)
python -m kernel_team.run_epilepsy_experiments

# Faster demo on a training subset
python -m kernel_team.run_epilepsy_experiments --train-rows 500

# Original Model-team linear baseline
python epilepsy_linear_dual_svm.py
```

---

## Coordination notes

- **Model team:** Optimizer logic preserved; only a thin kernel-matrix interface was added in a separate module. Linear baseline file and metrics remain the reference.
- **Data team:** Experiments use `Training_Data.csv` / `Testing_Data.csv` and the Model team’s preprocessor. Confirm whether `datasets/train_processed.csv` / `test_processed.csv` are intended as an alternate pipeline; Kernel team did **not** use those files, to stay consistent with the Model baseline.

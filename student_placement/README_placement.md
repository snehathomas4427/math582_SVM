# Kernel Team — Soft-Margin Dual SVM with Nonlinear Kernels (Student Placement)

**MATH 582 Mini-Project 2 · Student Placement Dataset**
Educational use only.

## Overview

This is the placement-dataset counterpart of the epilepsy Kernel Team work. The same three ideas carry over:

1. Vectorized NumPy kernels (linear, polynomial, RBF) in `kernels.py` — **unchanged**, they are dataset-independent.
2. A **precomputed Gram matrix** \(K_{ij}=k(x_i,x_j)\) fed to the two-coordinate working-set dual solver.
3. Prediction by kernel expansion \(f(x)=\sum_i\alpha_i y_i k(x_i,x)+b\), with no weight vector \(w\).

The kernel math, dual objective, and prediction rule are exactly those in the epilepsy README (§1–§3, §6); this document covers what is specific to the placement data.

---

## 1. What differs from the epilepsy pipeline

| Aspect | Epilepsy | Placement |
|--------|----------|-----------|
| Input files | `Training_Data.csv`, `Testing_Data.csv` (raw) | `train_processed.csv`, `test_processed.csv` (already standardized + one-hot) |
| Preprocessing | Model-team `BasicPreprocessor` | None needed; 20 features, no missing values |
| Label coding | 0/1, converted with \(y=2y_{01}-1\) | **−1/+1 already**, used directly in the dual |
| Positive class | seizure (1) | label \(+1\) (assumed to be "placed") |
| Training rows | 5,000 | 45,000 available → **5,000 stratified subsample** |
| Solver module | `kernel_dual_svm.py` (imports Model-team file) | `placement_kernel_svm.py` (self-contained) |
| Positive rate | ≈ 43.6% train / 45.4% test | ≈ 36.2% train / 36.2% test |

**Why a subsample.** The solver works on a dense \(n\times n\) Gram matrix. With \(n=45{,}000\) that is \(45000^2\) doubles ≈ 16 GB, so by default `--train-rows 5000` draws a class-proportional random subsample (seeded; `--seed`). The test set (5,000 rows) is used in full. Training on more rows is possible via `--train-rows` if memory allows.

---

## 2. Implementation

- `kernels.py` — linear, polynomial, RBF, and `get_kernel(name, **params)`.
- `placement_kernel_svm.py`
  - `read_processed_csv`, `stratified_subsample`
  - `solve_dual_kernel(K, y_pm1, C, tolerance, max_steps)` — same coordinate-pair update, KKT-gap stopping rule, box/equality feasibility, free-SV intercept, and dual objective as `kernel_dual_svm.py`; takes ±1 labels.
  - `decision_scores`, `predict_labels` (returns ±1), `measure` (accuracy, precision/recall/F1 for \(+1\), specificity, confusion matrix), `kkt_gap`
- `run_placement_experiments.py` — primary grid + follow-up sweep, writes `results/`.

```text
K_train = kernel(X_train, X_train)     # (n, n)
model   = solve_dual_kernel(K_train, y_train_pm1, C=C)
K_test  = kernel(X_test, X_train)      # (n_test, n_train)
y_hat   = where(K_test @ (alpha*y) + b >= 0, +1, -1)
```

**Verification.** Unit tests check dual feasibility (\(0\le\alpha\le C\), \(|y^\top\alpha|\approx0\)), shapes, error handling, metrics, and CSV reading. In addition, on a 1,500-row training subset the RBF (\(\gamma=0.1\), \(C=1\)) predictions on the 5,000-row holdout agreed 100% with scikit-learn's `SVC` (the test suite includes a ≥98% agreement check on synthetic data when scikit-learn is installed). Unlike the epilepsy README, no comparison against `solve_dual_linear` is claimed here because the Model-team file was not part of this pipeline.

---

## 3. Experiments and parameter choices

- Train used: 5,000 rows (label −1: 3,188; label +1: 1,812; positive rate 36.2%), subsampled from 45,000 (28,688 / 16,312).
- Test: 5,000 rows (−1: 3,188; +1: 1,812; positive rate 36.2%).
- 20 features: Age, CGPA, Internships, Projects, Coding_Skills, Communication_Skills, Aptitude_Test_Score, Soft_Skills_Rating, Certifications, Backlogs, plus one-hot Gender, Degree and Branch columns.
- Same holdout for every configuration. KKT tolerance 0.003.

**Primary grid** (\(C=1.0\)): `linear`; `poly_d2`, `poly_d3` (\(\gamma=1/20\), \(c_0=1\)); `rbf_g0.01`, `rbf_g0.1`, `rbf_g1.0`.

**Follow-up** (around best primary nonlinear = RBF \(\gamma=0.1\)): \(C\in\{0.1,10\}\) and \(\gamma\in\{\approx0.0333,\,0.3\}\) at \(C=1\).

---

## 4. Results

Primary holdout metrics (\(C=1.0\); all configs **converged**). Times are wall-clock on the machine used to generate these artifacts.

| Config | Accuracy | Precision (+1) | Recall (+1) | F1 (+1) | # SVs | Time (s) |
|--------|----------|----------------|-------------|---------|-------|----------|
| linear | 0.8720 | 0.8202 | 0.8284 | 0.8243 | 1664 | 2.4 |
| poly_d2 | 0.9566 | 0.9365 | 0.9443 | 0.9404 | 1077 | 0.8 |
| poly_d3 | 0.9674 | 0.9528 | 0.9575 | 0.9551 | 837 | 0.9 |
| rbf_g0.01 | 0.9244 | 0.8818 | 0.9139 | 0.8976 | 1544 | 0.6 |
| **rbf_g0.1** | **0.9700** | **0.9571** | **0.9603** | **0.9587** | 1020 | 0.7 |
| rbf_g1.0 | 0.7944 | 0.9851 | 0.4393 | 0.6076 | 4981 | 1.4 |

Follow-up:

| Config | Accuracy | Precision (+1) | Recall (+1) | F1 (+1) |
|--------|----------|----------------|-------------|---------|
| rbf \(\gamma=0.1\), \(C=0.1\) | 0.9460 | 0.9274 | 0.9233 | 0.9253 |
| rbf \(\gamma=0.1\), \(C=10\) | **0.9792** | 0.9734 | 0.9691 | **0.9712** |
| rbf \(\gamma\approx0.0333\), \(C=1\) | 0.9578 | 0.9415 | 0.9421 | 0.9418 |
| rbf \(\gamma=0.3\), \(C=1\) | 0.9694 | 0.9637 | 0.9514 | 0.9575 |

**Class balance note.** The \(+1\) class is a minority (~36%). Predicting all \(-1\) would score accuracy 0.6376, so accuracy alone is again supplemented by precision, recall and F1 for \(+1\). Notably, `rbf_g1.0` (accuracy 0.7944) does **not** collapse to all-negative here, but it is still the worst config: precision 0.985 with recall 0.439 — it is very conservative about predicting \(+1\), and every training point (4,981 of 5,000) is a support vector.

Artifacts: `results/placement_kernel_results.{json,csv}`, `experiment_log.txt`, `metric_comparison.png`, `best_confusion_matrix.png`, `support_vector_counts.png`, `recall_vs_training_time.png`, `best_kernel_model.npz`.

---

## 5. Which kernel performed best?

- **Best primary:** **RBF \(\gamma=0.1\)** (accuracy 0.9700, recall 0.9603, F1 0.9587), narrowly ahead of poly-3 (F1 0.9551).
- **Best overall (follow-up):** RBF \(\gamma=0.1\), \(C=10\) (F1 0.9712, accuracy 0.9792).
- **Linear baseline:** F1 0.8243 — roughly **13–15 F1 points** below the best nonlinear kernels.

**Interpretation.** Unlike the epilepsy data, where linear matched the nonlinear kernels, the placement data show a clear nonlinear structure: every reasonable nonlinear kernel (poly-2/3, RBF \(\gamma\in[0.03,0.3]\)) beats linear by a wide margin, and they also need fewer support vectors (≈ 800–1,100 vs 1,664). Moderate bandwidth is best; too small a \(\gamma\) (0.01) behaves almost linearly and loses accuracy, and too large (1.0) overfits locally. This is consistent with a label that depends on interactions between features (e.g., combinations of CGPA, skills, and backlogs) rather than a single weighted sum.

---

## 6. Limitations and improvements

- **Subsampling:** only 5,000 of 45,000 training rows are used because of the dense Gram matrix; accuracy may improve with more data (Nyström / random-feature approximations or a cached-column solver would allow this).
- **Hyperparameter selection on the test set.** As in the epilepsy work, the primary grid and the follow-up are compared on the same holdout, with no validation split or nested CV. The follow-up winner (\(C=10\)) in particular was chosen by looking at test metrics, so its 0.9792 is optimistic. A validation split (or CV) should be used before reporting a final number.
- **Single seed:** one random subsample; variation across seeds was not measured.
- **Imbalance:** no class-weighted \(C\) or resampling.
- **Label meaning:** \(+1\) is treated as the "placed" class; confirm with the Data team.
- **Scope:** educational data only.

---

## Project report blurb (Kernel Team — placement)

> We applied the same kernel-extended dual SVM to the student placement dataset, using the already-standardized processed files with ±1 labels and a 5,000-row class-stratified training subsample (a dense Gram matrix on all 45,000 rows would need ≈16 GB). With fixed \(C=1\), the linear kernel reached accuracy 0.8720 and F1 0.8243 for the positive class, while RBF with \(\gamma=0.1\) reached accuracy 0.9700 and F1 0.9587, and polynomial kernels of degree 2 and 3 reached F1 0.9404 and 0.9551. A follow-up sweep gave the best F1 of 0.9712 at \(\gamma=0.1\), \(C=10\), though that configuration was selected on the holdout and is optimistic. Very small \(\gamma\) behaved nearly linearly and very large \(\gamma\) (1.0) produced an over-conservative classifier (recall 0.439). In contrast with the epilepsy dataset, the placement decision boundary is clearly nonlinear.

---

## Repository layout

```text
student_placement/
  __init__.py
  kernels.py                     # shared, unchanged
  placement_kernel_svm.py        # dual solver on precomputed K, ±1 labels, metrics, CSV I/O
  run_placement_experiments.py
  tests/
    __init__.py
    test_kernels.py              # shared, unchanged
    test_placement_kernel_svm.py
  results/                       # metrics, plots, experiment_log.txt, best model
  README_placement.md
```

## How to rerun

The package is self-contained and works under any folder name; replace `student_placement` with your folder name if different . Run from the directory that contains that folder.

From the repository root (Python 3.10+, NumPy, Matplotlib), with `train_processed.csv` and `test_processed.csv` in `datasets`:

```bash
# Unit tests
python -m unittest student_placement.tests.test_kernels student_placement.tests.test_placement_kernel_svm -v

# Full comparison on a 5,000-row stratified training subsample
python -m student_placement.run_placement_experiments --data-dir datasets

# Faster demo / larger run
python -m student_placement.run_placement_experiments --data-dir datasets --train-rows 1500
python -m student_placement.run_placement_experiments --data-dir datasets --train-rows 10000
```
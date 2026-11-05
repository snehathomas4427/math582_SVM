"""Run kernel SVM experiments on the synthetic Epilepsy dataset.

Uses the Model team's CSV I/O and BasicPreprocessor unchanged. Solves the
soft-margin dual with a precomputed Gram matrix via solve_dual_kernel
(same two-coordinate optimizer as epilepsy_linear_dual_svm.py).

From the repository root:
    python -m kernel_team.run_epilepsy_experiments

Optional faster demonstration (subset of training rows):
    python -m kernel_team.run_epilepsy_experiments --train-rows 500

Outputs are written to kernel_team/results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernel_team.kernels import get_kernel
from kernel_team.kernel_dual_svm import (
    BasicPreprocessor,
    f1_score,
    measure,
    predict_labels,
    read_csv,
    solve_dual_kernel,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Primary comparison grid (C held at 1.0 to isolate kernel effect).
PRIMARY_CONFIGS = [
    {"name": "linear", "kernel": "linear", "params": {}, "C": 1.0},
    {
        "name": "poly_d2",
        "kernel": "polynomial",
        "params": {"degree": 2, "gamma": None, "coef0": 1.0},
        "C": 1.0,
    },
    {
        "name": "poly_d3",
        "kernel": "polynomial",
        "params": {"degree": 3, "gamma": None, "coef0": 1.0},
        "C": 1.0,
    },
    {"name": "rbf_g0.01", "kernel": "rbf", "params": {"gamma": 0.01}, "C": 1.0},
    {"name": "rbf_g0.1", "kernel": "rbf", "params": {"gamma": 0.1}, "C": 1.0},
    {"name": "rbf_g1.0", "kernel": "rbf", "params": {"gamma": 1.0}, "C": 1.0},
]


def _resolve_params(params: dict, n_features: int) -> dict:
    """Fill None gamma with the 1/n_features default used by the kernels."""
    out = dict(params)
    if "gamma" in out and out["gamma"] is None:
        out["gamma"] = 1.0 / n_features
    return out


def run_one(config: dict, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray, tolerance: float) -> dict:
    """Train and evaluate one kernel configuration."""
    resolved = _resolve_params(config["params"], X_train.shape[1])
    kernel_fn = get_kernel(config["kernel"], **resolved)

    t0 = time.perf_counter()
    K_train = kernel_fn(X_train, X_train)
    t_kernel_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = solve_dual_kernel(
        K_train, y_train, C=config["C"], tolerance=tolerance
    )
    t_optimize = time.perf_counter() - t1

    t2 = time.perf_counter()
    K_test = kernel_fn(X_test, X_train)
    y_pred = predict_labels(K_test, model)
    t_predict = time.perf_counter() - t2

    metrics = measure(y_test, y_pred)
    metrics["f1_seizure_1"] = f1_score(y_test, y_pred)

    train_time = t_kernel_train + t_optimize
    result = {
        "name": config["name"],
        "kernel": config["kernel"],
        "kernel_params": resolved,
        "C": config["C"],
        "tolerance": tolerance,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "encoded_features": int(X_train.shape[1]),
        "accuracy": metrics["accuracy"],
        "precision_seizure_1": metrics["precision_seizure_1"],
        "recall_seizure_1": metrics["recall_seizure_1"],
        "f1_seizure_1": metrics["f1_seizure_1"],
        "specificity_no_seizure_0": metrics["specificity_no_seizure_0"],
        "confusion_matrix_TN_FP_FN_TP": metrics["confusion_matrix_TN_FP_FN_TP"],
        "training_time_sec": train_time,
        "kernel_train_time_sec": t_kernel_train,
        "optimize_time_sec": t_optimize,
        "predict_time_sec": t_predict,
        "support_vectors": model["support_vectors"],
        "margin_support_vectors": model["margin_support_vectors"],
        "converged": model["converged"],
        "iterations": model["iterations"],
        "kkt_gap": model["kkt_gap"],
        "equality_residual": model["equality_residual"],
        "dual_objective_max": model["dual_objective_max"],
        "intercept": model["b"],
        "alpha": model["alpha"],
        "y_pm1": model["y"],
        "predictions": y_pred,
    }
    return result


def save_csv(results: list[dict], path: Path) -> None:
    fields = [
        "name", "kernel", "kernel_params", "C", "accuracy",
        "precision_seizure_1", "recall_seizure_1", "f1_seizure_1",
        "specificity_no_seizure_0", "confusion_matrix_TN_FP_FN_TP",
        "training_time_sec", "support_vectors", "margin_support_vectors",
        "converged", "iterations", "kkt_gap", "equality_residual",
        "dual_objective_max", "intercept",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            row["kernel_params"] = json.dumps(r["kernel_params"])
            row["confusion_matrix_TN_FP_FN_TP"] = json.dumps(
                r["confusion_matrix_TN_FP_FN_TP"]
            )
            writer.writerow(row)


def plot_metric_comparison(results: list[dict], path: Path) -> None:
    names = [r["name"] for r in results]
    metrics = ["accuracy", "precision_seizure_1", "recall_seizure_1", "f1_seizure_1"]
    labels = ["Accuracy", "Precision (seizure=1)", "Recall (seizure=1)", "F1 (seizure=1)"]
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [r[m] if r[m] is not None else 0.0 for r in results]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Kernel SVM metrics on synthetic Epilepsy holdout (C=1.0)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: list[int], title: str, path: Path) -> None:
    tn, fp, fn, tp = cm
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_support_vectors(results: list[dict], path: Path) -> None:
    names = [r["name"] for r in results]
    svs = [r["support_vectors"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, svs, color="#4C72B0")
    ax.set_ylabel("Number of support vectors")
    ax.set_title("Support vector count by kernel configuration")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_recall_vs_time(results: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        ax.scatter(r["training_time_sec"], r["recall_seizure_1"], s=60)
        ax.annotate(r["name"], (r["training_time_sec"], r["recall_seizure_1"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("Recall (seizure=1)")
    ax.set_title("Seizure recall vs training time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_best_nonlinear(results: list[dict]) -> dict:
    """Best nonlinear config by F1, then recall, then accuracy."""
    nonlinear = [r for r in results if r["kernel"] != "linear"]
    if not nonlinear:
        raise RuntimeError("No nonlinear results available")

    def key(r):
        f1 = r["f1_seizure_1"] if r["f1_seizure_1"] is not None else -1.0
        rec = r["recall_seizure_1"] if r["recall_seizure_1"] is not None else -1.0
        return (f1, rec, r["accuracy"])

    return max(nonlinear, key=key)


def followup_configs(best: dict) -> list[dict]:
    """Small C / gamma follow-up around the best nonlinear kernel."""
    configs = []
    if best["kernel"] == "rbf":
        gamma = best["kernel_params"]["gamma"]
        for C in (0.1, 1.0, 10.0):
            if abs(C - best["C"]) < 1e-12 and abs(gamma - best["kernel_params"]["gamma"]) < 1e-15:
                continue  # already in primary grid
            configs.append({
                "name": f"rbf_g{gamma}_C{C}",
                "kernel": "rbf",
                "params": {"gamma": gamma},
                "C": C,
            })
        # Nearby gamma at C=1
        for g in (gamma / 3.0, gamma * 3.0):
            if any(abs(g - r.get("kernel_params", {}).get("gamma", -1)) < 1e-12
                   for r in [{"kernel_params": best["kernel_params"]}]):
                pass
            configs.append({
                "name": f"rbf_g{g:g}_C1",
                "kernel": "rbf",
                "params": {"gamma": g},
                "C": 1.0,
            })
    elif best["kernel"] == "polynomial":
        degree = best["kernel_params"]["degree"]
        for C in (0.1, 10.0):
            configs.append({
                "name": f"poly_d{degree}_C{C}",
                "kernel": "polynomial",
                "params": {
                    "degree": degree,
                    "gamma": best["kernel_params"].get("gamma"),
                    "coef0": best["kernel_params"].get("coef0", 1.0),
                },
                "C": C,
            })
    # Deduplicate by name
    seen = set()
    unique = []
    for c in configs:
        if c["name"] not in seen:
            seen.add(c["name"])
            unique.append(c)
    return unique[:4]  # keep the follow-up small


def class_distribution(y: np.ndarray) -> dict:
    n = len(y)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    return {"n": n, "class_0": n0, "class_1": n1, "positive_rate": n1 / n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel Team epilepsy SVM experiments")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory containing Training_Data.csv and Testing_Data.csv",
    )
    parser.add_argument(
        "--train-rows",
        type=int,
        default=None,
        help="Optional first N training rows (default: all)",
    )
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument(
        "--skip-followup",
        action="store_true",
        help="Skip the small follow-up sweep on the best nonlinear kernel",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y_train, names = read_csv(args.data_dir / "Training_Data.csv")
    X_test_raw, y_test, test_names = read_csv(args.data_dir / "Testing_Data.csv")
    if names != test_names:
        raise ValueError("Training and test feature columns differ")
    if args.train_rows is not None:
        if not 2 <= args.train_rows <= len(y_train):
            parser.error(f"--train-rows must be 2..{len(y_train)}")
        X_raw = X_raw[: args.train_rows]
        y_train = y_train[: args.train_rows]

    # Same preprocessing pipeline as the Model team's linear baseline.
    prep = BasicPreprocessor(names).fit(X_raw)
    X_train = prep.transform(X_raw)
    X_test = prep.transform(X_test_raw)

    dist = {
        "train": class_distribution(y_train),
        "test": class_distribution(y_test),
        "note": (
            "Mild class imbalance (~44% positive in train). Accuracy alone can "
            "mislead; prioritize seizure-positive recall and F1."
        ),
    }
    print("Class distribution:", json.dumps(dist, indent=2))
    print(f"Encoded feature dim: {X_train.shape[1]}")
    print(f"Running {len(PRIMARY_CONFIGS)} primary kernel configurations...")

    primary_results = []
    for cfg in PRIMARY_CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        try:
            result = run_one(cfg, X_train, y_train, X_test, y_test, args.tolerance)
        except Exception as exc:
            print(f"FAILED: {cfg['name']}: {exc}")
            primary_results.append({
                "name": cfg["name"],
                "kernel": cfg["kernel"],
                "kernel_params": _resolve_params(cfg["params"], X_train.shape[1]),
                "C": cfg["C"],
                "error": str(exc),
                "failed": True,
            })
            continue
        primary_results.append(result)
        print(
            f"  acc={result['accuracy']:.4f}  "
            f"prec={result['precision_seizure_1']}  "
            f"rec={result['recall_seizure_1']}  "
            f"f1={result['f1_seizure_1']}  "
            f"SVs={result['support_vectors']}  "
            f"converged={result['converged']}  "
            f"iters={result['iterations']}  "
            f"time={result['training_time_sec']:.1f}s"
        )
        if not result["converged"]:
            print("  WARNING: did not meet KKT tolerance; do not claim convergence.")

    successful = [r for r in primary_results if not r.get("failed")]
    if not successful:
        raise SystemExit("All primary experiments failed; see errors above.")

    # Follow-up around best nonlinear kernel
    followup_results = []
    best_nl = pick_best_nonlinear(successful)
    print(f"\nBest nonlinear (primary): {best_nl['name']} "
          f"(F1={best_nl['f1_seizure_1']}, recall={best_nl['recall_seizure_1']})")

    if not args.skip_followup:
        for cfg in followup_configs(best_nl):
            # Skip configs already present in primary
            if any(c["name"] == cfg["name"] for c in PRIMARY_CONFIGS):
                continue
            print(f"\n=== follow-up {cfg['name']} ===")
            try:
                result = run_one(cfg, X_train, y_train, X_test, y_test, args.tolerance)
            except Exception as exc:
                print(f"FAILED: {cfg['name']}: {exc}")
                followup_results.append({
                    "name": cfg["name"], "kernel": cfg["kernel"],
                    "kernel_params": cfg["params"], "C": cfg["C"],
                    "error": str(exc), "failed": True,
                })
                continue
            followup_results.append(result)
            print(
                f"  acc={result['accuracy']:.4f}  "
                f"rec={result['recall_seizure_1']}  "
                f"f1={result['f1_seizure_1']}  "
                f"converged={result['converged']}  "
                f"time={result['training_time_sec']:.1f}s"
            )

    all_success = successful + [r for r in followup_results if not r.get("failed")]

    # Ranking for report: prefer F1 of seizure class among primary
    best_overall = max(
        successful,
        key=lambda r: (
            r["f1_seizure_1"] if r["f1_seizure_1"] is not None else -1,
            r["recall_seizure_1"] if r["recall_seizure_1"] is not None else -1,
            r["accuracy"],
        ),
    )

    # Strip large arrays from the archived JSON
    _drop = {"predictions", "alpha", "y_pm1"}
    archive = []
    for r in primary_results + followup_results:
        entry = {k: v for k, v in r.items() if k not in _drop}
        archive.append(entry)

    summary = {
        "description": (
            "Kernel soft-margin dual SVM comparison on synthetic Epilepsy data. "
            "Educational only; not clinical."
        ),
        "class_distribution": dist,
        "preprocessing": (
            "Model-team BasicPreprocessor: one-hot Sex and Type of Epilepsy; "
            "standardize remaining numerics using training statistics only."
        ),
        "label_encoding": "0/1 in CSV; dual uses y in {-1, +1} via 2*y01-1",
        "optimizer": (
            "Model-team feasible two-coordinate quadratic working set, "
            "adapted to accept precomputed K (see kernel_dual_svm.py)"
        ),
        "primary_results": archive[: len(primary_results)],
        "followup_results": archive[len(primary_results) :],
        "best_primary": {
            "name": best_overall["name"],
            "kernel": best_overall["kernel"],
            "kernel_params": best_overall["kernel_params"],
            "C": best_overall["C"],
            "accuracy": best_overall["accuracy"],
            "precision_seizure_1": best_overall["precision_seizure_1"],
            "recall_seizure_1": best_overall["recall_seizure_1"],
            "f1_seizure_1": best_overall["f1_seizure_1"],
            "confusion_matrix_TN_FP_FN_TP": best_overall[
                "confusion_matrix_TN_FP_FN_TP"
            ],
        },
        "best_nonlinear_primary": {
            "name": best_nl["name"],
            "kernel": best_nl["kernel"],
            "kernel_params": best_nl["kernel_params"],
            "C": best_nl["C"],
            "accuracy": best_nl["accuracy"],
            "precision_seizure_1": best_nl["precision_seizure_1"],
            "recall_seizure_1": best_nl["recall_seizure_1"],
            "f1_seizure_1": best_nl["f1_seizure_1"],
        },
    }

    results_json = args.results_dir / "kernel_experiment_results.json"
    results_csv = args.results_dir / "kernel_experiment_results.csv"
    results_json.write_text(json.dumps(summary, indent=2) + "\n")
    save_csv(all_success, results_csv)

    # Plots from successful primary runs
    plot_metric_comparison(
        successful, args.results_dir / "metric_comparison.png"
    )
    plot_confusion(
        best_overall["confusion_matrix_TN_FP_FN_TP"],
        f"Confusion matrix — best primary: {best_overall['name']}",
        args.results_dir / "best_confusion_matrix.png",
    )
    plot_support_vectors(
        successful, args.results_dir / "support_vector_counts.png"
    )
    plot_recall_vs_time(
        successful, args.results_dir / "recall_vs_training_time.png"
    )

    # Persist best primary model dual coefficients (no retrain)
    np.savez_compressed(
        args.results_dir / "best_kernel_model.npz",
        alpha=best_overall["alpha"],
        y_pm1=best_overall["y_pm1"],
        bias=np.array(best_overall["intercept"]),
        kernel_name=np.array(best_overall["kernel"]),
        kernel_params_json=np.array(json.dumps(best_overall["kernel_params"])),
        C=np.array(best_overall["C"]),
        raw_feature_names=np.array(names),
        numeric_means=prep.means,
        numeric_stds=prep.stds,
        numeric_positions=np.array(prep.numeric_positions),
        X_train=X_train,  # needed for kernel prediction
    )

    print("\n===== SUMMARY =====")
    print(json.dumps(summary["best_primary"], indent=2))
    print(f"\nWrote results to {args.results_dir}")


if __name__ == "__main__":
    main()

"""Run kernel SVM experiments on the student placement dataset.

Uses train_processed.csv / test_processed.csv (already standardized and
one-hot encoded; labels -1/+1). Solves the soft-margin dual with a
precomputed Gram matrix, as in run_epilepsy_experiments.py.

The training set has 45,000 rows; a dense 45,000^2 Gram matrix is ~16 GB, so
by default a class-stratified subsample (--train-rows, default 5000) is used.

    python run_placement_experiments.py --data-dir <dir with the CSVs>
    (or: python -m <this_folder_name>.run_placement_experiments ...)
    python run_placement_experiments.py --train-rows 1500   # quick demo
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

if __package__:  # run as: python -m <folder>.run_placement_experiments
    from .kernels import get_kernel
    from .placement_kernel_svm import (
        measure, predict_labels, read_processed_csv, solve_dual_kernel,
        stratified_subsample)
else:  # run directly: python run_placement_experiments.py
    from kernels import get_kernel
    from placement_kernel_svm import (
        measure, predict_labels, read_processed_csv, solve_dual_kernel,
        stratified_subsample)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PRIMARY_CONFIGS = [
    {"name": "linear", "kernel": "linear", "params": {}, "C": 1.0},
    {"name": "poly_d2", "kernel": "polynomial",
     "params": {"degree": 2, "gamma": None, "coef0": 1.0}, "C": 1.0},
    {"name": "poly_d3", "kernel": "polynomial",
     "params": {"degree": 3, "gamma": None, "coef0": 1.0}, "C": 1.0},
    {"name": "rbf_g0.01", "kernel": "rbf", "params": {"gamma": 0.01}, "C": 1.0},
    {"name": "rbf_g0.1", "kernel": "rbf", "params": {"gamma": 0.1}, "C": 1.0},
    {"name": "rbf_g1.0", "kernel": "rbf", "params": {"gamma": 1.0}, "C": 1.0},
]


def _resolve_params(params: dict, n_features: int) -> dict:
    out = dict(params)
    if "gamma" in out and out["gamma"] is None:
        out["gamma"] = 1.0 / n_features
    return out


def run_one(cfg, X_train, y_train, X_test, y_test, tolerance) -> dict:
    resolved = _resolve_params(cfg["params"], X_train.shape[1])
    kernel_fn = get_kernel(cfg["kernel"], **resolved)

    t0 = time.perf_counter()
    K_train = kernel_fn(X_train, X_train)
    t_kernel = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = solve_dual_kernel(K_train, y_train, C=cfg["C"], tolerance=tolerance)
    t_opt = time.perf_counter() - t1
    del K_train

    y_pred = predict_labels(kernel_fn(X_test, X_train), model)
    m = measure(y_test, y_pred)
    return {
        "name": cfg["name"], "kernel": cfg["kernel"], "kernel_params": resolved,
        "C": cfg["C"], "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)), "features": int(X_train.shape[1]),
        **m,
        "training_time_sec": t_kernel + t_opt,
        "support_vectors": model["support_vectors"],
        "margin_support_vectors": model["margin_support_vectors"],
        "converged": model["converged"], "iterations": model["iterations"],
        "kkt_gap": model["kkt_gap"],
        "equality_residual": model["equality_residual"],
        "dual_objective_max": model["dual_objective_max"],
        "intercept": model["b"],
        "_alpha": model["alpha"], "_y": model["y"],
    }


def _key(r):
    return (r["f1_pos"] if r["f1_pos"] is not None else -1,
            r["recall_pos"] if r["recall_pos"] is not None else -1,
            r["accuracy"])


def followup_configs(best: dict) -> list[dict]:
    """Small C / gamma sweep around the best nonlinear kernel."""
    out = []
    if best["kernel"] == "rbf":
        g = best["kernel_params"]["gamma"]
        for C in (0.1, 10.0):
            out.append({"name": f"rbf_g{g:g}_C{C:g}", "kernel": "rbf",
                        "params": {"gamma": g}, "C": C})
        for g2 in (g / 3.0, g * 3.0):
            out.append({"name": f"rbf_g{g2:.3g}_C1", "kernel": "rbf",
                        "params": {"gamma": g2}, "C": 1.0})
    elif best["kernel"] == "polynomial":
        p = best["kernel_params"]
        for C in (0.1, 10.0):
            out.append({"name": f"poly_d{p['degree']}_C{C:g}",
                        "kernel": "polynomial", "params": dict(p), "C": C})
    return out


def save_csv(results, path: Path) -> None:
    fields = ["name", "kernel", "kernel_params", "C", "accuracy",
              "precision_pos", "recall_pos", "f1_pos", "specificity_neg",
              "confusion_matrix_TN_FP_FN_TP", "training_time_sec",
              "support_vectors", "margin_support_vectors", "converged",
              "iterations", "kkt_gap", "equality_residual",
              "dual_objective_max", "intercept"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            row["kernel_params"] = json.dumps(r["kernel_params"])
            row["confusion_matrix_TN_FP_FN_TP"] = json.dumps(
                r["confusion_matrix_TN_FP_FN_TP"])
            w.writerow(row)


def plot_metrics(results, path: Path) -> None:
    names = [r["name"] for r in results]
    keys = [("accuracy", "Accuracy"), ("precision_pos", "Precision (placed)"),
            ("recall_pos", "Recall (placed)"), ("f1_pos", "F1 (placed)")]
    x, w = np.arange(len(names)), 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (k, lab) in enumerate(keys):
        ax.bar(x + (i - 1.5) * w, [r[k] or 0.0 for r in results], w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Kernel SVM metrics on student placement holdout")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, title, path: Path) -> None:
    tn, fp, fn, tp = cm
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred -1", "Pred +1"])
    ax.set_yticklabels(["True -1", "True +1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_svs(results, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([r["name"] for r in results], [r["support_vectors"] for r in results],
           color="#4C72B0")
    ax.set_ylabel("Number of support vectors")
    ax.set_title("Support vector count by kernel configuration")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_recall_vs_time(results, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        ax.scatter(r["training_time_sec"], r["recall_pos"] or 0.0, s=60)
        ax.annotate(r["name"], (r["training_time_sec"], r["recall_pos"] or 0.0),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("Recall (placed = +1)")
    ax.set_title("Placement recall vs training time")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def class_dist(y) -> dict:
    n1 = int(np.sum(y == 1))
    return {"n": int(len(y)), "class_neg1": int(len(y) - n1),
            "class_pos1": n1, "positive_rate": n1 / len(y)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Kernel SVM on student placement data")
    ap.add_argument("--data-dir", type=Path, default=Path.cwd(),
                    help="Directory with train_processed.csv / test_processed.csv")
    ap.add_argument("--train-rows", type=int, default=5000,
                    help="Stratified training subsample size (default 5000; "
                         "dense Gram memory grows as n^2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.003)
    ap.add_argument("--skip-followup", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, names = read_processed_csv(args.data_dir / "train_processed.csv")
    X_test, y_test, tnames = read_processed_csv(args.data_dir / "test_processed.csv")
    if names != tnames:
        raise ValueError("Training and test feature columns differ")
    if args.train_rows < 2:
        ap.error("--train-rows must be >= 2")
    idx = stratified_subsample(y_all, args.train_rows, args.seed)
    X_train, y_train = X_all[idx], y_all[idx]

    dist = {"train_full": class_dist(y_all), "train_used": class_dist(y_train),
            "test": class_dist(y_test)}
    print("Class distribution:", json.dumps(dist, indent=2))
    print(f"Features: {X_train.shape[1]}; training rows used: {len(y_train)}")

    def run_all(configs, tag):
        res = []
        for cfg in configs:
            print(f"\n=== {tag} {cfg['name']} ===")
            try:
                r = run_one(cfg, X_train, y_train, X_test, y_test, args.tolerance)
            except Exception as exc:
                print(f"FAILED: {exc}")
                res.append({"name": cfg["name"], "kernel": cfg["kernel"],
                            "kernel_params": cfg["params"], "C": cfg["C"],
                            "error": str(exc), "failed": True})
                continue
            res.append(r)
            print(f"  acc={r['accuracy']:.4f} prec={r['precision_pos']} "
                  f"rec={r['recall_pos']} f1={r['f1_pos']} "
                  f"SVs={r['support_vectors']} converged={r['converged']} "
                  f"iters={r['iterations']} time={r['training_time_sec']:.1f}s")
            if not r["converged"]:
                print("  WARNING: did not meet KKT tolerance.")
        return res

    primary = run_all(PRIMARY_CONFIGS, "")
    ok = [r for r in primary if not r.get("failed")]
    if not ok:
        raise SystemExit("All primary experiments failed.")
    nonlinear = [r for r in ok if r["kernel"] != "linear"]
    best_nl = max(nonlinear, key=_key) if nonlinear else None

    follow = []
    if best_nl and not args.skip_followup:
        print(f"\nBest nonlinear (primary): {best_nl['name']}")
        follow = run_all(followup_configs(best_nl), "follow-up")
    all_ok = ok + [r for r in follow if not r.get("failed")]
    best = max(ok, key=_key)

    drop = {"_alpha", "_y"}
    archive = lambda rs: [{k: v for k, v in r.items() if k not in drop} for r in rs]
    summary = {
        "description": "Kernel soft-margin dual SVM on the student placement data.",
        "class_distribution": dist,
        "train_subsample": {"rows": int(len(y_train)), "seed": args.seed,
                            "method": "class-stratified random subsample"},
        "label_encoding": "CSV labels are -1/+1 and are used directly in the dual",
        "primary_results": archive(primary),
        "followup_results": archive(follow),
        "best_primary": archive([best])[0],
        "best_nonlinear_primary": archive([best_nl])[0] if best_nl else None,
    }
    (args.results_dir / "placement_kernel_results.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    save_csv(all_ok, args.results_dir / "placement_kernel_results.csv")
    plot_metrics(ok, args.results_dir / "metric_comparison.png")
    plot_confusion(best["confusion_matrix_TN_FP_FN_TP"],
                   f"Confusion matrix - best primary: {best['name']}",
                   args.results_dir / "best_confusion_matrix.png")
    plot_svs(ok, args.results_dir / "support_vector_counts.png")
    plot_recall_vs_time(ok, args.results_dir / "recall_vs_training_time.png")
    np.savez_compressed(
        args.results_dir / "best_kernel_model.npz",
        alpha=best["_alpha"], y_pm1=best["_y"], bias=np.array(best["intercept"]),
        kernel_name=np.array(best["kernel"]),
        kernel_params_json=np.array(json.dumps(best["kernel_params"])),
        C=np.array(best["C"]), train_idx=idx)

    print("\n=== Summary (primary) ===")
    for r in ok:
        print(f"{r['name']:<10} acc={r['accuracy']:.4f} "
              f"rec={r['recall_pos']} f1={r['f1_pos']}")
    print(f"\nBest primary: {best['name']}\nResults in {args.results_dir}")


if __name__ == "__main__":
    main()

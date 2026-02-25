"""
SE Validation Plotting: SE score vs. teacher accuracy + AUROC.

Reads the inference results and generates:
  1. Scatter/binned plot: SE score vs. teacher accuracy (binned by SE quantile)
  2. ROC curve with AUROC (SE as predictor of teacher error)
  3. Summary statistics printed to stdout

No GPU needed — run after se_validation_infer.py finishes.

Usage:
  python se_validation_plot.py --input se_validation_results.jsonl --output_dir se_validation_plots
"""

import argparse
import json
import os
import numpy as np


def load_results(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("se") is not None:
                    data.append(d)
    return data


def compute_auroc(labels, scores):
    """Compute AUROC manually (no sklearn dependency).

    labels: binary array (1 = positive = teacher error)
    scores: continuous array (SE score, higher = more likely error)
    """
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return float("nan")

    tpr_list = [0.0]
    fpr_list = [0.0]

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    # Trapezoidal integration
    auroc = 0.0
    for i in range(1, len(fpr_list)):
        auroc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
    return auroc, np.array(fpr_list), np.array(tpr_list)


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Load data ----
    print(f"Loading {args.input}...")
    data = load_results(args.input)
    print(f"Loaded {len(data)} samples (with valid SE)")

    se_scores = np.array([d["se"] for d in data])
    correct = np.array([1 if d["correct"] else 0 for d in data])
    incorrect = 1 - correct

    overall_acc = correct.mean() * 100
    print(f"Overall teacher accuracy: {overall_acc:.2f}%")
    print(f"SE range: [{se_scores.min():.4f}, {se_scores.max():.4f}]")
    print(f"SE mean: {se_scores.mean():.4f}, median: {np.median(se_scores):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # Plot 1: Binned SE vs. Teacher Accuracy
    # ================================================================
    n_bins = args.n_bins

    # Use quantile-based bins so each bin has ~equal samples
    # But also include a fixed-threshold view
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []
    bin_errors = []  # standard error

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (se_scores >= lo) & (se_scores <= hi)
        else:
            mask = (se_scores >= lo) & (se_scores < hi)
        count = mask.sum()
        if count > 0:
            acc = correct[mask].mean() * 100
            se_err = np.sqrt(acc / 100 * (1 - acc / 100) / count) * 100
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(acc)
            bin_counts.append(count)
            bin_errors.append(se_err)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_acc = "#2196F3"
    color_count = "#BBDEFB"

    # Bar chart for sample count (background)
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins * 0.8,
            alpha=0.3, color=color_count, label="Sample count", zorder=1)
    ax2.set_ylabel("Sample Count", color="gray", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="gray")

    # Line + error bar for accuracy (foreground)
    ax1.errorbar(bin_centers, bin_accs, yerr=bin_errors, fmt="o-",
                 color=color_acc, linewidth=2, markersize=8, capsize=4,
                 label="Teacher accuracy", zorder=5)
    ax1.set_xlabel("SE Score (Spectral Entropy)", fontsize=13)
    ax1.set_ylabel("Teacher Accuracy (%)", color=color_acc, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 105)

    # Annotate each bin
    for x, y, n in zip(bin_centers, bin_accs, bin_counts):
        ax1.annotate(f"{y:.0f}%\n(n={n})", xy=(x, y), fontsize=7,
                     ha="center", va="bottom", color=color_acc)

    ax1.set_title("SE Score vs. Teacher Accuracy (Qwen3-8B)", fontsize=14, fontweight="bold")
    ax1.axhline(y=overall_acc, color="red", linestyle="--", alpha=0.5, label=f"Overall acc={overall_acc:.1f}%")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(args.output_dir, "se_vs_accuracy.png")
    fig.savefig(path1, dpi=200)
    print(f"Saved: {path1}")
    plt.close(fig)

    # ================================================================
    # Plot 2: ROC Curve (SE as predictor of teacher error)
    # ================================================================
    auroc, fpr, tpr = compute_auroc(incorrect.tolist(), se_scores.tolist())
    print(f"\nAUROC (SE predicts teacher error): {auroc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#E91E63", linewidth=2, label=f"SE → Error (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUROC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#E91E63")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve: SE Score as Predictor of Teacher Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(args.output_dir, "roc_curve.png")
    fig.savefig(path2, dpi=200)
    print(f"Saved: {path2}")
    plt.close(fig)

    # ================================================================
    # Plot 3: Scatter plot (SE vs correct, jittered)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(correct))
    colors = np.where(correct == 1, "#4CAF50", "#F44336")
    ax.scatter(se_scores, correct + jitter, c=colors, alpha=0.15, s=8, edgecolors="none")
    ax.set_xlabel("SE Score", fontsize=13)
    ax.set_ylabel("Correct (1) / Incorrect (0)", fontsize=13)
    ax.set_title("SE Score vs. Teacher Correctness (scatter)", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"])

    # Add density info
    se_correct = se_scores[correct == 1]
    se_wrong = se_scores[correct == 0]
    ax.axvline(x=np.median(se_correct), color="#4CAF50", linestyle="--", alpha=0.7,
               label=f"Correct median SE={np.median(se_correct):.3f}")
    ax.axvline(x=np.median(se_wrong), color="#F44336", linestyle="--", alpha=0.7,
               label=f"Wrong median SE={np.median(se_wrong):.3f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(args.output_dir, "scatter_correctness.png")
    fig.savefig(path3, dpi=200)
    print(f"Saved: {path3}")
    plt.close(fig)

    # ================================================================
    # Plot 4: SE distribution for correct vs incorrect (histogram)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(se_correct, bins=bins, alpha=0.6, color="#4CAF50", label=f"Correct (n={len(se_correct)})", density=True)
    ax.hist(se_wrong, bins=bins, alpha=0.6, color="#F44336", label=f"Incorrect (n={len(se_wrong)})", density=True)
    ax.set_xlabel("SE Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("SE Distribution: Correct vs. Incorrect Samples", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path4 = os.path.join(args.output_dir, "se_distribution.png")
    fig.savefig(path4, dpi=200)
    print(f"Saved: {path4}")
    plt.close(fig)

    # ================================================================
    # Summary table
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples:         {len(data)}")
    print(f"Overall accuracy:      {overall_acc:.2f}%")
    print(f"AUROC (SE→error):      {auroc:.4f}")
    print(f"Correct   → median SE: {np.median(se_correct):.4f}, mean SE: {np.mean(se_correct):.4f}")
    print(f"Incorrect → median SE: {np.median(se_wrong):.4f}, mean SE: {np.mean(se_wrong):.4f}")
    print()

    # Accuracy at different SE thresholds
    print("Accuracy by SE threshold:")
    print(f"  {'SE range':<20} {'Count':>8} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    thresholds = [(0, 0.01), (0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    for lo, hi in thresholds:
        mask = (se_scores >= lo) & (se_scores <= hi)
        cnt = mask.sum()
        if cnt > 0:
            acc = correct[mask].mean() * 100
            print(f"  [{lo:.2f}, {hi:.2f}]       {cnt:>8} {acc:>9.1f}%")

    print("=" * 60)

    # Save summary as JSON
    summary = {
        "total_samples": len(data),
        "overall_accuracy": round(overall_acc, 4),
        "auroc": round(auroc, 4),
        "correct_median_se": round(float(np.median(se_correct)), 4),
        "correct_mean_se": round(float(np.mean(se_correct)), 4),
        "incorrect_median_se": round(float(np.median(se_wrong)), 4),
        "incorrect_mean_se": round(float(np.mean(se_wrong)), 4),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SE vs teacher accuracy + AUROC")
    parser.add_argument("--input", type=str, default="se_validation_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="se_validation_plots")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for SE vs accuracy plot")
    args = parser.parse_args()
    main(args)

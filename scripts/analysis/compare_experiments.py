"""
compare_experiments.py

Loads pre-computed experiment_results/ from multiple experiment directories
and generates side-by-side comparison plots.

Run analyze_experiment.py on each experiment first, then run this.

Usage:
    python scripts/analysis/compare_experiments.py <exp_dir1> <exp_dir2> ... [OPTIONS]

    <exp_dir1> ...   One or more experiment directories that have already been
                     processed by analyze_experiment.py (each must contain
                     an experiment_results/ subfolder)
    --save_dir <dir> Where to write comparison PNGs (default: ./comparison_plots)
    --show           Also display figures interactively

Example:
    python scripts/analysis/compare_experiments.py \\
        test_output/test_clean \\
        test_output/test_onedropout \\
        test_output/test_threedropout

Plots generated:
    01_mse_vs_time_comparison.png
    02_mse_vs_evals_comparison.png
    03_throughput_comparison.png
    04_final_mse_comparison.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour cycle — extended automatically if more than 8 experiments are compared
COLORS = [
    "#2196F3", "#FF9800", "#F44336",
    "#9C27B0", "#4CAF50", "#00BCD4",
    "#FF5722", "#607D8B",
]

# ── Utility ───────────────────────────────────────────────────────────────────

def save_fig(fig, save_dir, filename, show):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)


def load_experiment_results(exp_dir):
    """
    Load pre-computed CSVs from <exp_dir>/experiment_results/.
    Returns dict with keys: label, n_trials, mse_time, mse_evals, throughput, summary
    or None if results are missing.
    """
    results_dir = os.path.join(exp_dir, "experiment_results")
    if not os.path.isdir(results_dir):
        print(f"  [warn] no experiment_results/ in {exp_dir} — run analyze_experiment.py first")
        return None

    def read(name):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            return df
        return None

    summary = read("summary.csv")
    label   = (summary["label"].iloc[0]
               if summary is not None and "label" in summary.columns
               else os.path.basename(exp_dir))
    n       = (int(summary["n_trials"].iloc[0])
               if summary is not None else 0)

    return {
        "label":      label,
        "n_trials":   n,
        "mse_time":   read("mse_vs_time.csv"),
        "mse_evals":  read("mse_vs_evals.csv"),
        "throughput": read("throughput.csv"),
        "summary":    summary,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_mse_vs_time(experiments, save_dir, show):
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, exp in enumerate(experiments):
        df = exp["mse_time"]
        if df is None:
            continue
        color = COLORS[i % len(COLORS)]
        n     = int(df["n"].iloc[0]) if "n" in df.columns else exp["n_trials"]
        ax.plot(df["time_s"], df["mean_mse"], color=color, linewidth=2,
                label=f"{exp['label']}  (n={n})")
        ax.fill_between(df["time_s"],
                        df["mean_mse"] - df["std_mse"],
                        df["mean_mse"] + df["std_mse"],
                        color=color, alpha=0.18)

    ax.set_xlabel("Wall-Clock Time (s)", fontsize=13)
    ax.set_ylabel("Global Best MSE", fontsize=13)
    ax.set_title("Global Best MSE vs Wall-Clock Time\n(mean ± 1 std across trials)",
                 fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "01_mse_vs_time_comparison.png", show)


def plot_mse_vs_evals(experiments, save_dir, show):
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, exp in enumerate(experiments):
        df = exp["mse_evals"]
        if df is None:
            continue
        color = COLORS[i % len(COLORS)]
        n     = int(df["n"].iloc[0]) if "n" in df.columns else exp["n_trials"]
        ax.plot(df["total_evals"], df["mean_mse"], color=color, linewidth=2,
                label=f"{exp['label']}  (n={n})")
        ax.fill_between(df["total_evals"],
                        df["mean_mse"] - df["std_mse"],
                        df["mean_mse"] + df["std_mse"],
                        color=color, alpha=0.18)

    ax.set_xlabel("Total Genomes Evaluated (all peers combined)", fontsize=13)
    ax.set_ylabel("Global Best MSE", fontsize=13)
    ax.set_title("Global Best MSE vs Total Evaluations\n(mean ± 1 std across trials)",
                 fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "02_mse_vs_evals_comparison.png", show)


def plot_throughput(experiments, save_dir, show):
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, exp in enumerate(experiments):
        df = exp["throughput"]
        if df is None:
            continue
        color = COLORS[i % len(COLORS)]
        n     = int(df["n"].iloc[0]) if "n" in df.columns else exp["n_trials"]
        ax.plot(df["time_s"], df["mean_gps"], color=color, linewidth=2,
                label=f"{exp['label']}  (n={n})")
        ax.fill_between(df["time_s"],
                        df["mean_gps"] - df["std_gps"],
                        df["mean_gps"] + df["std_gps"],
                        color=color, alpha=0.18)

    ax.set_xlabel("Wall-Clock Time (s)", fontsize=13)
    ax.set_ylabel("Total Genomes / sec  (15s rolling window)", fontsize=13)
    ax.set_title("System-Wide Genome Throughput over Time\n(mean ± 1 std across trials)",
                 fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "03_throughput_comparison.png", show)


def plot_final_mse(experiments, save_dir, show):
    labels, means, stds, ns = [], [], [], []

    for exp in experiments:
        s = exp["summary"]
        if s is None:
            continue
        labels.append(exp["label"])
        means.append(float(s["mean_final_mse"].iloc[0]))
        stds.append(float(s["std_final_mse"].iloc[0]))
        ns.append(int(s["n_trials"].iloc[0]))

        # Parse individual trial MSEs for scatter overlay
        exp["_mse_vals"] = []
        if "individual_mses" in s.columns:
            raw = s["individual_mses"].iloc[0]
            if pd.notna(raw) and str(raw).strip():
                exp["_mse_vals"] = [float(v) for v in str(raw).split(";") if v]

    if not labels:
        print("  [skip] no summary data for final MSE plot")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 6))
    x = np.arange(len(labels))

    for i, (exp, mean, std, n) in enumerate(zip(experiments, means, stds, ns)):
        color = COLORS[i % len(COLORS)]
        ax.bar(x[i], mean, yerr=std, capsize=7, color=color, alpha=0.82,
               error_kw={"elinewidth": 2, "ecolor": "black"})
        # Individual trial dots
        vals = exp.get("_mse_vals", [])
        if vals:
            ax.scatter([x[i]] * len(vals), vals, color="black",
                       zorder=5, s=25, alpha=0.55)
        ax.text(x[i], mean + std + (mean * 0.02),
                f"{mean:.5f}\n±{std:.5f}\n(n={n})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final Global Best MSE", fontsize=13)
    ax.set_title("Final Global Best MSE Comparison\n(mean ± 1 std, dots = individual trials)",
                 fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "04_final_mse_comparison.png", show)

    # Print table
    print("\n── Final MSE Comparison ──────────────────────────────────────────────")
    print(f"{'Experiment':<35} {'Mean MSE':>12} {'Std MSE':>12} {'N':>4}")
    print("─" * 67)
    for label, mean, std, n in zip(labels, means, stds, ns):
        print(f"{label:<35} {mean:>12.6f} {std:>12.6f} {n:>4}")
    print("─" * 67)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple analyzed experiments side by side.")
    parser.add_argument("experiment_dirs", nargs="+",
                        help="Experiment directories (each must have experiment_results/)")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory for PNGs (default: ./comparison_plots)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively after saving")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.getcwd(), "comparison_plots")

    print(f"\n=== Loading experiment results")
    experiments = []
    for d in args.experiment_dirs:
        d = os.path.abspath(d)
        print(f"  {d}")
        result = load_experiment_results(d)
        if result is not None:
            experiments.append(result)

    if not experiments:
        print("[error] No valid experiment results found. Run analyze_experiment.py first.")
        sys.exit(1)

    print(f"\n=== Generating comparison plots -> {save_dir}")
    plot_mse_vs_time(experiments, save_dir, args.show)
    plot_mse_vs_evals(experiments, save_dir, args.show)
    plot_throughput(experiments, save_dir, args.show)
    plot_final_mse(experiments, save_dir, args.show)

    print("\n=== Done.")


if __name__ == "__main__":
    main()

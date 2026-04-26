"""
aggregate_results.py

Reads all 9 trials per experiment from the run_experiments.sh output layout,
computes mean +/- std, and generates comparison plots across all 5 experiments.

Usage:
    python aggregate_results.py <experiments_dir> [--save_dir <dir>] [--show]

    <experiments_dir>  Path to test_output/experiments/
    --save_dir         Where to write PNGs (default: <experiments_dir>/plots)
    --show             Also open figures interactively

Expected directory layout:
    <experiments_dir>/
        clean/trial_1/p2p_rank_*/fitness_log.csv ...
        clean/trial_2/...
        dropout_1/trial_1/...
        dropout_3/trial_1/...
        graceful_1/trial_1/...
        graceful_3/trial_1/...

Plots generated:
    01_mse_vs_time.png         — Mean +/- std best MSE vs wall-clock time
    02_mse_vs_evals.png        — Mean +/- std best MSE vs total evaluations
    03_throughput.png          — Mean +/- std throughput per experiment
    04_final_mse_boxplot.png   — Boxplot of final best MSE across trials
    05_time_to_threshold.png   — How long each experiment takes to hit MSE targets
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("clean",      "Clean (no faults)",              "#2196F3"),
    ("dropout_1",  "1 Unintentional Dropout",        "#FF9800"),
    ("dropout_3",  "3 Simultaneous Dropouts",        "#FF5722"),
    ("graceful_1", "1 Graceful Shutdown",            "#9C27B0"),
    ("graceful_3", "3 Graceful Shutdowns",           "#E91E63"),
]

INFINITY_REPLACE = 1e7


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_or_show(fig, save_dir, filename, show):
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


def load_trial_global_best_vs_time(trial_dir):
    """
    For one trial directory, compute the global best MSE curve over wall-clock time.
    Returns a DataFrame with columns [time_s, global_best_mse] or None.
    """
    rank_dirs = sorted(glob.glob(os.path.join(trial_dir, "p2p_rank_*")))
    if not rank_dirs:
        return None

    frames = []
    for d in rank_dirs:
        path = os.path.join(d, "fitness_log.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df["Best Val. MSE"] = df["Best Val. MSE"].replace(INFINITY_REPLACE, np.nan)
            df["time_s"] = df["Time"] / 1000.0
            frames.append(df[["time_s", "Best Val. MSE"]].dropna())
        except Exception:
            continue

    if not frames:
        return None

    combined = (pd.concat(frames)
                  .sort_values("time_s")
                  .reset_index(drop=True))
    combined["global_best_mse"] = combined["Best Val. MSE"].cummin()
    return combined[["time_s", "global_best_mse"]]


def load_trial_global_best_vs_evals(trial_dir):
    """
    For one trial directory, compute the global best MSE curve over total evaluations.
    Returns a DataFrame with columns [total_evals, global_best_mse] or None.
    """
    rank_dirs = sorted(glob.glob(os.path.join(trial_dir, "p2p_rank_*")))
    if not rank_dirs:
        return None

    rank_frames = {}
    for d in rank_dirs:
        path = os.path.join(d, "fitness_log.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df["Best Val. MSE"] = df["Best Val. MSE"].replace(INFINITY_REPLACE, np.nan)
            df["time_s"] = df["Time"] / 1000.0
            df = df.sort_values("time_s").dropna(subset=["Best Val. MSE"])
            rank_frames[d] = df[["time_s", "Inserted Genomes", "Best Val. MSE"]]
        except Exception:
            continue

    if not rank_frames:
        return None

    all_times = sorted(set(
        t for df in rank_frames.values() for t in df["time_s"].tolist()
    ))

    rows = []
    for t in all_times:
        total_evals = 0
        best_mse = float("inf")
        for df in rank_frames.values():
            sub = df[df["time_s"] <= t]
            if sub.empty:
                continue
            latest = sub.iloc[-1]
            total_evals += int(latest["Inserted Genomes"])
            if pd.notna(latest["Best Val. MSE"]):
                best_mse = min(best_mse, float(latest["Best Val. MSE"]))
        if best_mse < float("inf"):
            rows.append({"total_evals": total_evals, "global_best_mse": best_mse})

    if not rows:
        return None

    result = pd.DataFrame(rows).sort_values("total_evals").reset_index(drop=True)
    result["global_best_mse"] = result["global_best_mse"].cummin()
    return result


def load_trial_throughput(trial_dir):
    """
    Returns mean genomes/sec per second across all ranks for this trial.
    """
    rank_dirs = sorted(glob.glob(os.path.join(trial_dir, "p2p_rank_*")))
    all_rates = []
    for d in rank_dirs:
        path = os.path.join(d, "fitness_log.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df["time_s"] = df["Time"] / 1000.0
            df = df.sort_values("time_s")
            dt = df["time_s"].diff()
            dg = df["Inserted Genomes"].diff()
            rate = (dg / dt).replace([np.inf, -np.inf], np.nan).dropna()
            all_rates.extend(rate.tolist())
        except Exception:
            continue
    return np.mean(all_rates) if all_rates else np.nan


def load_trial_final_mse(trial_dir):
    """Returns the global best final MSE for this trial."""
    rank_dirs = sorted(glob.glob(os.path.join(trial_dir, "p2p_rank_*")))
    best = float("inf")
    for d in rank_dirs:
        path = os.path.join(d, "fault_tolerance_summary.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            val = float(df["final_best_fitness"].iloc[0])
            if val < best:
                best = val
        except Exception:
            continue
    return best if best < float("inf") else np.nan


def interpolate_to_grid(curve_df, x_col, y_col, grid):
    """Interpolate a (x, y) curve onto a common grid."""
    if curve_df is None or curve_df.empty:
        return np.full(len(grid), np.nan)
    x = curve_df[x_col].values
    y = curve_df[y_col].values
    return np.interp(grid, x, y, left=np.nan, right=y[-1] if len(y) > 0 else np.nan)


def load_experiment(exp_dir, loader_fn, grid, x_col, y_col):
    """
    Load all trials for one experiment, interpolate onto grid,
    return (mean, std) arrays.
    """
    trial_dirs = sorted(glob.glob(os.path.join(exp_dir, "trial_*")))
    if not trial_dirs:
        return None, None, 0

    curves = []
    for td in trial_dirs:
        curve = loader_fn(td)
        interp = interpolate_to_grid(curve, x_col, y_col, grid)
        curves.append(interp)

    arr = np.array(curves)
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr,  axis=0)
    return mean, std, len(trial_dirs)


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_mse_vs_time(experiments_dir, save_dir, show):
    """Plot 1 — Mean +/- std best MSE vs wall-clock time for all experiments."""
    TIME_GRID = np.linspace(0, 600, 300)

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_key, exp_label, color in EXPERIMENTS:
        exp_dir = os.path.join(experiments_dir, exp_key)
        if not os.path.isdir(exp_dir):
            print(f"  [SKIP] {exp_key} — directory not found")
            continue

        mean, std, n = load_experiment(
            exp_dir,
            load_trial_global_best_vs_time,
            TIME_GRID, "time_s", "global_best_mse"
        )
        if mean is None:
            continue

        valid = ~np.isnan(mean)
        ax.plot(TIME_GRID[valid], mean[valid],
                color=color, linewidth=2, label=f"{exp_label} (n={n})")
        ax.fill_between(TIME_GRID[valid],
                        (mean - std)[valid], (mean + std)[valid],
                        color=color, alpha=0.15)

    ax.set_xlabel("Wall-Clock Time (s)")
    ax.set_ylabel("Global Best MSE")
    ax.set_title("Best Fitness vs. Wall-Clock Time\n"
                 "Mean ± 1 std across 9 trials per experiment",
                 fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(180, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="t=180s (fault time)")
    ax.text(181, ax.get_ylim()[0] * 1.5, "fault\nt=180s",
            fontsize=7, color="gray", va="bottom")

    plt.tight_layout()
    save_or_show(fig, save_dir, "01_mse_vs_time.png", show)


def plot_mse_vs_evals(experiments_dir, save_dir, show):
    """Plot 2 — Mean +/- std best MSE vs total global evaluations."""
    EVAL_GRID = np.linspace(0, 3000, 300)

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_key, exp_label, color in EXPERIMENTS:
        exp_dir = os.path.join(experiments_dir, exp_key)
        if not os.path.isdir(exp_dir):
            continue

        mean, std, n = load_experiment(
            exp_dir,
            load_trial_global_best_vs_evals,
            EVAL_GRID, "total_evals", "global_best_mse"
        )
        if mean is None:
            continue

        valid = ~np.isnan(mean)
        ax.plot(EVAL_GRID[valid], mean[valid],
                color=color, linewidth=2, label=f"{exp_label} (n={n})")
        ax.fill_between(EVAL_GRID[valid],
                        (mean - std)[valid], (mean + std)[valid],
                        color=color, alpha=0.15)

    ax.set_xlabel("Total Global Genome Evaluations (sum across all active ranks)")
    ax.set_ylabel("Global Best MSE")
    ax.set_title("Best Fitness vs. Total Evaluations  ←  Key Algorithmic Efficiency Chart\n"
                 "If curves overlap, fault tolerance does not degrade learning efficiency",
                 fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "02_mse_vs_evals.png", show)


def plot_throughput(experiments_dir, save_dir, show):
    """Plot 3 — Mean +/- std genomes/sec per experiment (bar chart)."""
    labels, means, stds, colors = [], [], [], []

    for exp_key, exp_label, color in EXPERIMENTS:
        exp_dir = os.path.join(experiments_dir, exp_key)
        if not os.path.isdir(exp_dir):
            continue
        trial_dirs = sorted(glob.glob(os.path.join(exp_dir, "trial_*")))
        rates = [load_trial_throughput(td) for td in trial_dirs]
        rates = [r for r in rates if not np.isnan(r)]
        if not rates:
            continue
        labels.append(exp_label.replace(" ", "\n"))
        means.append(np.mean(rates))
        stds.append(np.std(rates))
        colors.append(color)

    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                  error_kw={"elinewidth": 1.5})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean genomes / sec")
    ax.set_title("Average Throughput per Experiment\n"
                 "Mean ± 1 std across all trials (lower = fewer active ranks)",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "03_throughput.png", show)


def plot_final_mse_boxplot(experiments_dir, save_dir, show):
    """Plot 4 — Boxplot of final best MSE across 9 trials per experiment."""
    all_data, labels, colors = [], [], []

    for exp_key, exp_label, color in EXPERIMENTS:
        exp_dir = os.path.join(experiments_dir, exp_key)
        if not os.path.isdir(exp_dir):
            continue
        trial_dirs = sorted(glob.glob(os.path.join(exp_dir, "trial_*")))
        vals = [load_trial_final_mse(td) for td in trial_dirs]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        all_data.append(vals)
        labels.append(exp_label.replace(" ", "\n"))
        colors.append(color)

    if not all_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(all_data, patch_artist=True, tick_labels=labels,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Final Best MSE (lower = better)")
    ax.set_title("Final Model Quality per Experiment\n"
                 "9 trials per condition — overlapping boxes = statistically similar",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "04_final_mse_boxplot.png", show)


def plot_time_to_threshold(experiments_dir, save_dir, show):
    """
    Plot 5 — For each experiment, how many seconds to reach MSE thresholds.
    Shows wall-clock cost of fault tolerance at different quality levels.
    """
    THRESHOLDS = [0.01, 0.005, 0.003, 0.002]

    results = {t: {} for t in THRESHOLDS}

    for exp_key, exp_label, color in EXPERIMENTS:
        exp_dir = os.path.join(experiments_dir, exp_key)
        if not os.path.isdir(exp_dir):
            continue
        trial_dirs = sorted(glob.glob(os.path.join(exp_dir, "trial_*")))

        for thresh in THRESHOLDS:
            times = []
            for td in trial_dirs:
                curve = load_trial_global_best_vs_time(td)
                if curve is None or curve.empty:
                    continue
                hit = curve[curve["global_best_mse"] <= thresh]
                if not hit.empty:
                    times.append(float(hit.iloc[0]["time_s"]))
                # If never reached, don't include (shows as missing bar)
            if times:
                results[thresh][exp_label] = (np.mean(times), np.std(times))

    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(1, n_thresh, figsize=(4 * n_thresh, 5), sharey=False)
    if n_thresh == 1:
        axes = [axes]

    for ax, thresh in zip(axes, THRESHOLDS):
        data = results[thresh]
        if not data:
            ax.set_title(f"MSE ≤ {thresh}\n(not reached)")
            continue

        exp_labels = list(data.keys())
        means = [data[l][0] for l in exp_labels]
        stds  = [data[l][1] for l in exp_labels]
        clrs  = [c for _, el, c in EXPERIMENTS if el in exp_labels]

        x = np.arange(len(exp_labels))
        ax.bar(x, means, yerr=stds, capsize=4,
               color=clrs[:len(exp_labels)], alpha=0.8,
               error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(" ", "\n") for l in exp_labels], fontsize=7)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title(f"Time to reach\nMSE ≤ {thresh}", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Time to Reach MSE Thresholds per Experiment\n"
                 "(missing bar = threshold never reached in 600s)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_or_show(fig, save_dir, "05_time_to_threshold.png", show)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and compare fault-tolerance experiment results")
    parser.add_argument("experiments_dir",
                        help="Path to test_output/experiments/")
    parser.add_argument("--save_dir", default=None,
                        help="Where to write PNGs (default: <experiments_dir>/plots)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    experiments_dir = os.path.abspath(args.experiments_dir)
    save_dir = args.save_dir or os.path.join(experiments_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nReading from : {experiments_dir}")
    print(f"Saving plots : {save_dir}\n")

    # Check which experiments are present
    for exp_key, exp_label, _ in EXPERIMENTS:
        exp_dir = os.path.join(experiments_dir, exp_key)
        trial_dirs = sorted(glob.glob(os.path.join(exp_dir, "trial_*")))
        print(f"  {exp_key:<15} {len(trial_dirs)} trial(s) found")

    print("\nGenerating plots:\n")
    plot_mse_vs_time(experiments_dir, save_dir, args.show)
    plot_mse_vs_evals(experiments_dir, save_dir, args.show)
    plot_throughput(experiments_dir, save_dir, args.show)
    plot_final_mse_boxplot(experiments_dir, save_dir, args.show)
    plot_time_to_threshold(experiments_dir, save_dir, args.show)

    print(f"\nAll done. Plots saved to: {save_dir}\n")


if __name__ == "__main__":
    main()

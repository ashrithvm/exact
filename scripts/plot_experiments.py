"""
plot_experiments.py

Generates comparison plots for the 3 completed experiment types:
  - test_clean        (9 trials, no faults)
  - test_onedropout   (9 trials, 1 peer dropout at t=180s, recovery at t=240s)
  - test_threedropout (9 trials, 3 peer dropouts at t=180s, recovery at t=240s)

Usage:
    python scripts/plot_experiments.py <test_output_dir> [--save_dir <dir>] [--show]

    <test_output_dir>   Path to test_output/ directory
    --save_dir          Where to write PNGs (default: <test_output_dir>/plots)
    --show              Also open figures interactively

Plots generated:
    01_global_best_mse_vs_time.png    — mean +/- std global best MSE vs wall-clock time
    02_global_best_mse_vs_evals.png   — mean +/- std global best MSE vs total evaluations
    03_throughput_over_time.png       — system-wide genomes/sec showing dropout dip
    04_final_mse_summary.png          — bar chart of final MSE mean +/- std per experiment
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

# ── Config ────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("test_clean",        "Clean (no faults)",       "#2196F3"),
    ("test_onedropout",   "1 Peer Dropout",          "#FF9800"),
    ("test_threedropout", "3 Simultaneous Dropouts", "#F44336"),
]

INFINITY_VAL        = 1e7          # EXAMM uses 10000000 as infinity placeholder
TIME_GRID_S         = np.linspace(0, 600, 601)   # 0-600s at 1s resolution
EVAL_GRID           = np.linspace(0, 2000, 401)  # 0-2000 total genomes
THROUGHPUT_WINDOW_S = 15           # rolling window width for genomes/sec

DROPOUT_TIME_S  = 180.0
RECOVERY_TIME_S = 240.0

# ── Utility ───────────────────────────────────────────────────────────────────

def save_fig(fig, save_dir, filename, show):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)


def clean_mse(series):
    """Replace EXAMM's infinity placeholder (10000000) with NaN."""
    return series.replace(INFINITY_VAL, np.nan)


def load_fitness_log(rank_dir):
    """Load and normalise a single rank's fitness_log.csv."""
    path = os.path.join(rank_dir, "fitness_log.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df["time_s"] = df["Time"] / 1000.0          # ms -> seconds
        df["Best Val. MSE"] = clean_mse(df["Best Val. MSE"])
        return df.sort_values("time_s").reset_index(drop=True)
    except Exception as e:
        print(f"    [warn] could not read {path}: {e}")
        return None


def discover_trials(exp_dir, exp_name):
    """Return sorted list of trial directories for one experiment."""
    pattern = os.path.join(exp_dir, f"{exp_name}_*")
    dirs = sorted(glob.glob(pattern))
    return [d for d in dirs if os.path.isdir(d)]


def get_rank_dirs(trial_dir):
    return sorted(glob.glob(os.path.join(trial_dir, "p2p_rank_*")))


# ── Per-trial curve builders ──────────────────────────────────────────────────

def trial_best_mse_vs_time(trial_dir):
    """
    Global best MSE curve over wall-clock time for one trial.
    Merges all ranks' fitness logs, sorts by time, takes running minimum.
    Returns DataFrame [time_s, global_best_mse] or None.
    """
    frames = []
    for rd in get_rank_dirs(trial_dir):
        df = load_fitness_log(rd)
        if df is not None:
            frames.append(df[["time_s", "Best Val. MSE"]].dropna())
    if not frames:
        return None

    combined = (pd.concat(frames)
                  .sort_values("time_s")
                  .reset_index(drop=True))
    combined["global_best_mse"] = combined["Best Val. MSE"].cummin()
    return combined[["time_s", "global_best_mse"]]


def trial_best_mse_vs_evals(trial_dir):
    """
    Global best MSE curve over total evaluations (all peers combined) for one trial.
    At each logged time point: sums Inserted Genomes across all ranks, takes min MSE.
    Returns DataFrame [total_evals, global_best_mse] or None.
    """
    rank_dfs = {}
    for rd in get_rank_dirs(trial_dir):
        df = load_fitness_log(rd)
        if df is not None and "Inserted Genomes" in df.columns:
            sub = df[["time_s", "Inserted Genomes", "Best Val. MSE"]].dropna(subset=["Best Val. MSE"])
            rank_dfs[rd] = sub

    if not rank_dfs:
        return None

    # Collect all unique time points across all ranks
    all_times = sorted({t for df in rank_dfs.values() for t in df["time_s"].tolist()})

    rows = []
    for t in all_times:
        total_evals = 0
        best_mse = float("inf")
        for df in rank_dfs.values():
            sub = df[df["time_s"] <= t]
            if sub.empty:
                continue
            latest = sub.iloc[-1]
            total_evals += int(latest["Inserted Genomes"])
            mse = latest["Best Val. MSE"]
            if pd.notna(mse):
                best_mse = min(best_mse, float(mse))
        if best_mse < float("inf"):
            rows.append({"total_evals": total_evals, "global_best_mse": best_mse})

    if not rows:
        return None

    result = (pd.DataFrame(rows)
                .sort_values("total_evals")
                .reset_index(drop=True))
    result["global_best_mse"] = result["global_best_mse"].cummin()
    return result


def trial_throughput_over_time(trial_dir):
    """
    System-wide genomes/sec over time for one trial.
    Sums cumulative Inserted Genomes across all ranks onto a 1s grid,
    then applies a rolling derivative over THROUGHPUT_WINDOW_S seconds.
    Returns DataFrame [time_s, genomes_per_sec] or None.
    """
    frames = []
    for rd in get_rank_dirs(trial_dir):
        df = load_fitness_log(rd)
        if df is not None and "Inserted Genomes" in df.columns:
            frames.append(df[["time_s", "Inserted Genomes"]])

    if not frames:
        return None

    grid = TIME_GRID_S
    total_genomes = np.zeros(len(grid))

    for df in frames:
        t = df["time_s"].values
        g = df["Inserted Genomes"].values
        interp = np.interp(grid, t, g,
                           left=0.0,
                           right=float(g[-1]))
        total_genomes += interp

    # Rolling derivative
    window = THROUGHPUT_WINDOW_S
    rates = np.full(len(grid), np.nan)
    for i in range(window, len(grid)):
        dt = grid[i] - grid[i - window]
        dg = total_genomes[i] - total_genomes[i - window]
        rates[i] = dg / dt if dt > 0 else np.nan

    return pd.DataFrame({"time_s": grid, "genomes_per_sec": rates})


def trial_final_mse(trial_dir):
    """Global best final MSE for one trial from fault_tolerance_summary.csv."""
    best = float("inf")
    for rd in get_rank_dirs(trial_dir):
        path = os.path.join(rd, "fault_tolerance_summary.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            val = float(df["final_best_fitness"].iloc[0])
            best = min(best, val)
        except Exception:
            continue
    return best if best < float("inf") else np.nan


# ── Interpolation helper ──────────────────────────────────────────────────────

def interpolate_trials(curves, x_col, y_col, grid):
    """
    Interpolate each trial curve onto a common grid.
    Leading NaNs filled with first valid value; trailing extrapolated with last value.
    Returns ndarray of shape (n_valid_trials, len(grid)), or None if no curves.
    """
    out = []
    for df in curves:
        if df is None or df.empty:
            continue
        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)
        y_interp = np.interp(grid, x, y, left=np.nan, right=y[-1])
        # Fill leading NaNs with first valid observed value
        valid = y[~np.isnan(y)]
        if len(valid):
            y_interp[np.isnan(y_interp)] = valid[0]
        out.append(y_interp)
    return np.array(out) if out else None


# ── Load all experiments ──────────────────────────────────────────────────────

def load_experiment(root_dir, exp_name):
    exp_dir = os.path.join(root_dir, exp_name)
    if not os.path.isdir(exp_dir):
        print(f"  [warn] directory not found: {exp_dir}")
        return {"time_curves": [], "eval_curves": [],
                "throughput_curves": [], "final_mses": []}

    trials = discover_trials(exp_dir, exp_name)
    print(f"  {exp_name}: {len(trials)} trial(s) found")

    time_curves, eval_curves, throughput_curves, final_mses = [], [], [], []
    for t_dir in trials:
        time_curves.append(trial_best_mse_vs_time(t_dir))
        eval_curves.append(trial_best_mse_vs_evals(t_dir))
        throughput_curves.append(trial_throughput_over_time(t_dir))
        final_mses.append(trial_final_mse(t_dir))

    return {
        "time_curves":       time_curves,
        "eval_curves":       eval_curves,
        "throughput_curves": throughput_curves,
        "final_mses":        [m for m in final_mses if not np.isnan(m)],
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_mse_vs_time(experiments_data, save_dir, show):
    """Plot 1: Global Best MSE vs Wall-Clock Time with mean +/- std bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for (exp_name, label, color), data in zip(EXPERIMENTS, experiments_data):
        mat = interpolate_trials(
            data["time_curves"], "time_s", "global_best_mse", TIME_GRID_S)
        if mat is None:
            continue
        n    = mat.shape[0]
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat,  axis=0)
        ax.plot(TIME_GRID_S, mean, color=color, linewidth=2,
                label=f"{label}  (n={n})")
        ax.fill_between(TIME_GRID_S, mean - std, mean + std,
                        color=color, alpha=0.18)

    ax.axvline(DROPOUT_TIME_S,  color="black", linestyle="--",
               linewidth=1.2, alpha=0.65, label=f"Dropout  t={int(DROPOUT_TIME_S)}s")
    ax.axvline(RECOVERY_TIME_S, color="gray",  linestyle=":",
               linewidth=1.2, alpha=0.65, label=f"Recovery t={int(RECOVERY_TIME_S)}s")

    ax.set_xlabel("Wall-Clock Time (s)", fontsize=13)
    ax.set_ylabel("Global Best MSE", fontsize=13)
    ax.set_title("Global Best MSE vs Wall-Clock Time\n(mean ± 1 std across trials)",
                 fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "01_global_best_mse_vs_time.png", show)


def plot_mse_vs_evals(experiments_data, save_dir, show):
    """
    Plot 2: Global Best MSE vs Total Evaluations.
    Decouples algorithmic efficiency from wall-clock slowdowns caused by peer loss.
    If curves overlap here, fault only costs time — not solution quality.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for (exp_name, label, color), data in zip(EXPERIMENTS, experiments_data):
        mat = interpolate_trials(
            data["eval_curves"], "total_evals", "global_best_mse", EVAL_GRID)
        if mat is None:
            continue
        n    = mat.shape[0]
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat,  axis=0)
        ax.plot(EVAL_GRID, mean, color=color, linewidth=2,
                label=f"{label}  (n={n})")
        ax.fill_between(EVAL_GRID, mean - std, mean + std,
                        color=color, alpha=0.18)

    ax.set_xlabel("Total Genomes Evaluated (all peers combined)", fontsize=13)
    ax.set_ylabel("Global Best MSE", fontsize=13)
    ax.set_title("Global Best MSE vs Total Evaluations\n(mean ± 1 std across trials)",
                 fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "02_global_best_mse_vs_evals.png", show)


def plot_throughput(experiments_data, save_dir, show):
    """
    Plot 3: System-wide genome throughput over time.
    Shows the dip when peers drop out and recovery ramp-up afterwards.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for (exp_name, label, color), data in zip(EXPERIMENTS, experiments_data):
        valid = [c for c in data["throughput_curves"] if c is not None]
        if not valid:
            continue
        n   = len(valid)
        mat = np.array([c["genomes_per_sec"].values for c in valid])
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat,  axis=0)
        ax.plot(TIME_GRID_S, mean, color=color, linewidth=2,
                label=f"{label}  (n={n})")
        ax.fill_between(TIME_GRID_S, mean - std, mean + std,
                        color=color, alpha=0.18)

    ax.axvline(DROPOUT_TIME_S,  color="black", linestyle="--",
               linewidth=1.2, alpha=0.65, label=f"Dropout  t={int(DROPOUT_TIME_S)}s")
    ax.axvline(RECOVERY_TIME_S, color="gray",  linestyle=":",
               linewidth=1.2, alpha=0.65, label=f"Recovery t={int(RECOVERY_TIME_S)}s")

    ax.set_xlabel("Wall-Clock Time (s)", fontsize=13)
    ax.set_ylabel(f"Total Genomes / sec\n({THROUGHPUT_WINDOW_S}s rolling window)",
                  fontsize=13)
    ax.set_title("System-Wide Genome Throughput over Time\n(mean ± 1 std across trials)",
                 fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "03_throughput_over_time.png", show)


def plot_final_mse_summary(experiments_data, save_dir, show):
    """
    Plot 4: Bar chart of final global best MSE per experiment.
    Error bars show +/- 1 std across trials.
    Also prints a summary table to stdout.
    """
    labels = [label for _, label, _ in EXPERIMENTS]
    colors = [color for _, _, color in EXPERIMENTS]
    means, stds, ns = [], [], []

    for data in experiments_data:
        vals = data["final_mses"]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals)   if vals else np.nan)
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(9, 6))
    x    = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=7,
                  color=colors, alpha=0.85,
                  error_kw={"elinewidth": 2, "ecolor": "black"})

    for bar, mean, std, n in zip(bars, means, stds, ns):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    mean + std + 0.00003,
                    f"{mean:.5f}\n±{std:.5f}\n(n={n})",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final Global Best MSE", fontsize=13)
    ax.set_title("Final Global Best MSE by Experiment\n(mean ± 1 std across trials)",
                 fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "04_final_mse_summary.png", show)

    # Print summary table
    print("\n── Final MSE Summary ─────────────────────────────────────────────────")
    print(f"{'Experiment':<30} {'Mean MSE':>12} {'Std MSE':>12} {'N':>4}")
    print("─" * 62)
    for (_, label, _), mean, std, n in zip(EXPERIMENTS, means, stds, ns):
        mean_str = f"{mean:.6f}" if not np.isnan(mean) else "N/A"
        std_str  = f"{std:.6f}"  if not np.isnan(std)  else "N/A"
        print(f"{label:<30} {mean_str:>12} {std_str:>12} {n:>4}")
    print("─" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot fault-tolerance experiment results.")
    parser.add_argument("test_output_dir",
                        help="Path to test_output/ directory")
    parser.add_argument("--save_dir", default=None,
                        help="Directory for output PNGs (default: test_output/plots)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively after saving")
    args = parser.parse_args()

    root     = args.test_output_dir
    save_dir = args.save_dir or os.path.join(root, "plots")

    print(f"\n=== Loading experiments from: {root}")
    experiments_data = []
    for exp_name, label, _ in EXPERIMENTS:
        print(f"\nLoading {label}...")
        experiments_data.append(load_experiment(root, exp_name))

    print(f"\n=== Generating plots -> {save_dir}")
    plot_mse_vs_time(experiments_data, save_dir, args.show)
    plot_mse_vs_evals(experiments_data, save_dir, args.show)
    plot_throughput(experiments_data, save_dir, args.show)
    plot_final_mse_summary(experiments_data, save_dir, args.show)

    print("\n=== Done.")


if __name__ == "__main__":
    main()

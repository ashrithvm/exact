"""
analyze_experiment.py

Analyzes one experiment directory (any number of trials).
Generates per-experiment plots and saves intermediate result data
that compare_experiments.py can later load for cross-experiment comparison.

Usage:
    python scripts/analysis/analyze_experiment.py <experiment_dir> [OPTIONS]

    <experiment_dir>    Path to one experiment folder, e.g. test_output/test_onedropout
    --label <str>       Human-readable label for plots (default: folder name)
    --dropout_t <float> Wall-clock time when dropout fired, for plot annotation (default: none)
    --recovery_t <float>Wall-clock time when recovery fired, for plot annotation (default: none)
    --save_dir <dir>    Where to write PNGs (default: <experiment_dir>/plots)
    --show              Also display figures interactively

Output:
    <experiment_dir>/plots/
        01_mse_vs_time.png
        02_mse_vs_evals.png
        03_throughput.png
        04_final_mse.png
    <experiment_dir>/experiment_results/
        mse_vs_time.csv       — time_s, mean_mse, std_mse
        mse_vs_evals.csv      — total_evals, mean_mse, std_mse
        throughput.csv        — time_s, mean_gps, std_gps
        summary.csv           — label, n_trials, mean_final_mse, std_final_mse
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

INFINITY_VAL        = 1e7
TIME_GRID_S         = np.linspace(0, 600, 601)
EVAL_GRID           = np.linspace(0, 2000, 401)
THROUGHPUT_WINDOW_S = 15

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
    return series.replace(INFINITY_VAL, np.nan)


def load_fitness_log(rank_dir):
    path = os.path.join(rank_dir, "fitness_log.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df["time_s"] = df["Time"] / 1000.0
        df["Best Val. MSE"] = clean_mse(df["Best Val. MSE"])
        return df.sort_values("time_s").reset_index(drop=True)
    except Exception as e:
        print(f"    [warn] {path}: {e}")
        return None


def get_rank_dirs(trial_dir):
    return sorted(glob.glob(os.path.join(trial_dir, "p2p_rank_*")))


def discover_trials(experiment_dir):
    """
    Find all valid trial subdirectories inside experiment_dir.
    A directory is a valid trial if it contains at least one p2p_rank_*/fitness_log.csv.
    No naming convention assumed.
    """
    trials = []
    for entry in sorted(os.scandir(experiment_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        # Check that it has at least one rank subdir with a fitness log
        has_data = any(
            os.path.exists(os.path.join(rd, "fitness_log.csv"))
            for rd in glob.glob(os.path.join(entry.path, "p2p_rank_*"))
        )
        if has_data:
            trials.append(entry.path)
    return trials


# ── Per-trial builders ────────────────────────────────────────────────────────

def trial_best_mse_vs_time(trial_dir):
    frames = []
    for rd in get_rank_dirs(trial_dir):
        df = load_fitness_log(rd)
        if df is not None:
            frames.append(df[["time_s", "Best Val. MSE"]].dropna())
    if not frames:
        return None
    combined = pd.concat(frames).sort_values("time_s").reset_index(drop=True)
    combined["global_best_mse"] = combined["Best Val. MSE"].cummin()
    return combined[["time_s", "global_best_mse"]]


def trial_best_mse_vs_evals(trial_dir):
    rank_dfs = {}
    for rd in get_rank_dirs(trial_dir):
        df = load_fitness_log(rd)
        if df is not None and "Inserted Genomes" in df.columns:
            sub = df[["time_s", "Inserted Genomes", "Best Val. MSE"]].dropna(subset=["Best Val. MSE"])
            rank_dfs[rd] = sub
    if not rank_dfs:
        return None

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

    result = pd.DataFrame(rows).sort_values("total_evals").reset_index(drop=True)
    result["global_best_mse"] = result["global_best_mse"].cummin()
    return result


def trial_throughput_over_time(trial_dir):
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
        interp = np.interp(grid, df["time_s"].values, df["Inserted Genomes"].values,
                           left=0.0, right=float(df["Inserted Genomes"].iloc[-1]))
        total_genomes += interp

    window = THROUGHPUT_WINDOW_S
    rates = np.full(len(grid), np.nan)
    for i in range(window, len(grid)):
        dt = grid[i] - grid[i - window]
        dg = total_genomes[i] - total_genomes[i - window]
        rates[i] = dg / dt if dt > 0 else np.nan

    return pd.DataFrame({"time_s": grid, "genomes_per_sec": rates})


def trial_final_mse(trial_dir):
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


# ── Interpolation ─────────────────────────────────────────────────────────────

def interpolate_trials(curves, x_col, y_col, grid):
    out = []
    for df in curves:
        if df is None or df.empty:
            continue
        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)
        y_interp = np.interp(grid, x, y, left=np.nan, right=y[-1])
        valid = y[~np.isnan(y)]
        if len(valid):
            y_interp[np.isnan(y_interp)] = valid[0]
        out.append(y_interp)
    return np.array(out) if out else None


def vline(ax, t, color, style, label):
    if t is not None:
        ax.axvline(t, color=color, linestyle=style, linewidth=1.2,
                   alpha=0.65, label=label)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_mse_vs_time(mat, label, dropout_t, recovery_t, save_dir, show):
    if mat is None:
        print("  [skip] no data for MSE vs time")
        return
    n    = mat.shape[0]
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(TIME_GRID_S, mean, color="#2196F3", linewidth=2,
            label=f"{label}  (n={n})")
    ax.fill_between(TIME_GRID_S, mean - std, mean + std,
                    color="#2196F3", alpha=0.2, label="± 1 std")
    vline(ax, dropout_t,  "black", "--", f"Dropout  t={dropout_t}s")
    vline(ax, recovery_t, "gray",  ":",  f"Recovery t={recovery_t}s")

    ax.set_xlabel("Wall-Clock Time (s)", fontsize=13)
    ax.set_ylabel("Global Best MSE", fontsize=13)
    ax.set_title(f"{label}\nGlobal Best MSE vs Wall-Clock Time  (mean ± 1 std, n={n})",
                 fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "01_mse_vs_time.png", show)


def plot_mse_vs_evals(mat, label, save_dir, show):
    if mat is None:
        print("  [skip] no data for MSE vs evals")
        return
    n    = mat.shape[0]
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(EVAL_GRID, mean, color="#FF9800", linewidth=2,
            label=f"{label}  (n={n})")
    ax.fill_between(EVAL_GRID, mean - std, mean + std,
                    color="#FF9800", alpha=0.2, label="± 1 std")

    ax.set_xlabel("Total Genomes Evaluated (all peers combined)", fontsize=13)
    ax.set_ylabel("Global Best MSE", fontsize=13)
    ax.set_title(f"{label}\nGlobal Best MSE vs Total Evaluations  (mean ± 1 std, n={n})",
                 fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "02_mse_vs_evals.png", show)


def plot_throughput(mat, label, dropout_t, recovery_t, save_dir, show):
    if mat is None:
        print("  [skip] no data for throughput")
        return
    n    = mat.shape[0]
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(TIME_GRID_S, mean, color="#4CAF50", linewidth=2,
            label=f"{label}  (n={n})")
    ax.fill_between(TIME_GRID_S, mean - std, mean + std,
                    color="#4CAF50", alpha=0.2, label="± 1 std")
    vline(ax, dropout_t,  "black", "--", f"Dropout  t={dropout_t}s")
    vline(ax, recovery_t, "gray",  ":",  f"Recovery t={recovery_t}s")

    ax.set_xlabel("Wall-Clock Time (s)", fontsize=13)
    ax.set_ylabel(f"Total Genomes / sec  ({THROUGHPUT_WINDOW_S}s rolling window)",
                  fontsize=13)
    ax.set_title(f"{label}\nSystem-Wide Genome Throughput  (mean ± 1 std, n={n})",
                 fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "03_throughput.png", show)


def plot_final_mse(final_mses, label, save_dir, show):
    if not final_mses:
        print("  [skip] no final MSE data")
        return
    vals = np.array(final_mses)
    mean = np.mean(vals)
    std  = np.std(vals)
    n    = len(vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([label], [mean], yerr=[std], capsize=8, color="#9C27B0", alpha=0.85,
           error_kw={"elinewidth": 2, "ecolor": "black"})
    # Overlay individual trial points
    ax.scatter([label] * n, vals, color="black", zorder=5, s=30, alpha=0.6,
               label="Individual trials")
    ax.text(0, mean + std + (mean * 0.02),
            f"mean={mean:.5f}\nstd={std:.5f}\nn={n}",
            ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Final Global Best MSE", fontsize=13)
    ax.set_title(f"{label}\nFinal Global Best MSE across {n} trial(s)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_dir, "04_final_mse.png", show)


# ── Save intermediate results ─────────────────────────────────────────────────

def save_results(label, n_trials,
                 time_mat, eval_mat, tp_mat, final_mses,
                 results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # MSE vs time
    if time_mat is not None:
        pd.DataFrame({
            "time_s":   TIME_GRID_S,
            "mean_mse": np.nanmean(time_mat, axis=0),
            "std_mse":  np.nanstd(time_mat,  axis=0),
            "n":        time_mat.shape[0],
        }).to_csv(os.path.join(results_dir, "mse_vs_time.csv"), index=False)

    # MSE vs evals
    if eval_mat is not None:
        pd.DataFrame({
            "total_evals": EVAL_GRID,
            "mean_mse":    np.nanmean(eval_mat, axis=0),
            "std_mse":     np.nanstd(eval_mat,  axis=0),
            "n":           eval_mat.shape[0],
        }).to_csv(os.path.join(results_dir, "mse_vs_evals.csv"), index=False)

    # Throughput
    if tp_mat is not None:
        pd.DataFrame({
            "time_s":   TIME_GRID_S,
            "mean_gps": np.nanmean(tp_mat, axis=0),
            "std_gps":  np.nanstd(tp_mat,  axis=0),
            "n":        tp_mat.shape[0],
        }).to_csv(os.path.join(results_dir, "throughput.csv"), index=False)

    # Summary
    mses = [m for m in final_mses if not np.isnan(m)]
    pd.DataFrame([{
        "label":          label,
        "n_trials":       n_trials,
        "mean_final_mse": np.mean(mses) if mses else np.nan,
        "std_final_mse":  np.std(mses)  if mses else np.nan,
        "individual_mses": ";".join(f"{v:.8f}" for v in mses),
    }]).to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    print(f"  results saved -> {results_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze one experiment directory (any number of trials).")
    parser.add_argument("experiment_dir",
                        help="Path to experiment folder (e.g. test_output/test_onedropout)")
    parser.add_argument("--label", default=None,
                        help="Human-readable label (default: folder name)")
    parser.add_argument("--dropout_t",  type=float, default=None,
                        help="Wall-clock time when dropout fired (for annotation)")
    parser.add_argument("--recovery_t", type=float, default=None,
                        help="Wall-clock time when recovery fired (for annotation)")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory for PNGs (default: <experiment_dir>/plots)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively after saving")
    args = parser.parse_args()

    exp_dir  = os.path.abspath(args.experiment_dir)
    label    = args.label or os.path.basename(exp_dir)
    save_dir = args.save_dir or os.path.join(exp_dir, "plots")
    results_dir = os.path.join(exp_dir, "experiment_results")

    print(f"\n=== Analyzing: {exp_dir}")
    print(f"    Label: {label}")

    trials = discover_trials(exp_dir)
    print(f"    Trials found: {len(trials)}")
    for t in trials:
        print(f"      {os.path.basename(t)}")

    if not trials:
        print("  [error] No valid trial directories found. Exiting.")
        sys.exit(1)

    # Build per-trial curves
    time_curves, eval_curves, tp_curves, final_mses = [], [], [], []
    for t_dir in trials:
        time_curves.append(trial_best_mse_vs_time(t_dir))
        eval_curves.append(trial_best_mse_vs_evals(t_dir))
        tp_curves.append(trial_throughput_over_time(t_dir))
        final_mses.append(trial_final_mse(t_dir))

    # Interpolate onto common grids
    time_mat = interpolate_trials(time_curves, "time_s",      "global_best_mse", TIME_GRID_S)
    eval_mat = interpolate_trials(eval_curves, "total_evals", "global_best_mse", EVAL_GRID)
    tp_mat   = (np.array([c["genomes_per_sec"].values for c in tp_curves if c is not None])
                if any(c is not None for c in tp_curves) else None)

    print(f"\n=== Generating plots -> {save_dir}")
    plot_mse_vs_time(time_mat, label, args.dropout_t, args.recovery_t, save_dir, args.show)
    plot_mse_vs_evals(eval_mat, label, save_dir, args.show)
    plot_throughput(tp_mat, label, args.dropout_t, args.recovery_t, save_dir, args.show)
    plot_final_mse(final_mses, label, save_dir, args.show)

    print(f"\n=== Saving intermediate results -> {results_dir}")
    save_results(label, len(trials), time_mat, eval_mat, tp_mat, final_mses, results_dir)

    # Print summary to stdout
    mses = [m for m in final_mses if not np.isnan(m)]
    print(f"\n── Summary ───────────────────────────────────────────────────────────")
    print(f"  Experiment : {label}")
    print(f"  Trials     : {len(trials)}")
    if mses:
        print(f"  Final MSE  : mean={np.mean(mses):.6f}  std={np.std(mses):.6f}  "
              f"min={np.min(mses):.6f}  max={np.max(mses):.6f}")
    else:
        print("  Final MSE  : N/A")
    print("─" * 70)
    print("\n=== Done.")


if __name__ == "__main__":
    main()

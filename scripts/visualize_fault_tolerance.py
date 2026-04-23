"""
visualize_fault_tolerance.py

Reads all output from a decentralized P2P EXAMM fault-tolerance run and
produces a full set of analysis plots + a printed summary table.

Usage:
    python visualize_fault_tolerance.py <output_dir> [--save_dir <dir>] [--show]

    <output_dir>  Path to the run output (e.g. test_output/c172_fault_tolerance)
    --save_dir    Where to write PNG files (default: <output_dir>/plots)
    --show        Also open each figure interactively

Examples:
    python visualize_fault_tolerance.py ../test_output/c172_fault_tolerance
    python visualize_fault_tolerance.py ../test_output/c172_fault_tolerance --show
"""

import sys
import os
import argparse
import glob
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive by default; --show overrides
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── colour palette ────────────────────────────────────────────────────────────
RANK_COLORS  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
                "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]
EVENT_COLORS = {
    "DROPOUT_START":            "#FF5722",
    "HEARTBEAT_TIMEOUT":        "#FF9800",
    "PEER_FAILED_RECEIVED":     "#FFC107",
    "PEER_FAILED_DETECTED":     "#FF9800",
    "REJOIN_REQUESTED":         "#8BC34A",
    "REJOIN_SEEDED":            "#4CAF50",
    "REJOIN_COMPLETE":          "#2196F3",
    "REJOIN_NOTIFY_RECEIVED":   "#03A9F4",
    "GRACEFUL_SHUTDOWN_SENT":   "#9C27B0",
    "GRACEFUL_SHUTDOWN_RECEIVED": "#CE93D8",
    "TOKEN_RECOVERED":          "#F44336",
}
INFINITY_REPLACE = 1e7   # fitness_log uses 10000000 for "no genome yet"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_rank_dirs(output_dir):
    pattern = os.path.join(output_dir, "p2p_rank_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        sys.exit(f"No p2p_rank_* directories found in {output_dir}")
    return dirs


def rank_from_dir(d):
    return int(os.path.basename(d).split("_")[-1])


def read_csv_safe(path, **kwargs):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, **kwargs)
    df.columns = df.columns.str.strip()
    return df


def clean_infinity(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(INFINITY_REPLACE, np.nan)
    return df


def save_or_show(fig, save_dir, filename, show):
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


def add_event_vlines(ax, events_df, rank=None, alpha=0.7, label_y=None):
    """Draw vertical lines for each fault-tolerance event."""
    if events_df is None or events_df.empty:
        return
    plotted = set()
    for _, row in events_df.iterrows():
        etype = row["event_type"]
        color = EVENT_COLORS.get(etype, "#888888")
        ls = "--" if "RECEIVED" in etype or "NOTIFY" in etype else "-"
        lw = 1.2 if "RECEIVED" in etype else 1.8
        ax.axvline(row["wall_time_s"], color=color, linestyle=ls,
                   linewidth=lw, alpha=alpha)
        if etype not in plotted and label_y is not None:
            ax.text(row["wall_time_s"] + 0.5, label_y, etype.replace("_", "\n"),
                    fontsize=5, color=color, va="top", rotation=90, alpha=0.8)
            plotted.add(etype)


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_global_convergence(rank_data, all_events, save_dir, show):
    """
    Plot 1 — Global best MSE across all ranks over wall time,
    with fault-event markers.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    global_best = None
    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None:
            continue
        fl = clean_infinity(fl.copy(), ["Best Val. MSE"])
        fl["time_s"] = fl["Time"] / 1000.0
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.plot(fl["time_s"], fl["Best Val. MSE"],
                color=color, linewidth=1.2, alpha=0.6,
                label=f"Rank {rank}")
        # track global best
        if global_best is None:
            global_best = fl[["time_s", "Best Val. MSE"]].copy()
        else:
            combined = pd.concat([global_best, fl[["time_s", "Best Val. MSE"]]])
            combined = combined.sort_values("time_s")
            combined["Best Val. MSE"] = combined["Best Val. MSE"].cummin()
            global_best = combined

    if global_best is not None:
        global_best = global_best.dropna(subset=["Best Val. MSE"])
        ax.plot(global_best["time_s"], global_best["Best Val. MSE"],
                color="black", linewidth=2.5, label="Global best", zorder=5)

    # Fault event lines — use rank 0's events as they see everything
    ref_events = all_events.get(0) if 0 in all_events else next(iter(all_events.values()), None)
    if ref_events is not None:
        ymax = ax.get_ylim()[1]
        add_event_vlines(ax, ref_events, label_y=ymax)

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Best Validation MSE")
    ax.set_title("Global MSE Convergence with Fault Events")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # Custom legend for event lines
    event_handles = [
        Line2D([0], [0], color=c, linewidth=1.8, label=e.replace("_", " "))
        for e, c in EVENT_COLORS.items()
        if any(e in (ev.get("event_type", "") if isinstance(ev, dict) else "")
               for ev_df in all_events.values() if ev_df is not None
               for ev in ev_df.to_dict("records"))
    ]
    if event_handles:
        ax.legend(handles=event_handles, loc="lower left", fontsize=6,
                  title="Fault events", title_fontsize=7)

    save_or_show(fig, save_dir, "01_global_convergence.png", show)


def plot_per_rank_convergence(rank_data, all_events, save_dir, show):
    """Plot 2 — Individual rank MSE curves side by side."""
    ranks = sorted(rank_data.keys())
    n = len(ranks)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    for i, rank in enumerate(ranks):
        ax = axes[i]
        fl = rank_data[rank].get("fitness_log")
        if fl is not None:
            fl = clean_infinity(fl.copy(), ["Best Val. MSE"])
            fl["time_s"] = fl["Time"] / 1000.0
            color = RANK_COLORS[rank % len(RANK_COLORS)]
            ax.plot(fl["time_s"], fl["Best Val. MSE"],
                    color=color, linewidth=1.5)

        events = all_events.get(rank)
        add_event_vlines(ax, events)

        ax.set_title(f"Rank {rank}", fontweight="bold")
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("Best Val. MSE")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Rank MSE Convergence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, save_dir, "02_per_rank_convergence.png", show)


def plot_fault_timeline(rank_data, all_events, save_dir, show):
    """
    Plot 3 — Gantt-style rank state timeline.
    Green = active, red = dropped out, purple = gracefully shut down.
    """
    ranks = sorted(rank_data.keys())

    # Determine total run duration from fitness logs
    max_time = 0.0
    for data in rank_data.values():
        fl = data.get("fitness_log")
        if fl is not None and not fl.empty:
            max_time = max(max_time, fl["Time"].max() / 1000.0)
    if max_time == 0:
        max_time = 1000.0

    fig, ax = plt.subplots(figsize=(14, max(3, len(ranks) * 0.9)))

    state_colors = {
        "active":   "#4CAF50",
        "dropped":  "#FF5722",
        "shutdown": "#9C27B0",
    }

    for y, rank in enumerate(ranks):
        # Default: whole run is active
        intervals = [{"start": 0.0, "end": max_time, "state": "active"}]

        events = all_events.get(rank)
        if events is not None:
            for _, row in events.iterrows():
                t = row["wall_time_s"]
                et = row["event_type"]
                if et == "DROPOUT_START":
                    # Split active interval at dropout
                    intervals = _split_interval(intervals, t, max_time, "dropped")
                elif et == "REJOIN_COMPLETE":
                    intervals = _split_interval(intervals, t, max_time, "active")
                elif et == "GRACEFUL_SHUTDOWN_SENT":
                    intervals = _split_interval(intervals, t, max_time, "shutdown")

        for iv in intervals:
            color = state_colors[iv["state"]]
            ax.barh(y, iv["end"] - iv["start"], left=iv["start"],
                    height=0.6, color=color, alpha=0.85, edgecolor="white")

        # Overlay event markers
        if events is not None:
            for _, row in events.iterrows():
                et = row["event_type"]
                color = EVENT_COLORS.get(et, "#555")
                ax.plot(row["wall_time_s"], y, marker="v", color=color,
                        markersize=7, zorder=5)

    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"Rank {r}" for r in ranks])
    ax.set_xlabel("Wall time (s)")
    ax.set_title("Rank State Timeline", fontweight="bold")
    ax.set_xlim(0, max_time)

    legend_handles = [
        mpatches.Patch(color=c, label=s.capitalize())
        for s, c in state_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "03_fault_timeline.png", show)


def _split_interval(intervals, at_time, max_time, new_state):
    """Helper: split the last interval at at_time and add new_state interval."""
    result = []
    for iv in intervals:
        if iv["end"] > at_time >= iv["start"]:
            result.append({"start": iv["start"], "end": at_time, "state": iv["state"]})
            result.append({"start": at_time,     "end": max_time, "state": new_state})
        else:
            result.append(iv)
    return result


def plot_active_rank_count(all_events, max_time, save_dir, show):
    """Plot 4 — Number of active ranks over time."""
    # Collect all events from all ranks' perspectives (use rank 0 as ground truth)
    ranks = sorted(all_events.keys())
    n_ranks = len(ranks)

    # Build a timeline of active count changes from any event that changes active_ranks_count
    events_list = []
    for rank, ev_df in all_events.items():
        if ev_df is None:
            continue
        for _, row in ev_df.iterrows():
            et = row["event_type"]
            if et in ("HEARTBEAT_TIMEOUT", "PEER_FAILED_RECEIVED",
                       "GRACEFUL_SHUTDOWN_SENT", "GRACEFUL_SHUTDOWN_RECEIVED",
                       "REJOIN_COMPLETE", "REJOIN_SEEDED"):
                events_list.append({
                    "time": row["wall_time_s"],
                    "active_count": row["active_ranks_count"],
                    "event": et,
                    "rank": rank,
                })

    if not events_list:
        return

    ev_df_all = pd.DataFrame(events_list).sort_values("time")

    fig, ax = plt.subplots(figsize=(12, 4))

    times = [0.0] + list(ev_df_all["time"]) + [max_time]
    counts = [n_ranks] + list(ev_df_all["active_count"]) + [ev_df_all["active_count"].iloc[-1]]

    ax.step(times, counts, where="post", color="#2196F3", linewidth=2)
    ax.fill_between(times, counts, step="post", alpha=0.15, color="#2196F3")

    for _, row in ev_df_all.drop_duplicates("time").iterrows():
        color = EVENT_COLORS.get(row["event"], "#888")
        ax.axvline(row["time"], color=color, linestyle="--", alpha=0.6, linewidth=1.2)

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Active ranks")
    ax.set_title("Active Rank Count Over Time", fontweight="bold")
    ax.set_ylim(0, n_ranks + 0.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "04_active_rank_count.png", show)


def plot_island_heatmap(rank_data, save_dir, show):
    """
    Plot 5 — Heatmap of best island fitness at end of run.
    Rows = ranks, cols = islands.
    """
    ranks = sorted(rank_data.keys())
    island_cols = None
    matrices = {}

    for rank in ranks:
        fl = rank_data[rank].get("fitness_log")
        if fl is None or fl.empty:
            continue
        best_cols = [c for c in fl.columns if c.endswith("_best_fitness")]
        if not best_cols:
            continue
        island_cols = best_cols
        last_row = clean_infinity(fl.copy(), best_cols).iloc[-1]
        matrices[rank] = [last_row[c] for c in best_cols]

    if not matrices or island_cols is None:
        return

    n_islands = len(island_cols)
    data = np.array([matrices.get(r, [np.nan] * n_islands) for r in ranks])

    fig, ax = plt.subplots(figsize=(max(8, n_islands), max(3, len(ranks))))
    vmin = np.nanmin(data[data < 1e6])
    vmax = np.nanpercentile(data[data < 1e6], 95) if np.any(data < 1e6) else 1.0

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Best island fitness (MSE)")

    ax.set_xticks(range(n_islands))
    ax.set_xticklabels([f"Island {i}" for i in range(n_islands)], rotation=45, ha="right")
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"Rank {r}" for r in ranks])
    ax.set_title("Island Population Fitness at End of Run\n(green = better)", fontweight="bold")

    # Annotate cells
    for i, rank in enumerate(ranks):
        for j in range(n_islands):
            val = data[i, j]
            if not np.isnan(val) and val < 1e6:
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=6, color="black")

    plt.tight_layout()
    save_or_show(fig, save_dir, "05_island_heatmap.png", show)


def plot_genome_improvement(rank_data, save_dir, show):
    """
    Plot 6 — Scatter: initial fitness vs final fitness per genome.
    Points below the diagonal = backprop improved the genome.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    for rank, data in rank_data.items():
        gs = data.get("genome_stats")
        if gs is None or gs.empty:
            continue
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.scatter(gs["Initial Fitness"], gs["Final Fitness"],
                   color=color, alpha=0.4, s=15, label=f"Rank {rank}")

    lim_max = ax.get_xlim()[1]
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, alpha=0.5, label="No change")
    ax.set_xlabel("Initial Fitness (before backprop)")
    ax.set_ylabel("Final Fitness (after backprop)")
    ax.set_title("Genome Training Improvement\n(below diagonal = backprop helped)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "06_genome_improvement.png", show)


def plot_network_complexity(rank_data, all_events, save_dir, show):
    """
    Plot 7 — Enabled nodes and edges over time, showing architectural growth.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None:
            continue
        fl["time_s"] = fl["Time"] / 1000.0
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax1.plot(fl["time_s"], fl["Enabled Nodes"],
                 color=color, linewidth=1.2, alpha=0.7, label=f"Rank {rank}")
        ax2.plot(fl["time_s"], fl["Enabled Edges"],
                 color=color, linewidth=1.2, alpha=0.7)

    ref_events = all_events.get(0) if 0 in all_events else next(iter(all_events.values()), None)
    add_event_vlines(ax1, ref_events, alpha=0.4)
    add_event_vlines(ax2, ref_events, alpha=0.4)

    ax1.set_ylabel("Enabled nodes")
    ax1.set_title("Network Complexity Over Time", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Enabled edges")
    ax2.set_xlabel("Wall time (s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "07_network_complexity.png", show)


def plot_bp_training_stats(rank_data, save_dir, show):
    """
    Plot 8 — Backprop training time distribution and fitness improvement ratio.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    all_times = []
    all_ratios = []
    all_rank_labels = []

    for rank, data in rank_data.items():
        gs = data.get("genome_stats")
        if gs is None or gs.empty:
            continue
        times = gs["BP Time (ms)"].dropna()
        ratio = (gs["Initial Fitness"] / gs["Final Fitness"].replace(0, np.nan)).dropna()
        all_times.append(times)
        all_ratios.append(ratio)
        all_rank_labels.append(f"Rank {rank}")

    if all_times:
        ax1.boxplot(all_times, labels=all_rank_labels, patch_artist=True,
                    boxprops=dict(facecolor="#BBDEFB"),
                    medianprops=dict(color="#1565C0", linewidth=2))
        ax1.set_ylabel("BP training time (ms)")
        ax1.set_title("Backprop Time per Genome", fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

    if all_ratios:
        ax2.boxplot(all_ratios, labels=all_rank_labels, patch_artist=True,
                    boxprops=dict(facecolor="#C8E6C9"),
                    medianprops=dict(color="#2E7D32", linewidth=2))
        ax2.set_ylabel("Fitness improvement ratio (initial / final)")
        ax2.set_title("Genome Improvement Factor\n(higher = more improvement)", fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_yscale("log")

    plt.tight_layout()
    save_or_show(fig, save_dir, "08_bp_training_stats.png", show)


def print_summary_table(rank_data, all_events):
    """Print a formatted summary table to the terminal."""
    print("\n" + "=" * 80)
    print("  FAULT TOLERANCE RUN SUMMARY")
    print("=" * 80)

    summaries = []
    for rank, data in rank_data.items():
        s = data.get("summary")
        if s is not None and not s.empty:
            row = s.iloc[0].to_dict()
            row["rank"] = rank
            summaries.append(row)

    if summaries:
        print(f"\n{'Rank':<6} {'Final MSE':<12} {'Genomes':<10} {'Dropouts':<10} "
              f"{'Shutdowns':<11} {'Recoveries':<12} {'Downtime(s)':<13} "
              f"{'Peak Failed':<12} {'Wall Time':<10}")
        print("-" * 96)
        for s in summaries:
            print(f"{int(s.get('rank', -1)):<6} "
                  f"{float(s.get('final_best_fitness', 0)):<12.6f} "
                  f"{int(s.get('total_genomes_evaluated', 0)):<10} "
                  f"{int(s.get('total_dropout_events', 0)):<10} "
                  f"{int(s.get('total_shutdown_events', 0)):<11} "
                  f"{int(s.get('total_recovery_events', 0)):<12} "
                  f"{float(s.get('total_downtime_s', 0)):<13.2f} "
                  f"{int(s.get('peak_concurrent_failed', 0)):<12} "
                  f"{float(s.get('total_wall_time_s', 0)):<10.1f}")

    print("\n  Fault Event Log (chronological across all ranks):\n")
    all_ev_rows = []
    for rank, ev_df in all_events.items():
        if ev_df is None:
            continue
        for _, row in ev_df.iterrows():
            all_ev_rows.append({
                "time_s":   row["wall_time_s"],
                "reporter": row["reporting_rank"],
                "event":    row["event_type"],
                "subject":  row["subject_rank"],
                "active":   row["active_ranks_count"],
                "fitness":  row["best_fitness"],
                "downtime": row["downtime_s"],
            })

    if all_ev_rows:
        ev_all = pd.DataFrame(all_ev_rows).sort_values("time_s").drop_duplicates(
            subset=["time_s", "event", "subject"])
        print(f"  {'Time(s)':<10} {'Reporter':<10} {'Event':<35} {'Subject':<9} "
              f"{'Active':<8} {'Best MSE':<12} {'Downtime(s)'}")
        print("  " + "-" * 90)
        for _, r in ev_all.iterrows():
            dt = f"{r['downtime']:.1f}" if r["downtime"] >= 0 else "-"
            print(f"  {r['time_s']:<10.1f} {int(r['reporter']):<10} {r['event']:<35} "
                  f"{int(r['subject']):<9} {int(r['active']):<8} "
                  f"{r['fitness']:<12.6f} {dt}")

    print("\n" + "=" * 80 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Path to the run output directory")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save PNG files (default: output_dir/plots)")
    parser.add_argument("--show", action="store_true",
                        help="Open each figure interactively")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    output_dir = os.path.abspath(args.output_dir)
    save_dir   = args.save_dir or os.path.join(output_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nReading from : {output_dir}")
    print(f"Saving plots : {save_dir}\n")

    rank_dirs = load_rank_dirs(output_dir)
    rank_data   = {}
    all_events  = {}

    for d in rank_dirs:
        rank = rank_from_dir(d)
        rank_data[rank] = {
            "fitness_log":  read_csv_safe(os.path.join(d, "fitness_log.csv")),
            "genome_stats": read_csv_safe(os.path.join(d, "genome_stats_log.csv")),
            "summary":      read_csv_safe(os.path.join(d, "fault_tolerance_summary.csv")),
        }
        all_events[rank] = read_csv_safe(os.path.join(d, "fault_tolerance_events.csv"))
        loaded = [k for k, v in rank_data[rank].items() if v is not None]
        ev_rows = len(all_events[rank]) if all_events[rank] is not None else 0
        print(f"  rank {rank}: loaded {loaded}  |  {ev_rows} fault events")

    # Determine max wall time
    max_time = 0.0
    for data in rank_data.values():
        fl = data.get("fitness_log")
        if fl is not None and not fl.empty:
            max_time = max(max_time, fl["Time"].max() / 1000.0)
    if max_time == 0:
        max_time = 1000.0

    print(f"\nGenerating plots (total run time ≈ {max_time:.0f}s):\n")

    plot_global_convergence(rank_data, all_events, save_dir, args.show)
    plot_per_rank_convergence(rank_data, all_events, save_dir, args.show)
    plot_fault_timeline(rank_data, all_events, save_dir, args.show)
    plot_active_rank_count(all_events, max_time, save_dir, args.show)
    plot_island_heatmap(rank_data, save_dir, args.show)
    plot_genome_improvement(rank_data, save_dir, args.show)
    plot_network_complexity(rank_data, all_events, save_dir, args.show)
    plot_bp_training_stats(rank_data, save_dir, args.show)

    print_summary_table(rank_data, all_events)
    print(f"All done. Plots saved to: {save_dir}\n")


if __name__ == "__main__":
    main()

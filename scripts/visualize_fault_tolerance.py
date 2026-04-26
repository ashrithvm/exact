"""
visualize_fault_tolerance.py

Reads all output from a decentralized P2P EXAMM fault-tolerance run and
produces a full set of analysis plots + a printed summary table.

Usage:
    python visualize_fault_tolerance.py <output_dir> [--save_dir <dir>] [--show]

    <output_dir>  Path to the run output (e.g. test_output/c172_fault_tolerance)
    --save_dir    Where to write PNG files (default: <output_dir>/plots)
    --show        Also open each figure interactively

Plots generated:
    01_global_convergence_time.png      — Best MSE vs wall-clock time (log scale) + fault markers
    02_global_convergence_evals.png     — Best MSE vs total global evaluations (algorithmic efficiency)
    03_fault_timeline.png               — Gantt: green=active, red=dropped, purple=shutdown
    04_active_rank_count.png            — Step plot of active rank count over time
    05_throughput.png                   — Genomes/sec per rank (shows dropout dip + recovery)
    06_rmse_over_time.png               — RMSE over time per rank with fault markers
    07_r2_over_time.png                 — R² over time per rank with fault markers
    08_knowledge_preservation.png       — Fitness before/after each fault event
    09_mae_convergence.png              — Best validation MAE over time per rank

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

# ── colour palette ─────────────────────────────────────────────────────────────
RANK_COLORS  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
                "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]

EVENT_COLORS = {
    "DROPOUT_START":              "#FF5722",
    "HEARTBEAT_TIMEOUT":          "#FF9800",
    "PEER_FAILED_DETECTED":       "#FF9800",
    "PEER_FAILED_RECEIVED":       "#FFC107",
    "REJOIN_REQUESTED":           "#8BC34A",
    "REJOIN_SEEDED":              "#4CAF50",
    "REJOIN_COMPLETE":            "#2196F3",
    "REJOIN_NOTIFY_RECEIVED":     "#03A9F4",
    "GRACEFUL_SHUTDOWN_SENT":     "#9C27B0",
    "GRACEFUL_SHUTDOWN_RECEIVED": "#CE93D8",
    "TOKEN_RECOVERED":            "#F44336",
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
    try:
        df = pd.read_csv(path, **kwargs)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None


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


def add_event_vlines(ax, events_df, alpha=0.6, label_y=None):
    """Draw vertical lines for primary fault events only."""
    if events_df is None or events_df.empty:
        return
    plotted = set()
    for _, row in events_df.iterrows():
        etype = row["event_type"]
        if etype not in ("DROPOUT_START", "HEARTBEAT_TIMEOUT",
                         "REJOIN_COMPLETE", "GRACEFUL_SHUTDOWN_SENT",
                         "GRACEFUL_SHUTDOWN_RECEIVED"):
            continue
        color = EVENT_COLORS.get(etype, "#888888")
        ls = "--" if "RECEIVED" in etype else "-"
        ax.axvline(row["wall_time_s"], color=color, linestyle=ls,
                   linewidth=1.6, alpha=alpha)
        if etype not in plotted and label_y is not None:
            short = (etype.replace("GRACEFUL_", "")
                         .replace("_SENT", "\nSENT")
                         .replace("_RECEIVED", "\nRCV")
                         .replace("_", "\n"))
            ax.text(row["wall_time_s"] + 1, label_y, short,
                    fontsize=5, color=color, va="top", alpha=0.85)
            plotted.add(etype)


def build_reference_events(all_events):
    """
    Merge events from all ranks into a deduplicated timeline.
    For each (event_type, subject_rank) keep the earliest occurrence.
    """
    rows = []
    for _, df in all_events.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append(row.to_dict())
    if not rows:
        return None
    combined = (pd.DataFrame(rows)
                  .sort_values("wall_time_s")
                  .drop_duplicates(subset=["event_type", "subject_rank"], keep="first")
                  .sort_values("wall_time_s")
                  .reset_index(drop=True))
    return combined


def build_global_best_vs_time(rank_data):
    """
    Returns a DataFrame with columns [time_s, global_best_mse] representing
    the running global minimum MSE across all ranks over wall-clock time.
    """
    frames = []
    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None or fl.empty:
            continue
        fl = clean_infinity(fl.copy(), ["Best Val. MSE"])
        fl["time_s"] = fl["Time"] / 1000.0
        frames.append(fl[["time_s", "Best Val. MSE"]].dropna())

    if not frames:
        return None

    combined = (pd.concat(frames)
                  .sort_values("time_s")
                  .reset_index(drop=True))
    combined["global_best_mse"] = combined["Best Val. MSE"].cummin()
    return combined[["time_s", "global_best_mse"]]


def build_global_best_vs_evals(rank_data):
    """
    Returns a DataFrame with columns [total_evals, global_best_mse] where
    total_evals is the sum of all genomes inserted across ALL active ranks
    at each point in wall-clock time.

    Strategy: for each unique wall-clock time step (union of all ranks'
    fitness_log rows), compute the sum of genomes inserted across all ranks
    and the minimum MSE seen so far globally.
    """
    # Build per-rank (time_s, inserted_genomes, best_mse) tables
    rank_frames = {}
    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None or fl.empty:
            continue
        fl = clean_infinity(fl.copy(), ["Best Val. MSE"])
        fl["time_s"] = fl["Time"] / 1000.0
        fl = fl.sort_values("time_s").reset_index(drop=True)
        rank_frames[rank] = fl[["time_s", "Inserted Genomes", "Best Val. MSE"]].dropna()

    if not rank_frames:
        return None

    # Collect all time points across all ranks
    all_times = sorted(set(
        t for df in rank_frames.values() for t in df["time_s"].tolist()
    ))

    rows = []
    for t in all_times:
        total_evals = 0
        best_mse    = float("inf")
        for df in rank_frames.values():
            # Latest entry at or before time t for this rank
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


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_convergence_vs_time(rank_data, ref_events, save_dir, show):
    """
    Plot 1 — Best MSE vs wall-clock time (log scale).
    Per-rank curves + global minimum. Fault event markers.
    Shows that wall-clock convergence slows during dropout (hardware lost).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None or fl.empty:
            continue
        fl = clean_infinity(fl.copy(), ["Best Val. MSE"])
        fl["time_s"] = fl["Time"] / 1000.0
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.plot(fl["time_s"], fl["Best Val. MSE"],
                color=color, linewidth=1.2, alpha=0.55, label=f"Rank {rank}")

    global_curve = build_global_best_vs_time(rank_data)
    if global_curve is not None:
        ax.plot(global_curve["time_s"], global_curve["global_best_mse"],
                color="black", linewidth=2.5, label="Global best", zorder=5)

    if ref_events is not None and not ref_events.empty:
        ylim = ax.get_ylim()
        add_event_vlines(ax, ref_events, label_y=ylim[1])

    ax.set_xlabel("Wall-Clock Time (s)")
    ax.set_ylabel("Best Validation MSE")
    ax.set_title("Best Fitness vs. Wall-Clock Time\n"
                 "(curve slows during dropout — hardware lost, graceful degradation)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "01_global_convergence_time.png", show)


def plot_convergence_vs_evals(rank_data, ref_events, save_dir, show):
    """
    Plot 2 — Best MSE vs total global evaluations (sum across all active ranks).
    This is the key algorithmic-efficiency chart: if the P2P system is truly
    fault-tolerant, the fitness curve relative to evaluation count should be
    unaffected by node failures (only wall-clock time changes, not efficiency).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    evals_curve = build_global_best_vs_evals(rank_data)
    if evals_curve is not None:
        ax.plot(evals_curve["total_evals"], evals_curve["global_best_mse"],
                color="black", linewidth=2.5, label="Global best MSE",
                zorder=5)

    # Also draw per-rank for reference
    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None or fl.empty:
            continue
        fl = clean_infinity(fl.copy(), ["Best Val. MSE"])
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.plot(fl["Inserted Genomes"], fl["Best Val. MSE"],
                color=color, linewidth=1.0, alpha=0.45,
                linestyle="--", label=f"Rank {rank} (local)")

    # Mark fault events by total evaluations at event time
    if ref_events is not None and not ref_events.empty and evals_curve is not None:
        # For each event, find the closest total_evals at that wall time
        # (use rank-0 fitness_log as proxy for total evals at each time)
        fl0 = rank_data.get(0, {}).get("fitness_log")
        if fl0 is not None and not fl0.empty:
            fl0 = fl0.copy()
            fl0["time_s"] = fl0["Time"] / 1000.0
            for _, row in ref_events.iterrows():
                et = row["event_type"]
                if et not in ("DROPOUT_START", "REJOIN_COMPLETE",
                              "GRACEFUL_SHUTDOWN_SENT"):
                    continue
                # Find nearest total_evals for this wall time
                t = row["wall_time_s"]
                nearest = evals_curve.iloc[
                    (evals_curve["total_evals"] - t).abs().argmin()
                ] if len(evals_curve) > 0 else None
                if nearest is not None:
                    color = EVENT_COLORS.get(et, "#888")
                    ax.axvline(nearest["total_evals"], color=color,
                               linewidth=1.5, linestyle="-", alpha=0.65)

    ax.set_xlabel("Total Global Genome Evaluations (sum across all ranks)")
    ax.set_ylabel("Best Validation MSE")
    ax.set_title("Best Fitness vs. Total Evaluations  ←  Strongest Evidence\n"
                 "(if this curve is smooth despite faults, algorithmic efficiency is unaffected)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "02_global_convergence_evals.png", show)


def plot_fault_timeline(rank_data, all_events, max_time, save_dir, show):
    """
    Plot 3 — Gantt-style fault timeline.
    Green = active, red = dropped (unintentional), purple = gracefully shut down.
    """
    ranks = sorted(rank_data.keys())
    n_ranks = len(ranks)

    fig, ax = plt.subplots(figsize=(12, max(3, n_ranks * 0.9)))

    for i, rank in enumerate(ranks):
        ev_df = all_events.get(rank)
        y = i

        active_spans  = [(0.0, max_time)]
        dropout_spans = []
        shutdown_start = None

        if ev_df is not None:
            for _, row in ev_df.iterrows():
                et = row["event_type"]
                t  = row["wall_time_s"]
                if et == "DROPOUT_START":
                    active_spans[-1] = (active_spans[-1][0], t)
                    dropout_spans.append((t, max_time))
                elif et == "REJOIN_COMPLETE":
                    if dropout_spans:
                        dropout_spans[-1] = (dropout_spans[-1][0], t)
                    active_spans.append((t, max_time))
                elif et == "GRACEFUL_SHUTDOWN_SENT":
                    if active_spans:
                        active_spans[-1] = (active_spans[-1][0], t)
                    shutdown_start = t

        for s, e in active_spans:
            ax.barh(y, e - s, left=s, height=0.6,
                    color=RANK_COLORS[rank % len(RANK_COLORS)], alpha=0.75)
        for s, e in dropout_spans:
            ax.barh(y, e - s, left=s, height=0.6, color="#FF5722", alpha=0.75)
        if shutdown_start is not None:
            ax.barh(y, max_time - shutdown_start, left=shutdown_start,
                    height=0.6, color="#9C27B0", alpha=0.75)
            ax.axvline(shutdown_start, color="#9C27B0", linestyle="--",
                       linewidth=1.4, alpha=0.7)

    ax.set_yticks(range(n_ranks))
    ax.set_yticklabels([f"Rank {r}" for r in ranks])
    ax.set_xlabel("Wall time (s)")
    ax.set_title("Fault Event Timeline (Gantt)", fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=RANK_COLORS[0], alpha=0.75, label="Active"),
        mpatches.Patch(color="#FF5722",       alpha=0.75, label="Dropped (unintentional)"),
        mpatches.Patch(color="#9C27B0",       alpha=0.75, label="Graceful shutdown"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "03_fault_timeline.png", show)


def plot_active_rank_count(all_events, n_ranks, max_time, save_dir, show):
    """
    Plot 4 — Step plot of active rank count over time.
    """
    events_list = []
    for _, ev_df in all_events.items():
        if ev_df is None:
            continue
        for _, row in ev_df.iterrows():
            et = row["event_type"]
            if et in ("HEARTBEAT_TIMEOUT", "PEER_FAILED_RECEIVED",
                      "GRACEFUL_SHUTDOWN_SENT", "GRACEFUL_SHUTDOWN_RECEIVED",
                      "REJOIN_COMPLETE", "REJOIN_SEEDED"):
                events_list.append({
                    "time":         row["wall_time_s"],
                    "active_count": row["active_ranks_count"],
                    "event":        et,
                })

    if not events_list:
        return

    ev_all = (pd.DataFrame(events_list)
                .sort_values("time")
                .drop_duplicates("time"))

    fig, ax = plt.subplots(figsize=(12, 4))

    times  = [0.0]      + list(ev_all["time"])         + [max_time]
    counts = [n_ranks]  + list(ev_all["active_count"]) + [ev_all["active_count"].iloc[-1]]

    ax.step(times, counts, where="post", color="#2196F3", linewidth=2.2)
    ax.fill_between(times, counts, step="post", alpha=0.15, color="#2196F3")

    for _, row in ev_all.iterrows():
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


def plot_throughput(rank_data, ref_events, save_dir, show):
    """
    Plot 5 — Genomes/sec per rank over time.
    Derived from fitness_log (delta genomes / delta time).
    Dropout period appears as near-zero flatline; recovery shows resumption.
    """
    SMOOTH = 5

    fig, ax = plt.subplots(figsize=(12, 5))

    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None or len(fl) < 2:
            continue
        fl = fl.copy()
        fl["time_s"] = fl["Time"] / 1000.0
        fl = fl.sort_values("time_s").reset_index(drop=True)

        dt   = fl["time_s"].diff()
        dg   = fl["Inserted Genomes"].diff()
        rate = (dg / dt).replace([np.inf, -np.inf], np.nan)
        rate = rate.rolling(SMOOTH, min_periods=1, center=True).mean()

        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.plot(fl["time_s"], rate, color=color, linewidth=1.5,
                alpha=0.85, label=f"Rank {rank}")

    # Fault event lines
    if ref_events is not None and not ref_events.empty:
        for _, row in ref_events.iterrows():
            et = row["event_type"]
            if et not in ("DROPOUT_START", "REJOIN_COMPLETE",
                          "GRACEFUL_SHUTDOWN_SENT"):
                continue
            color = EVENT_COLORS.get(et, "#888")
            ax.axvline(row["wall_time_s"], color=color, linewidth=1.6,
                       linestyle="-", alpha=0.7)
            ax.text(row["wall_time_s"] + 1,
                    ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    et.replace("_", "\n"), fontsize=5.5, color=color,
                    va="top", alpha=0.85)

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Genomes inserted / sec")
    ax.set_title("Throughput per Rank Over Time\n"
                 "(dropout → flatline; recovery → resumption of pre-fault throughput)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "05_throughput.png", show)


def plot_rmse_over_time(rank_data, ref_events, save_dir, show):
    """
    Plot 6 — RMSE over time per rank.
    Uses test_metrics_log.csv if available, else falls back to sqrt(MSE).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    used_test_metrics = False
    for rank, data in rank_data.items():
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        tm = data.get("test_metrics")

        if tm is not None and not tm.empty and "rmse" in tm.columns:
            used_test_metrics = True
            ax.plot(tm["wall_time_s"], tm["rmse"],
                    color=color, linewidth=1.3, alpha=0.8, label=f"Rank {rank}")
        else:
            fl = data.get("fitness_log")
            if fl is None or fl.empty:
                continue
            fl = clean_infinity(fl.copy(), ["Best Val. MSE"])
            fl["time_s"] = fl["Time"] / 1000.0
            fl["rmse"]   = np.sqrt(fl["Best Val. MSE"].clip(lower=0))
            ax.plot(fl["time_s"], fl["rmse"],
                    color=color, linewidth=1.3, alpha=0.8,
                    linestyle="--", label=f"Rank {rank} (√MSE fallback)")

    if ref_events is not None and not ref_events.empty:
        ylim = ax.get_ylim()
        add_event_vlines(ax, ref_events, label_y=ylim[1])

    source = "test_metrics_log.csv" if used_test_metrics else "fitness_log.csv (√MSE)"
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("RMSE (normalized scale)")
    ax.set_title(f"RMSE Over Time per Rank  [source: {source}]", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "06_rmse_over_time.png", show)


def plot_r2_over_time(rank_data, ref_events, save_dir, show):
    """
    Plot 7 — R² over time per rank from test_metrics_log.csv.
    Skips with a message if test_metrics_log.csv is absent.
    """
    has_data = any(
        d.get("test_metrics") is not None
        and not d["test_metrics"].empty
        and "r2" in d["test_metrics"].columns
        for d in rank_data.values()
    )
    if not has_data:
        print("  [SKIP] 07_r2_over_time.png — no test_metrics_log.csv "
              "(rerun with updated binary)")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for rank, data in rank_data.items():
        tm = data.get("test_metrics")
        if tm is None or tm.empty or "r2" not in tm.columns:
            continue
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.plot(tm["wall_time_s"], tm["r2"],
                color=color, linewidth=1.3, alpha=0.8, label=f"Rank {rank}")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.4,
               label="Perfect  R²=1")
    ax.axhline(0.0, color="red",   linewidth=0.8, linestyle=":", alpha=0.4,
               label="R²=0 (predicts mean)")

    if ref_events is not None and not ref_events.empty:
        add_event_vlines(ax, ref_events)

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("R²  (coefficient of determination)")
    ax.set_title("R² Over Time per Rank", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "07_r2_over_time.png", show)


def plot_knowledge_preservation(rank_data, ref_events, save_dir, show):
    """
    Plot 8 — Fitness immediately before fault vs immediately after recovery.
    Equal bars = no knowledge lost.
    Lower 'after' bar = peers improved while rank was down (continued evolving).
    """
    if ref_events is None or ref_events.empty:
        print("  [SKIP] 08_knowledge_preservation.png — no fault events recorded")
        return

    events = ref_events.sort_values("wall_time_s")
    triples = []

    for _, row in events.iterrows():
        et = row["event_type"]
        if et == "DROPOUT_START":
            rank = row["subject_rank"]
            fitness_before = row["best_fitness"]
            rejoin = events[
                (events["event_type"] == "REJOIN_COMPLETE") &
                (events["subject_rank"] == rank) &
                (events["wall_time_s"] > row["wall_time_s"])
            ]
            if not rejoin.empty:
                r = rejoin.iloc[0]
                triples.append({
                    "label":   (f"Rank {int(rank)}\n"
                                f"Dropout→Rejoin\n"
                                f"({r.get('downtime_s', 0):.1f}s down)"),
                    "before":  fitness_before,
                    "after":   r["best_fitness"],
                    "kind":    "dropout",
                })
        elif et == "GRACEFUL_SHUTDOWN_SENT":
            rank = row["subject_rank"]
            recv = events[
                (events["event_type"] == "GRACEFUL_SHUTDOWN_RECEIVED") &
                (events["subject_rank"] == rank)
            ]
            if not recv.empty:
                triples.append({
                    "label":  f"Rank {int(rank)}\nGraceful\nShutdown",
                    "before": row["best_fitness"],
                    "after":  recv.iloc[0]["best_fitness"],
                    "kind":   "shutdown",
                })

    if not triples:
        print("  [SKIP] 08_knowledge_preservation.png — no matching event pairs found")
        return

    n = len(triples)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, n * 2.8), 5))

    befores = [t["before"] for t in triples]
    afters  = [t["after"]  for t in triples]
    labels  = [t["label"]  for t in triples]
    kinds   = [t["kind"]   for t in triples]

    bar_b = ax.bar(x - w / 2, befores, w,
                   color=["#FF5722" if k == "dropout" else "#9C27B0" for k in kinds],
                   alpha=0.8, label="Fitness at fault event")
    bar_a = ax.bar(x + w / 2, afters, w,
                   color=["#2196F3" if k == "dropout" else "#4CAF50" for k in kinds],
                   alpha=0.8, label="Fitness at recovery / peer reception")

    for rect, val in zip(list(bar_b) + list(bar_a), befores + afters):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    for i, t in enumerate(triples):
        if t["before"] > 0:
            pct   = 100 * (t["after"] - t["before"]) / t["before"]
            sign  = "↓" if pct < 0 else ("=" if abs(pct) < 0.01 else "↑")
            color = "#4CAF50" if pct <= 0 else "#F44336"
            ax.text(i, max(t["before"], t["after"]) * 1.12,
                    f"{sign} {abs(pct):.1f}%",
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best Fitness (MSE, lower = better)")
    ax.set_title("Knowledge Preservation Across Fault Events\n"
                 "(equal = no loss; lower 'after' = peers kept improving while rank was down)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "08_knowledge_preservation.png", show)


def plot_mae_convergence(rank_data, ref_events, save_dir, show):
    """
    Plot 9 — Best validation MAE over time per rank.
    MAE is in the same units as the normalized target (Pitch), making it
    more interpretable than MSE for discussing prediction accuracy.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for rank, data in rank_data.items():
        fl = data.get("fitness_log")
        if fl is None or fl.empty:
            continue
        fl = clean_infinity(fl.copy(), ["Best Val. MAE"])
        fl["time_s"] = fl["Time"] / 1000.0
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.plot(fl["time_s"], fl["Best Val. MAE"],
                color=color, linewidth=1.3, alpha=0.8, label=f"Rank {rank}")

    if ref_events is not None and not ref_events.empty:
        ylim = ax.get_ylim()
        add_event_vlines(ax, ref_events, label_y=ylim[1])

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Best Validation MAE (normalized)")
    ax.set_title("MAE Convergence Over Time per Rank", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_dir, "09_mae_convergence.png", show)


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(rank_data, all_events):
    W = 110
    print("\n" + "=" * W)
    print("  FAULT TOLERANCE RUN SUMMARY")
    print("=" * W)

    header = (f"{'Rank':<6} {'Final MSE':<12} {'RMSE':<10} {'Final R²':<10} "
              f"{'Genomes':<10} {'Dropouts':<10} {'Shutdowns':<10} "
              f"{'Recoveries':<12} {'Downtime(s)':<13} {'Wall Time':<10}")
    print(header)
    print("-" * W)

    for rank, data in sorted(rank_data.items()):
        s  = data.get("summary")
        tm = data.get("test_metrics")

        if s is None or s.empty:
            continue
        row  = s.iloc[0]
        mse  = float(row.get("final_best_fitness", float("nan")))
        rmse = math.sqrt(mse) if mse > 0 else float("nan")
        r2   = float("nan")
        if tm is not None and not tm.empty and "r2" in tm.columns:
            r2 = float(tm["r2"].iloc[-1])

        print(f"{rank:<6} {mse:<12.6f} {rmse:<10.6f} {r2:<10.4f} "
              f"{int(row.get('total_genomes_evaluated', 0)):<10} "
              f"{int(row.get('total_dropout_events', 0)):<10} "
              f"{int(row.get('total_shutdown_events', 0)):<10} "
              f"{int(row.get('total_recovery_events', 0)):<12} "
              f"{float(row.get('total_downtime_s', 0)):<13.2f} "
              f"{float(row.get('total_wall_time_s', 0)):<10.1f}")

    rows = []
    for _, ev_df in all_events.items():
        if ev_df is None:
            continue
        for _, r in ev_df.iterrows():
            rows.append(r.to_dict())

    if rows:
        all_ev = pd.DataFrame(rows).sort_values("wall_time_s")
        print(f"\n  Fault Event Log (chronological across all ranks):\n")
        hdr = (f"  {'Time(s)':<10} {'Reporter':<10} {'Event':<36} "
               f"{'Subject':<8} {'Active':<8} {'Best MSE':<12} {'Downtime(s)'}")
        print(hdr)
        print("  " + "-" * 95)
        for _, r in all_ev.iterrows():
            dt = f"{r['downtime_s']:.1f}" if float(r['downtime_s']) >= 0 else "-"
            print(f"  {r['wall_time_s']:<10.1f} {r['reporting_rank']:<10} "
                  f"{r['event_type']:<36} {r['subject_rank']:<8} "
                  f"{r['active_ranks_count']:<8} {r['best_fitness']:<12.4f} {dt}")

    print("=" * W + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize P2P EXAMM fault-tolerance run output")
    parser.add_argument("output_dir",
                        help="Path to run output directory")
    parser.add_argument("--save_dir", default=None,
                        help="Where to write PNGs (default: <output_dir>/plots)")
    parser.add_argument("--show", action="store_true",
                        help="Also open each figure interactively")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    save_dir   = args.save_dir or os.path.join(output_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    if args.show:
        matplotlib.use("TkAgg")

    print(f"\nReading from : {output_dir}")
    print(f"Saving plots : {save_dir}\n")

    # ── Load all per-rank data ─────────────────────────────────────────────────
    rank_dirs  = load_rank_dirs(output_dir)
    rank_data  = {}
    all_events = {}
    max_time   = 0.0
    n_ranks    = len(rank_dirs)

    for d in rank_dirs:
        rank = rank_from_dir(d)
        fl   = read_csv_safe(os.path.join(d, "fitness_log.csv"))
        sm   = read_csv_safe(os.path.join(d, "fault_tolerance_summary.csv"))
        ev   = read_csv_safe(os.path.join(d, "fault_tolerance_events.csv"))
        tm   = read_csv_safe(os.path.join(d, "test_metrics_log.csv"))

        has_tm   = tm is not None and not tm.empty and "rmse" in (tm.columns if tm is not None else [])
        loaded   = [name for name, df in [("fitness_log", fl), ("summary", sm)]
                    if df is not None and not df.empty]
        n_ev     = len(ev) - 1 if ev is not None and not ev.empty else 0
        tm_note  = " [test_metrics ✓]" if has_tm else " [test_metrics — rerun needed for R²/RMSE]"
        print(f"  rank {rank}: loaded {loaded}  |  {n_ev} fault events{tm_note}")

        rank_data[rank]  = {"fitness_log": fl, "summary": sm, "test_metrics": tm}
        all_events[rank] = ev

        if sm is not None and not sm.empty and "total_wall_time_s" in sm.columns:
            max_time = max(max_time, float(sm["total_wall_time_s"].iloc[0]))

    if max_time == 0.0:
        max_time = 1000.0

    ref_events = build_reference_events(all_events)

    # ── Generate plots ─────────────────────────────────────────────────────────
    print(f"\nGenerating plots (total run time ≈ {max_time:.0f}s):\n")

    plot_convergence_vs_time(rank_data, ref_events, save_dir, args.show)
    plot_convergence_vs_evals(rank_data, ref_events, save_dir, args.show)
    plot_fault_timeline(rank_data, all_events, max_time, save_dir, args.show)
    plot_active_rank_count(all_events, n_ranks, max_time, save_dir, args.show)
    plot_throughput(rank_data, ref_events, save_dir, args.show)
    plot_rmse_over_time(rank_data, ref_events, save_dir, args.show)
    plot_r2_over_time(rank_data, ref_events, save_dir, args.show)
    plot_knowledge_preservation(rank_data, ref_events, save_dir, args.show)
    plot_mae_convergence(rank_data, ref_events, save_dir, args.show)

    print_summary_table(rank_data, all_events)
    print(f"All done. Plots saved to: {save_dir}\n")


if __name__ == "__main__":
    main()

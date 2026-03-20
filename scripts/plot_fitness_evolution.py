#!/usr/bin/env python3
"""
Script to combine and plot fitness logs from decentralized EXAMM runs.

This script:
1. Finds all fitness_log.csv files in subdirectories (e.g., p2p_rank_0/, p2p_rank_1/, etc.)
2. Combines them by averaging fitness values across ranks for each "Inserted Genomes" step
3. Plots the evolution of "Best Val. MSE" over time

Requirements: pandas, matplotlib
Install with: pip install pandas matplotlib

Usage: Run this script in the directory containing the rank subdirectories (e.g., test_output/c172_mpi/)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def combine_fitness_logs(directory="."):
    """
    Combine fitness_log.csv files from multiple MPI ranks.

    Args:
        directory (str): Directory containing rank subdirectories (default: current dir)

    Returns:
        pd.DataFrame: Combined dataframe with averaged fitness values
    """
    # Find all fitness_log.csv files in rank subdirectories (supports nested paths)
    pattern = os.path.join(directory, "**", "p2p_rank_*")
    pattern = os.path.join(directory, "**", "p2p_rank_*/fitness_log.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No fitness_log.csv files found in: {pattern}")
        return None

    print(f"Found {len(files)} fitness log files: {files}")

    # Read and combine dataframes
    dfs = []
    for f in files:
        df = pd.read_csv(f, skipinitialspace=True)
        dfs.append(df)

    # Concatenate and group by 'Inserted Genomes', averaging numeric columns
    combined = pd.concat(dfs, ignore_index=True)
    # Group by Inserted Genomes and compute mean for numeric columns (exclude grouping key)
    numeric_cols = combined.select_dtypes(include=[float, int]).columns.drop('Inserted Genomes', errors='ignore')
    combined_avg = combined.groupby('Inserted Genomes')[numeric_cols].mean().reset_index()

    print(f"Combined data has {len(combined_avg)} rows")
    return combined_avg

def plot_fitness_evolution(df, save_path=None):
    """
    Plot the evolution of best validation MSE over time.

    Args:
        df (pd.DataFrame): Combined fitness dataframe
        save_path (str): Optional path to save the plot
    """
    if df is None or df.empty:
        print("No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Best Val. MSE vs Inserted Genomes
    ax1.plot(df['Inserted Genomes'], df['Best Val. MSE'], 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Inserted Genomes')
    ax1.set_ylabel('Best Validation MSE')
    ax1.set_title('Evolution of Best Validation MSE')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale often better for MSE

    # Plot 2: Best Val. MSE vs Time
    ax2.plot(df['Time'], df['Best Val. MSE'], 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Best Validation MSE')
    ax2.set_title('Best Validation MSE Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot DEAMM fitness evolution from rank logs')
    parser.add_argument('--root', default='.', help='Root directory under which to scan for p2p_rank_*/fitness_log.csv')
    parser.add_argument('--output-csv', default='combined_fitness_log.csv', help='Output CSV filename')
    parser.add_argument('--output-img', default='fitness_evolution_plot.png', help='Output plot image filename')
    args = parser.parse_args()

    combined_df = combine_fitness_logs(args.root)

    if combined_df is not None and not combined_df.empty:
        combined_df.to_csv(args.output_csv, index=False)
        print(f"Combined fitness log saved to {args.output_csv}")
        plot_fitness_evolution(combined_df, save_path=args.output_img)
    else:
        print('No data combined; exiting.')


if __name__ == "__main__":
    main()
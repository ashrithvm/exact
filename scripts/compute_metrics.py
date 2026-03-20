#!/usr/bin/env python3
"""
Script to compute additional accuracy metrics from the combined fitness log.

This script:
- Reads the combined_fitness_log.csv
- Extracts the best validation MSE and MAE
- Computes derived metrics like RMSE
- Prints a summary for comparison

Requirements: pandas, numpy
Install with: pip install pandas numpy
"""

import pandas as pd
import numpy as np

def compute_metrics(csv_path):
    """
    Compute and print accuracy metrics from the fitness log.

    Args:
        csv_path (str): Path to combined_fitness_log.csv
    """
    df = pd.read_csv(csv_path)

    # Get the final best metrics (last row)
    final_row = df.iloc[-1]
    best_mse = final_row['Best Val. MSE']
    best_mae = final_row['Best Val. MAE']

    # Derived metrics
    rmse = np.sqrt(best_mse)  # Root Mean Squared Error
    mape = None  # Mean Absolute Percentage Error - can't compute without actual values

    print("=== Accuracy Metrics Summary ===")
    print(f"Best Validation MSE: {best_mse:.6f}")
    print(f"Best Validation MAE: {best_mae:.6f}")
    print(f"Best Validation RMSE: {rmse:.6f}")
    print(f"Total Genomes Evolved: {int(final_row['Inserted Genomes'])}")
    print(f"Total Time (ms): {final_row['Time']:.0f}")
    print(f"Network Complexity: {int(final_row['Enabled Nodes'])} nodes, {int(final_row['Enabled Edges'])} edges")

    # Trend analysis
    initial_mse = df['Best Val. MSE'].iloc[0]
    improvement = (initial_mse - best_mse) / initial_mse * 100
    print(f"MSE Improvement: {improvement:.1f}% from initial to final")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute accuracy metrics from fitness log')
    parser.add_argument('--csv', default='combined_fitness_log.csv', help='Path to combined_fitness_log.csv')
    args = parser.parse_args()

    compute_metrics(args.csv)
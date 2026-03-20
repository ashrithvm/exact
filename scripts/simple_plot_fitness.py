#!/usr/bin/env python3
"""
Simple CSV plotter using only built-in Python libraries.
No external dependencies required.

Usage: python3 simple_plot_fitness.py
Run in the directory containing p2p_rank_*/ subdirectories.
"""

import argparse
import csv
import os
import glob
from collections import defaultdict

def read_csv_simple(filepath):
    """Read CSV file into list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            normalized_row = {}
            # Convert numeric columns and strip whitespace from keys
            for key, value in row.items():
                if key is None:
                    continue
                key = key.strip()
                try:
                    normalized_row[key] = float(value)
                except (ValueError, TypeError):
                    normalized_row[key] = value
            data.append(normalized_row)
    return data

def combine_fitness_logs(directory="."):
    """Combine fitness logs by averaging across ranks."""
    pattern = os.path.join(directory, "**", "p2p_rank_*/fitness_log.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No fitness_log.csv files found in: {pattern}")
        return {}

    print(f"Found {len(files)} fitness log files")

    # Group data by 'Inserted Genomes'
    grouped_data = defaultdict(list)

    for filepath in files:
        data = read_csv_simple(filepath)
        for row in data:
            genomes = int(row['Inserted Genomes'])
            grouped_data[genomes].append(row)

    # Average numeric values for each genome count
    combined = {}
    for genomes, rows in grouped_data.items():
        if not rows:
            continue

        # Get all numeric keys from first row
        numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]

        averaged_row = {'Inserted Genomes': genomes}
        for key in numeric_keys:
            if key == 'Inserted Genomes':
                continue
            values = [row[key] for row in rows if key in row]
            averaged_row[key] = sum(values) / len(values) if values else 0

        combined[genomes] = averaged_row

    print(f"Combined {len(combined)} unique genome insertion points")
    return combined

def print_summary_stats(combined_data):
    """Print basic statistics from the combined data."""
    if not combined_data:
        return

    genomes_list = sorted(combined_data.keys())
    first_row = combined_data[genomes_list[0]]
    last_row = combined_data[genomes_list[-1]]

    print("\n=== FITNESS EVOLUTION SUMMARY ===")
    print(f"Total genome insertions: {len(genomes_list)}")
    print(".6f")
    print(".6f")
    print(".2f")
    print(".2f")

    # Find best fitness
    best_mse = min(row['Best Val. MSE'] for row in combined_data.values())
    best_genomes = next(g for g, row in combined_data.items() if row['Best Val. MSE'] == best_mse)
    print(".6f")

    # Show improvement trend
    mse_values = [combined_data[g]['Best Val. MSE'] for g in genomes_list]
    if len(mse_values) > 1:
        improvement = mse_values[0] - mse_values[-1]
        print(".6f")

def save_combined_csv(combined_data, output_file="combined_fitness_log.csv"):
    """Save combined data to CSV."""
    if not combined_data:
        return

    genomes_list = sorted(combined_data.keys())
    first_row = combined_data[genomes_list[0]]
    fieldnames = ['Inserted Genomes'] + [k for k in first_row.keys() if k != 'Inserted Genomes']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for genomes in genomes_list:
            writer.writerow(combined_data[genomes])

    print(f"Combined data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Simple DEAMM fitness evolution plotter')
    parser.add_argument('--root', default='.', help='Root directory under which to scan for p2p_rank_*/fitness_log.csv')
    parser.add_argument('--output-csv', default='combined_fitness_log.csv', help='Output CSV filename')
    args = parser.parse_args()

    combined = combine_fitness_logs(args.root)

    if combined:
        print_summary_stats(combined)
        save_combined_csv(combined, output_file=args.output_csv)

        # Simple text-based plot
        print("\n=== SIMPLE MSE PLOT ===")
        genomes_list = sorted(combined.keys())
        mse_values = [combined[g]['Best Val. MSE'] for g in genomes_list]

        # Normalize for plotting (simple bar chart)
        if mse_values:
            min_mse = min(mse_values)
            max_mse = max(mse_values)
            range_mse = max_mse - min_mse if max_mse != min_mse else 1

            print("Best Val. MSE evolution (higher bars = worse fitness):")
            for i, (g, mse) in enumerate(zip(genomes_list, mse_values)):
                if i % 10 == 0:  # Print every 10th point to avoid clutter
                    bar_length = int(50 * (mse - min_mse) / range_mse)
                    bar = '█' * bar_length
                    print("6d")

if __name__ == "__main__":
    main()
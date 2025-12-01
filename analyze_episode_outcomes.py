"""
analyze_episode_outcomes.py

Scan directories where GameController logs per-episode CSVs and produce summary statistics and plots.

Usage:
    python analyze_episode_outcomes.py --run-path "path/to/trajectory_logs_default_run" --out output_dir

If --run-path omitted will search for directories matching trajectory_logs_* under current working directory.

Output:
 - summary printed to stdout
 - saved charts into the output dir
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_episode_meta_files(run_path):
    # run_path may be a specific run folder, or a workspace root.
    results = []
    # If a `combine.csv` exists at the run folder, prefer that
    combine = os.path.join(run_path, 'combine.csv')
    if os.path.exists(combine):
        return [combine]

    # If run_path is a workspace root, check for combine files in trajectory_log folders
    if os.path.isdir(run_path):
        # pattern like workspace/trajectory_logs_* /combine.csv
        for d in glob.glob(os.path.join(run_path, 'trajectory_logs_*')):
            c = os.path.join(d, 'combine.csv')
            if os.path.exists(c):
                results.append(c)
        if results:
            return sorted(results)
    if os.path.exists(os.path.join(run_path, 'episode_0')) or os.path.exists(run_path):
        search_pattern = os.path.join(run_path, 'episode_*', 'episode_meta.csv')
    else:
        # find directories matching trajectory_logs_
        search_pattern = os.path.join(run_path, 'trajectory_logs_*', 'episode_*', 'episode_meta.csv')
    for path in glob.glob(search_pattern):
        results.append(path)
    return sorted(results)


def read_metadata(files):
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                # If file contains multiple rows (e.g., combine.csv), add all
                for _, r in df.iterrows():
                    rows.append(r)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    if rows:
        # rows may be a list of Series or dict-like rows; normalize to DataFrame
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()


def summarize_and_plot(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = len(df)
    print(f"Total episodes found: {total}")

    if total == 0:
        print("No episodes found; aborting")
        return

    # Normalize types
    df['outcome'] = df['outcome'].astype(str)
    df['winner'] = df['winner'].astype(str)

    # Basic counts - `outcome` is the configured outcome according to GameController's `successPerspective`
    if 'outcome' in df.columns:
        counts = df['outcome'].value_counts()
        print("Outcome counts (configured):\n", counts.to_string())
    else:
        print("No 'outcome' column found in metadata files.")

    winners_count = df['winner'].value_counts()
    print("Winner counts:\n", winners_count.to_string())

    # Also report outcome counts specifically from Seekers/Hiders perspective if available
    if 'outcome_seekers' in df.columns:
        print("Outcome counts (seekers perspective):\n", df['outcome_seekers'].value_counts().to_string())
    if 'outcome_hiders' in df.columns:
        print("Outcome counts (hiders perspective):\n", df['outcome_hiders'].value_counts().to_string())

    # Statistics for numeric columns
    numeric_cols = ['episode_duration_seconds', 'stepsHidden', 'timeHidden', 'hidersCaptured']
    for col in numeric_cols:
        if col in df.columns:
            print(f"{col} stats:")
            print(df[col].describe())

    # Plot 1: Outcome counts bar chart (configured)
    if 'outcome' in df.columns:
        plt.figure(figsize=(6,4))
        counts.reindex(['Success','Failure','Timeout']).fillna(0).plot(kind='bar', color=['#2ca02c','#d62728','#ff7f0e'])
    plt.title('Episode Outcome Counts')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'outcome_counts.png'))
    plt.close()

    # Plot 1b: Outcome counts for seekers/hiders if available
    if 'outcome_seekers' in df.columns:
        plt.figure(figsize=(6,4))
        df['outcome_seekers'].value_counts().reindex(['Success','Failure','Timeout']).fillna(0).plot(kind='bar', color=['#2ca02c','#d62728','#ff7f0e'])
        plt.title('Outcome Counts (Seekers perspective)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'outcome_counts_seekers.png'))
        plt.close()
    if 'outcome_hiders' in df.columns:
        plt.figure(figsize=(6,4))
        df['outcome_hiders'].value_counts().reindex(['Success','Failure','Timeout']).fillna(0).plot(kind='bar', color=['#2ca02c','#d62728','#ff7f0e'])
        plt.title('Outcome Counts (Hiders perspective)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'outcome_counts_hiders.png'))
        plt.close()

    # Plot 2: Winner counts
    plt.figure(figsize=(6,4))
    winners_count.plot(kind='bar', color=['#1f77b4','#ff7f0e'])
    plt.title('Winner Counts')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'winner_counts.png'))
    plt.close()

    # Plot 3: timeHidden timeseries
    if 'timeHidden' in df.columns:
        plt.figure(figsize=(8,4))
        df['timeHidden'] = pd.to_numeric(df['timeHidden'], errors='coerce')
        df['episode_index'] = pd.to_numeric(df['episode_index'], errors='coerce')
        df = df.sort_values(by='episode_index')
        plt.plot(df['episode_index'], df['timeHidden'], marker='o', linestyle='-')
        plt.title('Time Hidden (per episode)')
        plt.xlabel('Episode Index')
        plt.ylabel('Time Hidden (fraction)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'time_hidden_per_episode.png'))
        plt.close()

    # Plot 4: episode duration histogram
    if 'episode_duration_seconds' in df.columns:
        plt.figure(figsize=(6,4))
        df['episode_duration_seconds'].hist(bins=30)
        plt.title('Episode Duration (seconds)')
        plt.xlabel('Seconds')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'episode_duration_hist.png'))
        plt.close()
    if 'stepsHidden' in df.columns:
        plt.figure(figsize=(6,4))
        df['stepsHidden'].hist(bins=30, orientation='horizontal')
        plt.title('Steps Hidden Histogram')
        plt.xlabel('Steps Hidden')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'steps_hidden_hist.png'))
        plt.close()

    # Save summary CSV
    out_csv = os.path.join(out_dir, 'episode_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved summary CSV: {out_csv}")
    print(f"Saved charts to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-path', default='.', help='Path to run directory or workspace root (searches for trajectory_logs_*)')
    parser.add_argument('--out', default='analysis_output', help='Output folder for summary and charts')
    args = parser.parse_args()

    files = find_episode_meta_files(args.run_path)
    print(f"Found {len(files)} episode_meta.csv files")
    df = read_metadata(files)
    if df.empty:
        print("No episode metadata rows loaded. Are you pointing to the right folder?")
        return

    summarize_and_plot(df, args.out)


if __name__ == '__main__':
    main()

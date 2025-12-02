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
from datetime import datetime
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


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


def load_run_df(run_path):
    """Load a single run's metadata as a DataFrame.
    The function prefers run-level `combine.csv` (if available); otherwise, merges per-episode `episode_meta.csv` files.
    run_path can be the run folder, or the workspace root containing `trajectory_logs_*` folders.
    """
    candidate_files = find_episode_meta_files(run_path)
    if not candidate_files:
        return pd.DataFrame()
    # If a single combine.csv, read all rows
    if len(candidate_files) == 1 and os.path.basename(candidate_files[0]) == 'combine.csv':
        try:
            df = pd.read_csv(candidate_files[0])
            df['run_path'] = run_path
            return df
        except Exception as e:
            print(f"Failed to read combine file {candidate_files[0]}: {e}")
            return pd.DataFrame()
    # Otherwise, multiple episode_meta files -> merge
    rows = []
    for f in candidate_files:
        try:
            df = pd.read_csv(f)
            # each episode file will include a single row
            if not df.empty:
                for _, r in df.iterrows():
                    r = r.copy()
                    r['source_file'] = f
                    rows.append(r)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df['run_path'] = run_path
        return df
    return pd.DataFrame()


def find_latest_runs(base_path='.', limit=2):
    """Find the latest `limit` run folders under trajectory_logs_* based on modification time and return full paths.
    If base_path is itself a run-like folder (contains combine.csv) it will be returned directly.
    """
    # If base_path points directly to a combine.csv, return it
    if os.path.exists(os.path.join(base_path, 'combine.csv')):
        return [base_path]
    # Otherwise find directories named trajectory_logs_*
    pattern = os.path.join(base_path, 'trajectory_logs_*')
    runs = []
    for d in glob.glob(pattern):
        if os.path.isdir(d):
            # prefer combine.csv or existence of episode_ directories
            runs.append((d, os.path.getmtime(d)))
    runs = sorted(runs, key=lambda x: x[1], reverse=True)
    return [r[0] for r in runs[:limit]]


def find_event_files_for_run(run_path):
    """Find TensorBoard event files for one run. Returns a list of event file paths (may be empty).
    Typical ML-Agents log location is `results/<run_id>/summaries` or simply `results/<run_id>`.
    If run_path is already a directory containing event files, look there.
    """
    event_files = []
    if os.path.isdir(run_path):
        for p in glob.glob(os.path.join(run_path, '**', 'events.out.tfevents.*'), recursive=True):
            event_files.append(p)
    # If run_path is a workspace, also search results/ folders
    if os.path.isdir(run_path):
        for d in glob.glob(os.path.join(run_path, 'results', '*')):
            for p in glob.glob(os.path.join(d, '**', 'events.out.tfevents.*'), recursive=True):
                event_files.append(p)
    return sorted(set(event_files))


def read_event_scalars(event_file, tags=None):
    """Read scalar series from an event file using EventAccumulator. Returns dict tag -> pandas Series indexed by step.
    If tags is None, returns all scalar tags.
    """
    if EventAccumulator is None:
        raise RuntimeError('TensorBoard EventAccumulator is not available. Install `tensorboard` package to read event files.')
    ea = EventAccumulator(event_file)
    try:
        ea.Reload()
    except Exception as e:
        print(f"Failed to reload event file {event_file}: {e}")
        return {}
    scalars = {}
    all_tags = ea.Tags().get('scalars', [])
    use_tags = tags if tags is not None else all_tags
    for tag in use_tags:
        try:
            events = ea.Scalars(tag)
            if events:
                # convert to pandas Series indexed by step (which may be global step)
                steps = [ev.step for ev in events]
                vals = [ev.value for ev in events]
                scalars[tag] = pd.Series(vals, index=steps)
        except Exception:
            continue
    return scalars


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


def compare_runs(df_a, label_a, df_b, label_b, out_dir):
    """Compare two runs, producing difference charts and summary statistics.
    df_a and df_b should contain combined episode rows for their runs.
    Uses timeHidden as a proxy for cumulative reward for hiders, episode_duration_seconds as episode length, and outcome columns as win/loss.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Ensure both dataframes have episode_index
    if 'episode_index' in df_a.columns:
        df_a = df_a.sort_values('episode_index')
    if 'episode_index' in df_b.columns:
        df_b = df_b.sort_values('episode_index')

    metrics = {
        'timeHidden': {
            'label': 'Time Hidden (fraction)',
            'higher_is_better': True,
        },
        'episode_duration_seconds': {
            'label': 'Episode duration (s)',
            'higher_is_better': False,
        }
    }

    # Helper to compute mean and sem
    def stats(arr):
        a = np.array(arr.dropna())
        if a.size == 0:
            return {'mean': np.nan, 'std': np.nan, 'count': 0}
        return {'mean': a.mean(), 'std': a.std(ddof=1), 'count': a.size}

    # Helper: compute linear trend slope (per-episode)
    def slope(arr):
        a = np.array(arr.dropna())
        n = a.size
        if n < 2:
            return np.nan
        x = np.arange(n)
        # Fit line y = m*x + b
        m, b = np.polyfit(x, a, 1)
        return m

    summary = {}
    # For each metric, compute stats and plot time series
    for key, info in metrics.items():
        if key not in df_a.columns and key not in df_b.columns:
            continue
        arr_a = pd.to_numeric(df_a[key], errors='coerce') if key in df_a.columns else pd.Series()
        arr_b = pd.to_numeric(df_b[key], errors='coerce') if key in df_b.columns else pd.Series()
        s_a = stats(arr_a)
        s_b = stats(arr_b)
        mean_diff = None
        if not np.isnan(s_a['mean']) and not np.isnan(s_b['mean']):
            mean_diff = s_b['mean'] - s_a['mean']
        slope_a = slope(arr_a) if len(arr_a) > 0 else np.nan
        slope_b = slope(arr_b) if len(arr_b) > 0 else np.nan
        summary[key] = {
            'a_mean': s_a['mean'],
            'b_mean': s_b['mean'],
            'diff': mean_diff,
            'improved': None
        }
        summary[key]['a_slope'] = slope_a
        summary[key]['b_slope'] = slope_b
        if mean_diff is not None:
            # higher_is_better determines improvement sense
            improved = (mean_diff > 0) if info['higher_is_better'] else (mean_diff < 0)
            summary[key]['improved'] = improved

        # plot
        plt.figure(figsize=(10,4))
        if len(arr_a) > 0:
            x_a = np.arange(1, len(arr_a)+1)
            plt.plot(x_a, arr_a, label=label_a, alpha=0.4)
            # overlay rolling mean
            plt.plot(x_a, arr_a.rolling(window=10, min_periods=1).mean(), label=f"{label_a} (ema)")
        if len(arr_b) > 0:
            x_b = np.arange(1, len(arr_b)+1)
            plt.plot(x_b, arr_b, label=label_b, alpha=0.4)
            plt.plot(x_b, arr_b.rolling(window=10, min_periods=1).mean(), label=f"{label_b} (ema)")
        plt.title(info['label'])
        plt.xlabel('Episode')
        plt.ylabel(info['label'])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"compare_{key}.png"))
        plt.close()
        # If possible, compute difference series (b - a) aligned by episode_index when available
        if key in df_a.columns and key in df_b.columns:
            if 'episode_index' in df_a.columns and 'episode_index' in df_b.columns:
                df_a_idx = df_a.set_index('episode_index')
                df_b_idx = df_b.set_index('episode_index')
                common_idx = df_a_idx.index.intersection(df_b_idx.index)
                if len(common_idx) > 0:
                    diff_series = pd.to_numeric(df_b_idx.loc[common_idx, key], errors='coerce') - pd.to_numeric(df_a_idx.loc[common_idx, key], errors='coerce')
                    plt.figure(figsize=(10,4))
                    plt.plot(common_idx, diff_series)
                    plt.title(f"Difference {label_b} - {label_a} ({key}) aligned by episode_index")
                    plt.xlabel('Episode Index')
                    plt.ylabel('Difference')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"compare_diff_{key}.png"))
                    plt.close()
            else:
                # fallback align by min length
                minlen = min(len(df_a[key]), len(df_b[key]))
                if minlen > 0:
                    diff_series = pd.to_numeric(df_b[key].iloc[:minlen], errors='coerce') - pd.to_numeric(df_a[key].iloc[:minlen], errors='coerce')
                    plt.figure(figsize=(10,4))
                    plt.plot(np.arange(1, len(diff_series)+1), diff_series)
                    plt.title(f"Difference {label_b} - {label_a} ({key}) aligned by position")
                    plt.xlabel('Episode (position-aligned)')
                    plt.ylabel('Difference')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"compare_diff_{key}_pos.png"))
                    plt.close()

    # Win/loss ratio
    def win_ratio(df, col_name):
        if col_name not in df.columns:
            return None
        s = df[col_name].astype(str)
        success = (s == 'Success')
        if len(success) == 0:
            return None
        return success.cumsum() / np.arange(1, len(success)+1)

    # Seekers/hiders perspective
    for perspective in ['outcome_seekers', 'outcome_hiders', 'outcome']:
        if perspective in df_a.columns or perspective in df_b.columns:
            r_a = win_ratio(df_a, perspective)
            r_b = win_ratio(df_b, perspective)
            plt.figure(figsize=(10,4))
            if r_a is not None:
                plt.plot(np.arange(1, len(r_a)+1), r_a, label=f"{label_a}")
            if r_b is not None:
                plt.plot(np.arange(1, len(r_b)+1), r_b, label=f"{label_b}")
            plt.title(f"Cumulative Success Ratio ({perspective})")
            plt.xlabel('Episode')
            plt.ylabel('Cumulative success ratio')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"compare_{perspective}.png"))
            plt.close()

    # Write summary to CSV
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.to_csv(os.path.join(out_dir, 'compare_summary.csv'))
    print(f"Saved comparison summary to {out_dir}/compare_summary.csv")


def compare_tensorboard_runs(run_a, run_b, label_a, label_b, out_dir, tags=None):
    """Compare scalar tags from TensorBoard event files for two runs (run paths)."""
    os.makedirs(out_dir, exist_ok=True)
    files_a = find_event_files_for_run(run_a)
    files_b = find_event_files_for_run(run_b)
    if not files_a:
        print(f"No TensorBoard event files found for {run_a}")
        return
    if not files_b:
        print(f"No TensorBoard event files found for {run_b}")
        return
    # Use the latest event file in each
    ea_file_a = files_a[-1]
    ea_file_b = files_b[-1]
    print(f"Reading TF events: {ea_file_a} and {ea_file_b}")
    scalars_a = read_event_scalars(ea_file_a, tags)
    scalars_b = read_event_scalars(ea_file_b, tags)
    # Find common tags
    common_tags = set(scalars_a.keys()).intersection(scalars_b.keys())
    if not common_tags:
        print("No common scalar tags found between the two runs. Available tags for run A:", list(scalars_a.keys()))
        print("run B tags:", list(scalars_b.keys()))
        return
    # For each common tag, plot and compute summary stats
    summary = {}
    for tag in sorted(common_tags):
        s_a = scalars_a[tag]
        s_b = scalars_b[tag]
        # align by step index (inner join)
        df_a = pd.DataFrame({'value': s_a}).reset_index().rename(columns={'index':'step'})
        df_b = pd.DataFrame({'value': s_b}).reset_index().rename(columns={'index':'step'})
        merged = pd.merge(df_a, df_b, on='step', how='inner', suffixes=('_a','_b'))
        # fallback if no overlap: align by position
        if merged.empty:
            minlen = min(len(df_a), len(df_b))
            merged = pd.DataFrame({'value_a':df_a['value'].iloc[:minlen].values, 'value_b':df_b['value'].iloc[:minlen].values})
        merged['diff'] = merged['value_b'] - merged['value_a']
        summary[tag] = {'a_mean': merged['value_a'].mean(), 'b_mean': merged['value_b'].mean(), 'diff': merged['diff'].mean(), 'count': len(merged)}
        # plot
        plt.figure(figsize=(10,4))
        plt.plot(merged['value_a'], label=label_a)
        plt.plot(merged['value_b'], label=label_b)
        plt.title(tag)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tb_compare_{tag.replace('/', '_')}.png"))
        plt.close()
        # diff plot
        plt.figure(figsize=(10,4))
        plt.plot(merged['diff'])
        plt.title(f"Diff {tag} ({label_b} - {label_a})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tb_compare_diff_{tag.replace('/', '_')}.png"))
        plt.close()
    pd.DataFrame.from_dict(summary, orient='index').to_csv(os.path.join(out_dir, 'tb_compare_summary.csv'))
    print(f"Saved TF event comparison summary to {out_dir}/tb_compare_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-path', default='.', help='Path to run directory or workspace root (searches for trajectory_logs_*)')
    parser.add_argument('--out', default='analysis_output', help='Output folder for summary and charts')
    parser.add_argument('--runs', nargs=2, metavar=('RUN_A','RUN_B'), help='Paths to two runs to compare. If omitted, use --auto-compare or single run analysis')
    parser.add_argument('--auto-compare', action='store_true', help='If specified and --run-path points to workspace root, automatically pick the latest two runs for comparison')
    parser.add_argument('--include-tensorboard', dest='include_tb', action='store_true', help='Include TensorBoard event (tfevents) analysis and comparison (requires tensorboard package)')
    parser.add_argument('--tb-tags', nargs='*', help='Optional list of scalar tags to load from TensorBoard event files (default: load all tags)')
    args = parser.parse_args()

    # If two runs specified explicitly, compare them
    if args.runs:
        run_a = args.runs[0]
        run_b = args.runs[1]
        df_a = load_run_df(run_a)
        df_b = load_run_df(run_b)
        if df_a.empty:
            print(f"No metadata found for run: {run_a}")
            return
        if df_b.empty:
            print(f"No metadata found for run: {run_b}")
            return
        print(f"Comparing runs: {run_a} and {run_b}")
        # Compare CSV/meta first
        compare_runs(df_a, os.path.basename(run_a), df_b, os.path.basename(run_b), args.out)
        # Then optionally compare TF event scalars
        if args.include_tb:
            try:
                compare_tensorboard_runs(run_a, run_b, os.path.basename(run_a), os.path.basename(run_b), args.out, tags=args.tb_tags)
            except Exception as e:
                print(f"TensorBoard comparison failed: {e}")
        return
    if args.auto_compare:
        latest = find_latest_runs(args.run_path, limit=2)
        if len(latest) < 2:
            print("Not enough runs to auto-compare (need at least 2)")
            return
        run_a, run_b = latest[1], latest[0]
        df_a = load_run_df(run_a)
        df_b = load_run_df(run_b)
        print(f"Auto comparing runs: {run_a} (previous) vs {run_b} (latest)")
        compare_runs(df_a, os.path.basename(run_a), df_b, os.path.basename(run_b), args.out)
        if args.include_tb:
            try:
                compare_tensorboard_runs(run_a, run_b, os.path.basename(run_a), os.path.basename(run_b), args.out, tags=args.tb_tags)
            except Exception as e:
                print(f"TensorBoard comparison failed: {e}")
        return

    # if user specified to include TF event analysis, attempt to find and compare tf events
    if args.runs and args.include_tb:
        # run-level TF compare for explicit run paths
        run_a, run_b = args.runs
        compare_tensorboard_runs(run_a, run_b, os.path.basename(run_a), os.path.basename(run_b), args.out)
        # continue to CSV compare if present
    if args.auto_compare and args.include_tb:
        latest = find_latest_runs(args.run_path, limit=2)
        if len(latest) >= 2:
            run_a, run_b = latest[1], latest[0]
            compare_tensorboard_runs(run_a, run_b, os.path.basename(run_a), os.path.basename(run_b), args.out)

    files = find_episode_meta_files(args.run_path)
    print(f"Found {len(files)} episode_meta.csv files or combine files")
    df = read_metadata(files)
    if df.empty:
        print("No episode metadata rows loaded. Are you pointing to the right folder?")
        return

    summarize_and_plot(df, args.out)


if __name__ == '__main__':
    main()

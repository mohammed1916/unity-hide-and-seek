import os
import glob
import pandas as pd


def find_episode_meta_files(run_path):
    """Return a sorted list of episode_meta or combine CSVs for a given run_path."""
    results = []
    combine = os.path.join(run_path, 'combine.csv')
    if os.path.exists(combine):
        return [combine]

    if os.path.isdir(run_path):
        for d in glob.glob(os.path.join(run_path, 'trajectory_logs_*')):
            c = os.path.join(d, 'combine.csv')
            if os.path.exists(c):
                results.append(c)
        if results:
            return sorted(results)

    # look for per-episode meta files
    if os.path.exists(os.path.join(run_path, 'episode_0')) or os.path.exists(run_path):
        search_pattern = os.path.join(run_path, 'episode_*', 'episode_meta.csv')
    else:
        search_pattern = os.path.join(run_path, 'trajectory_logs_*', 'episode_*', 'episode_meta.csv')
    for path in glob.glob(search_pattern):
        results.append(path)
    return sorted(results)


def read_metadata(files):
    """Read a list of episode_meta CSV files and return concatenated DataFrame."""
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                # If file contains multiple rows, extend; otherwise append single row
                for _, r in df.iterrows():
                    rows.append(r.to_dict())
        except Exception:
            continue
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def load_run_df(run_path):
    """Load a single run's metadata as a DataFrame, preferring combine.csv if present."""
    candidate_files = find_episode_meta_files(run_path)
    if not candidate_files:
        return pd.DataFrame()
    if len(candidate_files) == 1 and os.path.basename(candidate_files[0]) == 'combine.csv':
        try:
            df = pd.read_csv(candidate_files[0])
            df['run_path'] = run_path
            return df
        except Exception:
            return pd.DataFrame()
    return read_metadata(candidate_files)


def find_latest_runs(base_path='.', limit=2):
    pattern = os.path.join(base_path, 'trajectory_logs_*')
    runs = []
    for d in glob.glob(pattern):
        if os.path.isdir(d):
            runs.append((d, os.path.getmtime(d)))
    runs = sorted(runs, key=lambda x: x[1], reverse=True)
    return [r[0] for r in runs[:limit]]

"""analysis_lib

Expose a small set of functions for loading TensorBoard scalars and
episode metadata and producing quick summaries/plots.
"""
from .io import find_episode_meta_files, read_metadata, load_run_df, find_latest_runs
from .tensorboard import load_tensorboard_logs, extract_scalars, read_event_scalars, find_event_files_for_run
from .analyze import summarize_and_plot, compare_runs, compare_tensorboard_runs

__all__ = [
    'find_episode_meta_files',
    'read_metadata',
    'load_run_df',
    'find_latest_runs',
    'load_tensorboard_logs',
    'extract_scalars',
    'read_event_scalars',
    'find_event_files_for_run',
    'summarize_and_plot',
    'compare_runs',
    'compare_tensorboard_runs',
]

import os
import glob
import pandas as pd
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


def find_event_files_for_run(run_path):
    """Find TensorBoard event files for a run (recursive)."""
    event_files = []
    if os.path.isdir(run_path):
        for p in glob.glob(os.path.join(run_path, '**', 'events.out.tfevents.*'), recursive=True):
            event_files.append(p)
    if os.path.isdir(run_path):
        for d in glob.glob(os.path.join(run_path, 'results', '*')):
            for p in glob.glob(os.path.join(d, '**', 'events.out.tfevents.*'), recursive=True):
                event_files.append(p)
    return sorted(set(event_files))


def load_tensorboard_logs(log_dir):
    """Return an EventAccumulator for the latest event file in `log_dir`, or None."""
    if EventAccumulator is None:
        raise RuntimeError('TensorBoard EventAccumulator not available. Install tensorboard.')
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None
    event_file = max(event_files, key=os.path.getctime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    return ea


def extract_scalars(ea, tag):
    """Return list of (step, value) for a scalar tag from EventAccumulator."""
    if ea is None:
        return []
    tags = ea.Tags().get('scalars', [])
    if tag not in tags:
        return []
    return [(s.step, s.value) for s in ea.Scalars(tag)]


def read_event_scalars(event_file, tags=None):
    """Read scalar series from an event file; return dict tag->pandas.Series indexed by step."""
    if EventAccumulator is None:
        raise RuntimeError('TensorBoard EventAccumulator not available. Install tensorboard.')
    ea = EventAccumulator(event_file)
    try:
        ea.Reload()
    except Exception:
        return {}
    scalars = {}
    all_tags = ea.Tags().get('scalars', [])
    use_tags = tags if tags is not None else all_tags
    for tag in use_tags:
        try:
            events = ea.Scalars(tag)
            if events:
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                scalars[tag] = pd.Series(vals, index=steps)
        except Exception:
            continue
    return scalars

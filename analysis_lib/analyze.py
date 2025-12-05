import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_and_plot(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if df.empty:
        print('Empty dataframe; nothing to summarize')
        return
    # ensure some columns
    df = df.copy()
    if 'episode_index' in df.columns:
        df['episode_index'] = pd.to_numeric(df['episode_index'], errors='coerce')
        df = df.sort_values('episode_index')

    # Save CSV
    out_csv = os.path.join(out_dir, 'episode_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f'Saved summary CSV: {out_csv}')

    # Basic plots if columns exist
    if 'timeHidden' in df.columns and 'episode_index' in df.columns:
        plt.figure(figsize=(8,3))
        plt.plot(df['episode_index'], pd.to_numeric(df['timeHidden'], errors='coerce'))
        plt.title('TimeHidden per episode')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'timeHidden.png'))
        plt.close()

    if 'episode_duration_seconds' in df.columns:
        plt.figure(figsize=(6,3))
        pd.to_numeric(df['episode_duration_seconds'], errors='coerce').hist(bins=30)
        plt.title('Episode duration (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'episode_duration_hist.png'))
        plt.close()


def compare_runs(df_a, label_a, df_b, label_b, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for key in ['timeHidden', 'episode_duration_seconds']:
        a = pd.to_numeric(df_a.get(key, pd.Series()), errors='coerce')
        b = pd.to_numeric(df_b.get(key, pd.Series()), errors='coerce')
        summary[key] = {
            'a_mean': float(a.mean()) if not a.empty else None,
            'b_mean': float(b.mean()) if not b.empty else None,
            'diff': float(b.mean() - a.mean()) if (not a.empty and not b.empty) else None,
        }
        # quick plot
        plt.figure(figsize=(8,3))
        if not a.empty:
            plt.plot(a.reset_index(drop=True), label=label_a, alpha=0.5)
        if not b.empty:
            plt.plot(b.reset_index(drop=True), label=label_b, alpha=0.5)
        plt.legend()
        plt.title(key)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'compare_{key}.png'))
        plt.close()

    pd.DataFrame.from_dict(summary, orient='index').to_csv(os.path.join(out_dir, 'compare_summary.csv'))
    print(f'Saved comparison summary to {out_dir}/compare_summary.csv')


def compare_tensorboard_runs(run_a, run_b, label_a, label_b, out_dir, tags=None, tb_reader=None):
    """Compare tensorboard scalars for two runs. tb_reader should provide find_event_files_for_run and read_event_scalars.
    If not provided, caller can import analysis_lib.tensorboard and pass it.
    """
    os.makedirs(out_dir, exist_ok=True)
    if tb_reader is None:
        raise RuntimeError('Pass a tb_reader (analysis_lib.tensorboard) to read TB events')
    files_a = tb_reader.find_event_files_for_run(run_a)
    files_b = tb_reader.find_event_files_for_run(run_b)
    if not files_a or not files_b:
        print('No event files found for one of the runs')
        return
    scalars_a = tb_reader.read_event_scalars(files_a[-1], tags)
    scalars_b = tb_reader.read_event_scalars(files_b[-1], tags)
    common = set(scalars_a.keys()).intersection(scalars_b.keys())
    if not common:
        print('No common scalar tags between runs')
        return
    summary = {}
    for tag in sorted(common):
        s_a = scalars_a[tag]
        s_b = scalars_b[tag]
        # align by index (step)
        df_a = s_a.reset_index().rename(columns={'index':'step', 0:'value_a'}) if hasattr(s_a, 'reset_index') else pd.DataFrame()
        try:
            merged = pd.merge(pd.DataFrame({'step':s_a.index, 'value_a':s_a.values}), pd.DataFrame({'step':s_b.index, 'value_b':s_b.values}), on='step', how='inner')
        except Exception:
            merged = pd.DataFrame()
        if merged.empty:
            # fallback align by position
            minlen = min(len(s_a), len(s_b))
            merged = pd.DataFrame({'value_a': s_a.values[:minlen], 'value_b': s_b.values[:minlen]})
        merged['diff'] = merged['value_b'] - merged['value_a']
        summary[tag] = {'a_mean': float(merged['value_a'].mean()), 'b_mean': float(merged['value_b'].mean()), 'diff': float(merged['diff'].mean())}
        # plot
        plt.figure(figsize=(8,3))
        plt.plot(merged['value_a'], label=label_a)
        plt.plot(merged['value_b'], label=label_b)
        plt.title(tag)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'tb_{tag.replace("/","_")}.png'))
        plt.close()

    pd.DataFrame.from_dict(summary, orient='index').to_csv(os.path.join(out_dir, 'tb_compare_summary.csv'))
    print(f'Saved TF compare summary to {out_dir}/tb_compare_summary.csv')

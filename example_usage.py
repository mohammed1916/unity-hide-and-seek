"""Example usage for analysis_lib

Run small examples after installing requirements:

    pip install -r requirements.txt

Then run examples below or import the library in your scripts.
"""
from analysis_lib import io as aio, tensorboard as atb, analyze as aan
from analysis_lib import viz as aviz


def example_summarize(run_path, out_dir='analysis_output'):
    files = aio.find_episode_meta_files(run_path)
    print('Found files:', files)
    df = aio.read_metadata(files)
    aan.summarize_and_plot(df, out_dir)


def example_compare_runs(run_a, run_b, out_dir='compare_output'):
    df_a = aio.load_run_df(run_a)
    df_b = aio.load_run_df(run_b)
    aan.compare_runs(df_a, run_a, df_b, run_b, out_dir)


def example_compare_tb(run_a, run_b, out_dir='tb_compare', tags=None):
    aan.compare_tensorboard_runs(run_a, run_b, run_a, run_b, out_dir, tags=tags, tb_reader=atb)


def example_3d_demo(out_dir='out_3d'):
    """Generate sample trajectories and produce 3D visualizations."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = aviz.make_sample_trajectory(num_agents=4, steps=120)
    p1 = aviz.plot_3d_trajectories(df, os.path.join(out_dir, 'trajectories.html'))
    p2 = aviz.plot_3d_scatter(df, os.path.join(out_dir, 'scatter.html'), color_by='step')
    p3 = aviz.animate_3d(df, os.path.join(out_dir, 'animate.html'), frame_col='step')
    print('Generated 3D outputs:', p1, p2, p3)


if __name__ == '__main__':
    print('Module example. Call functions programmatically.')

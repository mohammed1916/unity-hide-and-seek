"""Run 3D visualization demo and/or apply to an existing positions CSV.

The script will:
- generate sample trajectories and produce HTML (if no positions CSV provided)
- or load `positions.csv` if present and visualize that
"""
import os
from example_usage import example_3d_demo

POSITIONS_CSV = 'positions.csv'  # optional: expected columns agent,step,x,y,z

if __name__ == '__main__':
    if os.path.exists(POSITIONS_CSV):
        import pandas as pd
        from analysis_lib import viz
        df = pd.read_csv(POSITIONS_CSV)
        os.makedirs('out_3d', exist_ok=True)
        viz.plot_3d_trajectories(df, 'out_3d/trajectories_positions.html')
        viz.plot_3d_scatter(df, 'out_3d/scatter_positions.html', color_by='step')
        try:
            viz.animate_3d(df, 'out_3d/animate_positions.html', frame_col='step')
        except Exception as e:
            print('Animate failed:', e)
    else:
        example_3d_demo(out_dir='out_3d')

"""3D visualization helpers using Plotly.

Functions expect a DataFrame with columns for agent id and coordinates (x,y,z)
and an optional time/step column. Outputs are interactive HTML files.
"""
import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_3d_trajectories(df, out_html, agent_col='agent', x='x', y='y', z='z', time_col=None):
    """Plot 3D trajectories per agent and save as interactive HTML.

    df: DataFrame with at least columns [agent_col, x, y, z].
    If time_col provided, points are ordered by it.
    """
    os.makedirs(os.path.dirname(out_html) or '.', exist_ok=True)
    df = df.copy()
    if time_col is not None and time_col in df.columns:
        df = df.sort_values(time_col)

    fig = go.Figure()
    agents = sorted(df[agent_col].unique())
    for a in agents:
        sub = df[df[agent_col] == a]
        fig.add_trace(go.Scatter3d(
            x=sub[x], y=sub[y], z=sub[z],
            mode='lines+markers', name=str(a),
            marker=dict(size=3), line=dict(width=2)
        ))

    fig.update_layout(scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z), margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_html)
    return out_html


def plot_3d_scatter(df, out_html, color_by=None, size=3, x='x', y='y', z='z'):
    """3D scatter plot colored by a column (categorical or numeric)."""
    os.makedirs(os.path.dirname(out_html) or '.', exist_ok=True)
    df = df.copy()
    if color_by is None:
        colors = None
    else:
        colors = df[color_by]

    fig = go.Figure(data=[go.Scatter3d(
        x=df[x], y=df[y], z=df[z], mode='markers',
        marker=dict(size=size, color=colors, colorscale='Viridis', showscale=True)
    )])
    fig.update_layout(scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z), margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_html)
    return out_html


def animate_3d(df, out_html, agent_col='agent', x='x', y='y', z='z', frame_col='step'):
    """Create an animated 3D scatter with frames defined by `frame_col`.

    Each frame shows agent positions at that frame. Saves an HTML with play controls.
    """
    os.makedirs(os.path.dirname(out_html) or '.', exist_ok=True)
    df = df.copy()
    if frame_col not in df.columns:
        raise ValueError(f"Frame column '{frame_col}' not present in dataframe")

    frames = []
    frame_vals = sorted(df[frame_col].unique())
    agents = sorted(df[agent_col].unique())

    # Build initial data traces (one per agent)
    data_traces = []
    for a in agents:
        sub0 = df[df[agent_col] == a]
        sub0 = sub0[sub0[frame_col] == frame_vals[0]]
        data_traces.append(go.Scatter3d(x=sub0[x], y=sub0[y], z=sub0[z], mode='markers', name=str(a), marker=dict(size=4)))

    for fv in frame_vals:
        frame_data = []
        sdf = df[df[frame_col] == fv]
        for a in agents:
            sub = sdf[sdf[agent_col] == a]
            frame_data.append(go.Scatter3d(x=sub[x], y=sub[y], z=sub[z], mode='markers', marker=dict(size=4)))
        frames.append(go.Frame(data=frame_data, name=str(fv)))

    fig = go.Figure(data=data_traces, frames=frames)
    # animation settings
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'mode': 'immediate', 'frame': {'duration': 0}}]}
            ]
        }]
    )
    fig.update_layout(scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z), margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_html)
    return out_html


def make_sample_trajectory(num_agents=3, steps=100):
    """Return a sample DataFrame with random walk trajectories for quick demos."""
    rows = []
    for a in range(num_agents):
        pos = np.array([0.0, 0.0, 0.0]) + np.random.randn(3) * 0.1
        for s in range(steps):
            pos = pos + np.random.randn(3) * 0.2
            rows.append({'agent': f'A{a}', 'step': s, 'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])})
    return pd.DataFrame(rows)

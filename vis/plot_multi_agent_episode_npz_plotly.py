"""
Plot all Unity agent NPZ files for a single episode in 3D using Plotly with a time slider animation.

Usage:
    python vis/plot_multi_agent_episode_npz_plotly.py -d trajectory_logs/ -e 0 -o out.html --show

Requirements:
    pip install numpy plotly

What it does:
    - Finds all files matching 'episode_<ep>_agent_*.npz' in the given directory.
    - Loads each agent’s positions array (shape (1, T, 3)) and stacks them into (num_agents, T, 3).
    - Draws animated 3D trajectories for all agents with color-coded trails and markers.
    - Saves an interactive HTML visualization (optionally opens in browser).
"""

import argparse
import os
import re
import webbrowser
from typing import Tuple, List

import numpy as np
import plotly.graph_objects as go


DEFAULT_COLOR_TUPLES = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def rgb(tup): return f"rgb({tup[0]},{tup[1]},{tup[2]})"
def rgba(tup, a): return f"rgba({tup[0]},{tup[1]},{tup[2]},{a})"


def load_episode_agents(dir_path: str, episode_idx: int) -> Tuple[np.ndarray, List[str]]:
    """
    Load all agent NPZs for a given episode number and stack positions into (N, T, 3).
    Compatible with filenames like:
        episode_3_agent_agent_0-3_tm0.npz
    """
    # Match more flexible patterns like "episode_3_agent_agent_0-3_tm0.npz"
    pattern = re.compile(rf"episode_{episode_idx}_agent_.*\.npz$")
    files = sorted([f for f in os.listdir(dir_path) if pattern.match(f)])

    if not files:
        raise FileNotFoundError(f"No agent NPZ files found for episode {episode_idx} in {dir_path}")

    all_positions = []
    agent_names = []

    for f in files:
        path = os.path.join(dir_path, f)
        data = np.load(path, allow_pickle=True)
        if "positions" not in data:
            raise ValueError(f"'positions' key not found in {f}")

        pos = np.asarray(data["positions"])
        if pos.ndim == 3 and pos.shape[-1] == 3:
            pos = pos[0]  # (1, T, 3)
        elif pos.ndim == 2 and pos.shape[-1] == 3:
            pass
        else:
            raise ValueError(f"Unexpected positions shape {pos.shape} in {f}")

        all_positions.append(pos)

        # Clean up agent name from file (remove prefix/suffix)
        base = os.path.splitext(f)[0]
        agent_id = base.replace(f"episode_{episode_idx}_agent_", "")
        agent_names.append(agent_id)

    # Ensure equal time dimension for all
    max_T = max(p.shape[0] for p in all_positions)
    aligned_positions = []
    for p in all_positions:
        if p.shape[0] < max_T:
            pad = np.repeat(p[-1][None, :], max_T - p.shape[0], axis=0)
            p = np.vstack([p, pad])
        aligned_positions.append(p)

    positions = np.stack(aligned_positions, axis=0)
    return positions, agent_names



def make_3d_animation(positions: np.ndarray, agent_names=None, title="Episode") -> go.Figure:
    num_agents, T, _ = positions.shape
    if agent_names is None:
        agent_names = [f"agent_{i}" for i in range(num_agents)]

    static_traces = []
    for i in range(num_agents):
        xs, ys, zs = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]
        col = DEFAULT_COLOR_TUPLES[i % len(DEFAULT_COLOR_TUPLES)]
        static_traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=2, color=rgba(col, 0.25)),
                name=f"{agent_names[i]}_traj",
                hoverinfo="name",
                showlegend=False,
            )
        )

    moving_markers = []
    trailing_traces = []
    for i in range(num_agents):
        col = DEFAULT_COLOR_TUPLES[i % len(DEFAULT_COLOR_TUPLES)]
        moving_markers.append(
            go.Scatter3d(
                x=[positions[i, 0, 0]], y=[positions[i, 0, 1]], z=[positions[i, 0, 2]],
                mode="markers",
                marker=dict(size=8, color=rgb(col)),
                name=agent_names[i],
                text=["t=0"],
                showlegend=True,
            )
        )
        trailing_traces.append(
            go.Scatter3d(
                x=[positions[i, 0, 0]], y=[positions[i, 0, 1]], z=[positions[i, 0, 2]],
                mode="lines+markers",
                line=dict(width=4, color=rgb(col)),
                marker=dict(size=3, color=rgb(col)),
                showlegend=False,
            )
        )

    frames = []
    for t in range(T):
        frame_data = []
        for i in range(num_agents):
            col = DEFAULT_COLOR_TUPLES[i % len(DEFAULT_COLOR_TUPLES)]
            frame_data.append(
                go.Scatter3d(
                    x=[positions[i, t, 0]], y=[positions[i, t, 1]], z=[positions[i, t, 2]],
                    mode="markers",
                    marker=dict(size=8, color=rgb(col)),
                    name=agent_names[i],
                    text=[f"t={t}"],
                )
            )
            frame_data.append(
                go.Scatter3d(
                    x=positions[i, :t + 1, 0], y=positions[i, :t + 1, 1], z=positions[i, :t + 1, 2],
                    mode="lines+markers",
                    line=dict(width=4, color=rgb(col)),
                    marker=dict(size=3, color=rgb(col)),
                    showlegend=False,
                )
            )
        frames.append(go.Frame(data=frame_data, name=str(t)))

    sliders = [{
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {"args": [[str(k)], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
             "label": str(k), "method": "animate"} for k in range(T)
        ],
    }]

    updatemenus = [{
        "type": "buttons",
        "showactive": False,
        "y": 0.05,
        "x": 0.0,
        "xanchor": "left",
        "yanchor": "bottom",
        "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
        ],
    }]

    fig = go.Figure(data=static_traces + moving_markers + trailing_traces, frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="auto"),
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=True,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot multi-agent NPZ episode (Plotly)")
    parser.add_argument("-d", "--dir", required=True, help="Directory containing NPZ logs")
    parser.add_argument("-e", "--episode", type=int, required=True, help="Episode number to visualize")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path")
    parser.add_argument("--show", action="store_true", help="Open HTML in browser after saving")
    args = parser.parse_args()

    positions, agent_names = load_episode_agents(args.dir, args.episode)
    title = f"Episode {args.episode} ({len(agent_names)} agents)"
    fig = make_3d_animation(positions, agent_names, title)

    out_path = args.output or os.path.join(args.dir, f"episode_{args.episode}_plot.html")
    fig.write_html(out_path, auto_open=False)
    print(f"Saved 3D animation: {out_path}")

    if args.show:
        webbrowser.open("file://" + os.path.abspath(out_path))


if __name__ == "__main__":
    main()

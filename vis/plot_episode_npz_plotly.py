"""
Plot a single Unity NPZ episode in 3D using Plotly with a time slider animation.

Usage:
    python vis/plot_episode_npz_plotly.py -i path/to/episode_0.npz -o out.html --show

Requirements:
    pip install numpy plotly

What it does:
    - Loads an .npz file and heuristically finds position arrays (expects last dim == 3).
    - Supports shapes: (num_agents, T, 3) or (T, 3) (single agent). Also inspects keys.
    - Draws full-trajectory lines for each agent and an animated moving marker per agent.
    - Saves an interactive HTML file (and optionally opens it in the default browser).

Notes:
    - If your NPZ layout differs, open the file in Python and pass an array with shape
      (num_agents, T, 3) under a named key like 'positions' for best results.
"""

from __future__ import annotations
import argparse
import os
import webbrowser
from typing import Tuple

import numpy as np
import plotly.graph_objects as go


def find_positions_in_npz(npz: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, str]:
    """
    Try to find a positions array in the loaded npz file.
    Returns array shaped (num_agents, T, 3) and the key name found.
    """
    # Prefer explicit key
    if "positions" in npz:
        arr = npz["positions"]
        return normalize_positions(arr), "positions"

    # Search for any array with last dim == 3
    for key in npz.files:
        arr = npz[key]
        if isinstance(arr, np.ndarray) and arr.size > 0:
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return normalize_positions(arr), key
            if arr.ndim == 2 and arr.shape[-1] == 3:
                # (T,3) -> single agent
                return normalize_positions(arr), key
    # As fallback: look for flattened or object arrays that can be reshaped to (-1, 8) etc.
    raise ValueError(f"No position-like array (last dim == 3) found in NPZ. Keys: {npz.files}")


def normalize_positions(arr: np.ndarray) -> np.ndarray:
    """
    Normalize to shape (num_agents, T, 3).
    Accepts:
      - (num_agents, T, 3) -> unchanged
      - (T, 3) -> -> (1, T, 3)
      - other shapes -> attempt conversion
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr[np.newaxis, ...]
    # If arr is 1D or other, try to reshape if possible
    if arr.ndim == 1 and arr.size % 3 == 0:
        T = arr.size // 3
        return arr.reshape(1, T, 3)
    # Otherwise cannot normalize
    raise ValueError(f"Cannot normalize positions array of shape {arr.shape} to (N,T,3)")


def make_3d_animation(positions: np.ndarray, agent_names=None, title: str = "Episode") -> go.Figure:
    """
    Build a Plotly Figure with static trajectory lines and animated markers across time.
    positions: (num_agents, T, 3)
    """
    num_agents, T, _ = positions.shape
    if agent_names is None:
        agent_names = [f"agent_{i}" for i in range(num_agents)]

    # Static traces: full trajectories for each agent (lines)
    static_traces = []
    for i in range(num_agents):
        xs = positions[i, :, 0]
        ys = positions[i, :, 1]
        zs = positions[i, :, 2]
        static_traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=2),
                name=f"{agent_names[i]}_traj",
                hoverinfo="name",
            )
        )

    # Initial marker traces (at t=0)
    init_markers = []
    for i in range(num_agents):
        init_markers.append(
            go.Scatter3d(
                x=[positions[i, 0, 0]],
                y=[positions[i, 0, 1]],
                z=[positions[i, 0, 2]],
                mode="markers",
                marker=dict(size=6),
                name=f"{agent_names[i]}",
                hoverinfo="name+text",
                text=[f"t=0"],
            )
        )

    # Build frames: one frame per time step with marker positions
    frames = []
    for t in range(T):
        data = []
        # Keep trajectory lines as static in layout; frames only need marker positions
        for i in range(num_agents):
            data.append(
                go.Scatter3d(
                    x=[positions[i, t, 0]],
                    y=[positions[i, t, 1]],
                    z=[positions[i, t, 2]],
                    mode="markers",
                    marker=dict(size=6),
                    name=f"{agent_names[i]}",
                    hoverinfo="name+text",
                    text=[f"t={t}"],
                )
            )
        frames.append(go.Frame(data=data, name=str(t)))

    # Compose the figure with static traces + initial markers
    fig = go.Figure(data=static_traces + init_markers, frames=frames)

    # Slider and play button
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[str(k)], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                    "label": str(k),
                    "method": "animate",
                }
                for k in range(T)
            ],
        }
    ]

    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "y": 0.05,
            "x": 0.0,
            "xanchor": "left",
            "yanchor": "bottom",
            "pad": {"t": 45, "r": 10},
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                },
            ],
        }
    ]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="auto",
        ),
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=True,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot NPZ episode in 3D (Plotly)")
    parser.add_argument("-i", "--input", required=True, help="Path to episode .npz file")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path (defaults to same name .html)")
    parser.add_argument("--show", action="store_true", help="Open the saved HTML in a browser")
    args = parser.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    npz = np.load(in_path, allow_pickle=True)
    try:
        positions, key = find_positions_in_npz(npz)
    except Exception as e:
        raise RuntimeError(f"Failed to locate positions in NPZ: {e}")

    title = f"Episode visualization: {os.path.basename(in_path)} (key={key})"
    fig = make_3d_animation(positions, title=title)

    if args.output is None:
        out_path = os.path.splitext(in_path)[0] + ".html"
    else:
        out_path = args.output
    fig.write_html(out_path, auto_open=False)
    print(f"Saved interactive plot to: {out_path}")
    if args.show:
        webbrowser.open("file://" + os.path.abspath(out_path))


if __name__ == "__main__":
    main()

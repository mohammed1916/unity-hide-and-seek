"""
python vis/load_unity_trajectories_npz.py -i trajectory_logs -o vis_output --dashboard --visualize-episode 0
"""

import argparse
import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

# Optional imports from your visualization utilities
try:
    from generate_and_visualize import (
        visualize_episode_with_metrics,
        plot_metrics_dashboard,
        enrich_with_ppo_metrics,
        save_metrics,
    )
except Exception as e:
    visualize_episode_with_metrics = None
    plot_metrics_dashboard = None
    enrich_with_ppo_metrics = None
    save_metrics = None


def load_npz_episodes(folder):
    """
    Loads all episode_*.npz files saved by AgentTrajectoryLogger.

    Returns:
        episodes (list): Each entry is a dict with:
            - 'positions': np.ndarray of shape (num_agents, T, 3)
            - 'goals': np.ndarray of shape (num_agents, 3)
            - 'episode_index': int (parsed from filename)
    """
    files = sorted(glob(os.path.join(folder, "episode_*.npz")))
    episodes = []

    if not files:
        print(f"No episode_*.npz files found in {folder}")
        return episodes

    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            ep_idx = int(os.path.basename(f).split("_")[-1].split(".")[0])
            episodes.append({
                "positions": data["positions"],
                "goals": data["goals"],
                "episode_index": ep_idx,
            })
            data.close()
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    return episodes


def save_outputs(out_dir, positions, velocities, ep_df, agent_df):
    os.makedirs(out_dir, exist_ok=True)
    # Positions and velocities may be variable-shaped per episode (num_agents and T may differ).
    # Save as pickled lists to preserve ragged shapes.
    np.save(os.path.join(out_dir, 'positions_list.npy'), positions, allow_pickle=True)
    np.save(os.path.join(out_dir, 'velocities_list.npy'), velocities, allow_pickle=True)
    ep_csv = os.path.join(out_dir, 'episode_metrics.csv')
    agent_csv = os.path.join(out_dir, 'agent_metrics.csv')
    ep_df.to_csv(ep_csv, index=False)
    agent_df.to_csv(agent_csv, index=False)
    print(f"Saved positions/velocities and CSV metrics to {out_dir}")


def flatten_to_dataframe(episodes):
    """
    Convert loaded NPZ episodes into a flat DataFrame with per-agent positions over time.
    Columns: ['episode', 'agent', 't', 'x', 'y', 'z']
    """
    rows = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        pos = ep["positions"]  # (num_agents, T, 3)
        num_agents, T, _ = pos.shape
        for a in range(num_agents):
            for t in range(T):
                x, y, z = pos[a, t]
                rows.append({
                    "episode": ep_idx,
                    "agent": a,
                    "t": t,
                    "x": x,
                    "y": y,
                    "z": z,
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description='Load Unity NPZ trajectory logs and convert them for visualization')
    p.add_argument('--input-dir', '-i', required=True, help='Directory containing episode_*.npz files')
    p.add_argument('--out-dir', '-o', default='vis_output', help='Output directory for numpy arrays and CSVs')
    p.add_argument('--dt', type=float, default=0.02, help='Time delta between FixedUpdate steps (for velocity computation)')
    p.add_argument('--visualize-episode', type=int, default=None, help='Generate HTML visualization for this episode index')
    p.add_argument('--dashboard', action='store_true', help='Build a metrics dashboard HTML')
    args = p.parse_args()

    # Load trajectory data
    episodes = load_npz_episodes(args.input_dir)
    if not episodes:
        print('No .npz episodes found in', args.input_dir)
        return

    # We allow variable numbers of agents and timesteps per episode.
    # Don't attempt to stack; instead keep per-episode arrays in lists.
    shapes = [ep["positions"].shape for ep in episodes]
    print(f"Shape of loaded episodes: {shapes}")
    positions_list = [ep["positions"] for ep in episodes]
    goals_list = [ep["goals"] for ep in episodes]

    # Compute per-episode velocities (difference along the time axis)
    velocities_list = []
    for pos in positions_list:
        # pos: (num_agents, T, 3); compute difference along axis=1 (time axis)
        if pos.ndim != 3:
            # If shape unexpected, produce an empty velocities array with same shape
            velocities_list.append(np.zeros_like(pos))
            continue
        # Prepend first step to keep velocity array same length as pos
        first = pos[:, :1, :]
        diffs = np.diff(pos, axis=1)
        vel = np.concatenate([first, diffs], axis=1)
        velocities_list.append(vel)

    # Build simple episode- and agent-level metrics
    ep_df = pd.DataFrame({
        "episode": [ep["episode_index"] for ep in episodes],
        "num_agents": [ep["positions"].shape[0] for ep in episodes],
        "steps": [ep["positions"].shape[1] for ep in episodes],
        "timestamp": [datetime.now().isoformat()] * len(episodes),
    })

    agent_rows = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        num_agents = ep["positions"].shape[0]
        for a_idx in range(num_agents):
            agent_rows.append({
                "episode": ep_idx,
                "agent": a_idx,
                "steps": ep["positions"].shape[1],
            })
    agent_df = pd.DataFrame(agent_rows)

    # Save data and CSVs
    save_outputs(args.out_dir, positions_list, velocities_list, ep_df, agent_df)

    # Also save a flattened trajectory CSV for external tools (e.g. Plotly 3D)
    traj_df = flatten_to_dataframe(episodes)
    traj_csv = os.path.join(args.out_dir, 'trajectories_flat.csv')
    traj_df.to_csv(traj_csv, index=False)
    print(f"Saved flattened trajectory data to {traj_csv}")

    # Optional: generate PPO-enriched dashboard
    if args.dashboard:
        if enrich_with_ppo_metrics is None or plot_metrics_dashboard is None:
            print('Dashboard functions not available; ensure vis/generate_and_visualize.py is importable')
        else:
            # Pass per-episode lists to the enrichment function
            ep_df2, agent_df2, sample_eff = enrich_with_ppo_metrics(
                ep_df, agent_df, positions_list, goals=goals_list
            )
            plot_metrics_dashboard(
                ep_df2, sample_eff,
                out_html=os.path.join(args.out_dir, 'metrics_dashboard.html')
            )
            print("Metrics dashboard saved.")

    # Optional: visualize a single episode
    if args.visualize_episode is not None:
        idx = args.visualize_episode
        if visualize_episode_with_metrics is None:
            print('Visualization function not available; ensure vis/generate_and_visualize.py is importable')
        else:
            try:
                # visualize_episode_with_metrics expects an array shaped (1, num_agents, T, 3)
                vis_pos = positions_list[idx]
                vis_positions = np.expand_dims(vis_pos, axis=0)
                vis_goals = np.expand_dims(goals_list[idx], axis=0)
                out_html = os.path.join(args.out_dir, f'episode_{idx}.html')
                visualize_episode_with_metrics(
                    vis_positions,
                    vis_goals,
                    episode=idx,
                    output_html=out_html,
                )
                print('Wrote episode visualization to', out_html)
            except Exception as e:
                print(f'Failed to visualize episode {idx}: {e}')


if __name__ == '__main__':
    main()

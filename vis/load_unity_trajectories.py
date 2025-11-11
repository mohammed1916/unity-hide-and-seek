import argparse
import json
import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

# Import the visualizer functions from the repo (assumes vis/ is on the PYTHONPATH or run from repo root)
try:
    from generate_and_visualize import visualize_episode_with_metrics, plot_metrics_dashboard, enrich_with_ppo_metrics, save_metrics
except Exception as e:
    # we'll only import when needed
    visualize_episode_with_metrics = None
    plot_metrics_dashboard = None
    enrich_with_ppo_metrics = None
    save_metrics = None


def load_json_episodes(folder):
    files = sorted(glob(os.path.join(folder, "episode_*.json")))
    episodes = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                episodes.append(json.load(fh))
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    return episodes


def episodes_to_arrays(episodes, pad_mode='edge'):
    """Convert Unity EpisodeTrajectory JSONs to positions/velocities arrays and dataframes.

    Returns:
      positions: np.array shape (num_episodes, num_agents, T, 3)
      velocities: np.array shape (num_episodes, num_agents, T, 3)
      ep_df: pandas.DataFrame episode-level metrics
      agent_df: pandas.DataFrame per-agent metrics
    """
    if len(episodes) == 0:
        return None, None, None, None

    # Determine max agents and max T
    max_agents = 0
    max_steps = 0
    for ep in episodes:
        agents = ep.get('agents', [])
        max_agents = max(max_agents, len(agents))
        max_steps = max(max_steps, ep.get('steps', 0))

    num_eps = len(episodes)
    positions = np.zeros((num_eps, max_agents, max_steps, 3), dtype=float)
    velocities = np.zeros_like(positions)

    ep_rows = []
    agent_rows = []

    for i, ep in enumerate(episodes):
        agents = ep.get('agents', [])
        ep_steps = ep.get('steps', 0)
        avg_reward = ep.get('averageEpisodicReward', None)
        # per-agent positions
        for a_idx, a in enumerate(agents):
            pos_list = a.get('positions', [])
            # convert to numpy array (T_a, 3)
            if len(pos_list) == 0:
                continue
            arr = np.array([[p['x'], p['y'], p['z']] for p in pos_list], dtype=float)
            T_a = arr.shape[0]
            # copy into positions, pad if necessary
            positions[i, a_idx, :T_a, :] = arr
            if T_a < max_steps:
                # pad using last value or edge
                if pad_mode == 'edge' and T_a > 0:
                    positions[i, a_idx, T_a:, :] = arr[-1:]
            # compute velocities as diff (for t>0) and copy (we'll have 0 at t=0)
            if T_a > 1:
                vel = np.zeros((max_steps, 3), dtype=float)
                vel[:T_a-1, :] = arr[1:] - arr[:T_a-1]
                # last frames remain 0 or last diff
                velocities[i, a_idx, :, :] = vel
            # collect agent row
            # cum_reward = a.get('cumulativeReward', None)
            agent_rows.append({
                'episode': int(i),
                'agent': int(a_idx),
                'instanceId': a.get('instanceId', None),
                'name': a.get('name', ''),
                # 'cum_reward': float(cum_reward) if cum_reward is not None else np.nan,
                'steps': int(ep_steps),
            })
        # episode row
        ep_rows.append({
            'episode': int(i),
            'steps': int(ep_steps),
            'averageEpisodicReward': float(avg_reward) if avg_reward is not None else np.nan,
            'timestamp': ep.get('timestamp', None)
        })

    ep_df = pd.DataFrame(ep_rows)
    agent_df = pd.DataFrame(agent_rows)
    return positions, velocities, ep_df, agent_df


def save_outputs(out_dir, positions, velocities, ep_df, agent_df):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'positions.npy'), positions)
    np.save(os.path.join(out_dir, 'velocities.npy'), velocities)
    ep_csv = os.path.join(out_dir, 'episode_metrics.csv')
    agent_csv = os.path.join(out_dir, 'agent_metrics.csv')
    ep_df.to_csv(ep_csv, index=False)
    agent_df.to_csv(agent_csv, index=False)
    print(f"Saved positions/velocities and CSV metrics to {out_dir}")


def main():
    p = argparse.ArgumentParser(description='Load Unity JSON trajectories and convert them for visualization')
    p.add_argument('--input-dir', '-i', required=True, help='Directory containing Unity episode_*.json files')
    p.add_argument('--out-dir', '-o', default='vis_output', help='Output directory for numpy arrays and CSVs')
    p.add_argument('--dt', type=float, default=0.02, help='Time delta between FixedUpdate steps used to compute velocities (default 0.02)')
    p.add_argument('--visualize-episode', type=int, default=None, help='If set, generate an interactive HTML visualization for this episode index')
    p.add_argument('--dashboard', action='store_true', help='If set, build a metrics dashboard HTML (requires generate_and_visualize.plot_metrics_dashboard)')
    args = p.parse_args()

    episodes = load_json_episodes(args.input_dir)
    if not episodes:
        print('No episodes found in', args.input_dir)
        return

    # add timestamp to episodes if missing
    for ep in episodes:
        if 'timestamp' not in ep:
            ep['timestamp'] = datetime.now().isoformat()

    positions, velocities, ep_df, agent_df = episodes_to_arrays(episodes)
    if positions is None:
        print('No positions parsed')
        return

    save_outputs(args.out_dir, positions, velocities, ep_df, agent_df)

    # Optionally call the enrichment and plotting functions from generate_and_visualize
    if args.dashboard:
        if enrich_with_ppo_metrics is None or plot_metrics_dashboard is None:
            print('Dashboard functions not available; ensure vis/generate_and_visualize.py is importable (run from repo root)')
        else:
            ep_df2, agent_df2, sample_eff = enrich_with_ppo_metrics(ep_df, agent_df, positions, goals=np.zeros((positions.shape[0], positions.shape[1], 3)))
            plot_metrics_dashboard(ep_df2, sample_eff, out_html=os.path.join(args.out_dir, 'metrics_dashboard.html'))

    if args.visualize_episode is not None:
        idx = args.visualize_episode
        if visualize_episode_with_metrics is None:
            print('Visualization function not available; ensure vis/generate_and_visualize.py is importable (run from repo root)')
        else:
            # We need per-episode positions shape (1, num_agents, T, 3)
            try:
                vis_positions = positions[idx:idx+1]
            except Exception as e:
                print(f'Failed to select episode {idx}:', e)
                return
            out_html = os.path.join(args.out_dir, f'episode_{idx}.html')
            visualize_episode_with_metrics(vis_positions, np.zeros((1, vis_positions.shape[1], 3)), episode=0, output_html=out_html)
            print('Wrote episode visualization to', out_html)


if __name__ == '__main__':
    main()

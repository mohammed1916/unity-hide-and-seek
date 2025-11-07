import argparse
import json
import os
import uuid
import time
from datetime import datetime

try:
    from mlagents_envs.side_channel.side_channel import SideChannel
    from mlagents_envs.environment import UnityEnvironment
except Exception as e:
    raise RuntimeError("mlagents_envs is required for the sidechannel receiver. pip install mlagents") from e

# Optional visualizer import
try:
    from generate_and_visualize import visualize_episode_with_metrics
except Exception:
    visualize_episode_with_metrics = None

# TensorBoard writer
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class TrajSideChannel(SideChannel):
    def __init__(self, channel_guid):
        super().__init__(uuid.UUID(channel_guid))
        self._counter = 0
        self._out_dir = None
        self._writer = None

    def set_out_dir(self, out_dir):
        self._out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def set_summary_writer(self, writer):
        self._writer = writer

    def on_message_received(self, msg) -> None:
        try:
            s = msg.read_string()
        except Exception as e:
            print('Failed to read SideChannel message:', e)
            return
        # parse JSON
        try:
            obj = json.loads(s)
        except Exception as e:
            print('Malformed JSON from SideChannel:', e)
            return
        # Save to disk
        if self._out_dir is not None:
            filename = f"sc_episode_{self._counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(self._out_dir, filename)
            try:
                with open(path, 'w', encoding='utf-8') as fh:
                    json.dump(obj, fh, indent=2)
                print(f"SideChannel: saved payload to {path}")
            except Exception as e:
                print('Failed to write payload to disk:', e)
        # Optionally log average episodic reward to TensorBoard
        if self._writer is not None and isinstance(obj, dict):
            avg = obj.get('averageEpisodicReward', None)
            if avg is not None:
                try:
                    self._writer.add_scalar('trajectories/averageEpisodicReward', float(avg), global_step=self._counter)
                    self._writer.flush()
                except Exception as e:
                    print('Failed to write to TensorBoard:', e)
        # Optionally call visualizer for this single episode
        if visualize_episode_with_metrics is not None and isinstance(obj, dict):
            # Build positions array expected by visualize function: (1, num_agents, T, 3)
            agents = obj.get('agents', [])
            if len(agents) > 0:
                # determine T
                T = max(len(a.get('positions', [])) for a in agents)
                num_agents = len(agents)
                import numpy as np
                positions = np.zeros((1, num_agents, T, 3), dtype=float)
                goals = np.zeros((1, num_agents, 3), dtype=float)
                for ai, a in enumerate(agents):
                    pos_list = a.get('positions', [])
                    for t, p in enumerate(pos_list):
                        positions[0, ai, t, 0] = p.get('x', 0.0)
                        positions[0, ai, t, 1] = p.get('y', 0.0)
                        positions[0, ai, t, 2] = p.get('z', 0.0)
                out_html = os.path.join(self._out_dir, f"sc_episode_{self._counter}.html")
                try:
                    visualize_episode_with_metrics(positions, goals, episode=0, output_html=out_html)
                    print(f"SideChannel: wrote visualization to {out_html}")
                except Exception as e:
                    print('Visualizer failed:', e)
        self._counter += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', default='sc_trajectories', help='Directory to save incoming episode payloads and visualizations')
    parser.add_argument('--editor', action='store_true', help='Connect to Editor (default behavior).')
    parser.add_argument('--unity-exe', default=None, help='Path to Unity build to connect to (if not Editor)')
    parser.add_argument('--tb-logdir', default=None, help='Optional tensorboard logdir to write scalars')
    args = parser.parse_args()

    channel_guid = '11111111-2222-3333-4444-555555555555'
    sc = TrajSideChannel(channel_guid)
    sc.set_out_dir(args.out_dir)

    writer = None
    if args.tb_logdir is not None:
        if SummaryWriter is None:
            print('TensorBoard SummaryWriter not available. Install torch or tensorboardX.')
        else:
            writer = SummaryWriter(log_dir=args.tb_logdir)
            sc.set_summary_writer(writer)

    # Connect to Unity
    env = None
    try:
        env = UnityEnvironment(file_name=args.unity_exe if args.unity_exe else None, side_channels=[sc])
        env.reset()
        print('Connected to Unity, listening for SideChannel messages. Press Ctrl-C to exit.')
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('Interrupted by user, closing.')
    except Exception as e:
        print('Failed to connect to Unity:', e)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()

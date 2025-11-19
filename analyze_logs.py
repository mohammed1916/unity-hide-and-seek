import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import pandas as pd

# Function to load TensorBoard logs
def load_tensorboard_logs(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None
    event_file = max(event_files, key=os.path.getctime)  # Get the latest event file
    ea = EventAccumulator(event_file)
    ea.Reload()
    print(f"Available scalar tags in {log_dir}: {ea.Tags()['scalars']}")
    return ea

# Function to extract scalar values
def extract_scalars(ea, tag):
    if tag in ea.Tags()['scalars']:
        return [(s.step, s.value) for s in ea.Scalars(tag)]
    return []

# Paths to the run directories
run_dirs = [
    r'c:\Users\BBBS-AI-01\d\rl\unity_proj\unity-hide-and-seek\results\run41',
    r'c:\Users\BBBS-AI-01\d\rl\unity_proj\unity-hide-and-seek\results\run41_1',
    r'c:\Users\BBBS-AI-01\d\rl\unity_proj\unity-hide-and-seek\results\run41_2',
    r'c:\Users\BBBS-AI-01\d\rl\unity_proj\unity-hide-and-seek\results\run41_3',
    r'c:\Users\BBBS-AI-01\d\rl\unity_proj\unity-hide-and-seek\results\run41_4',
    r'c:\Users\BBBS-AI-01\d\rl\unity_proj\unity-hide-and-seek\results\run41_5'
]

agents = ['Hider0', 'Hider1', 'Hider2', 'Seeker0', 'Seeker1', 'Seeker2']

# For each run, for each agent, load logs
data = {}
for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)
    data[run_name] = {}
    for agent in agents:
        agent_dir = os.path.join(run_dir, agent)
        if os.path.exists(agent_dir):
            ea = load_tensorboard_logs(agent_dir)
            if ea:
                data[run_name][agent] = {
                    'entropy': extract_scalars(ea, 'Policy/Entropy'),
                    'reward': extract_scalars(ea, 'Environment/Cumulative Reward'),
                    'loss': extract_scalars(ea, 'Policy/Loss'),
                    'episode_length': extract_scalars(ea, 'Environment/Episode Length')
                }
            else:
                data[run_name][agent] = None

# Now, for exploration/exploitation study
# Plot entropy over time for each agent
for run_name, agents_data in data.items():
    for agent, metrics in agents_data.items():
        if metrics and metrics['entropy']:
            steps, entropies = zip(*metrics['entropy'])
            plt.figure()
            plt.plot(steps, entropies)
            plt.title(f'Entropy over time for {agent} in {run_name}')
            plt.xlabel('Steps')
            plt.ylabel('Entropy')
            plt.savefig(f'entropy_{agent}_{run_name}.png')
            plt.close()

# To observe shift: when entropy starts decreasing
# For actions repeated vs changed: need action logs, but not available
# Perhaps look at policy loss or something

# For rewards, plot cumulative reward
for run_name, agents_data in data.items():
    for agent, metrics in agents_data.items():
        if metrics and metrics['reward']:
            steps, rewards = zip(*metrics['reward'])
            plt.figure()
            plt.plot(steps, rewards)
            plt.title(f'Cumulative Reward over time for {agent} in {run_name}')
            plt.xlabel('Steps')
            plt.ylabel('Reward')
            plt.savefig(f'reward_{agent}_{run_name}.png')
            plt.close()

# For reward decomposition, since not logged, suggest modifying code to log individual rewards
print("To decompose rewards, modify HideAndSeekAgent.cs to log individual reward components using StatsRecorder.")
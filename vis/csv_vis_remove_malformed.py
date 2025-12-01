import pandas as pd
import plotly.graph_objects as go
import glob

# Helper to read CSV and filter only rows with exact columns
def read_csv_filtered(file_path, expected_cols):
    valid_rows = []
    malformed_times = set()
    with open(file_path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == expected_cols:
                valid_rows.append(parts)
            else:
                # Capture timestamp if malformed
                if len(parts) > 0:
                    try:
                        malformed_times.add(float(parts[0]))
                    except ValueError:
                        pass
    df = pd.DataFrame(valid_rows, columns=header[:expected_cols])
    for col in df.columns:
        if col != "block_id":  # keep block_id as string
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df, malformed_times

# Load blocks (7 columns)
blocks_df, malformed_blocks = read_csv_filtered("trajectory_logs_/run41_5/episode_9/blocks.csv", 7)

# Load agents (5 columns)
agent_files = glob.glob("trajectory_logs_/run41_5/episode_11/agent_*.csv")
agents = {}
malformed_agents = set()
for file in agent_files:
    df, malformed_times = read_csv_filtered(file, 5)
    name = file.split("/")[-1].replace(".csv", "")
    agents[name] = df
    malformed_agents.update(malformed_times)

# Combine all malformed timestamps
all_malformed = malformed_blocks.union(malformed_agents)
print("Malformed timestamps to ignore:", sorted(all_malformed))

# Drop malformed timestamps from data
blocks_df = blocks_df[~blocks_df['time'].isin(all_malformed)]
for name in agents:
    agents[name] = agents[name][~agents[name]['time'].isin(all_malformed)]

# Ensure no remaining NaNs
blocks_df.dropna(inplace=True)
for name in agents:
    agents[name].dropna(inplace=True)

# Unique timestamps (valid only)
timestamps = sorted(blocks_df['time'].unique())

# Create frames
frames = []
for t in timestamps:
    blocks_t = blocks_df[blocks_df['time'] == t]
    block_trace = go.Scatter3d(
        x=blocks_t['x'],
        y=blocks_t['z'],  # Z as vertical
        z=[1]*len(blocks_t),  # horizontal plane
        mode='markers',
        marker=dict(size=5, color='brown'),
        name='Blocks'
    )

    agent_traces = []
    for name, df in agents.items():
        df_t = df[df['time'] == t]
        agent_traces.append(
            go.Scatter3d(
                x=df_t['x'],
                y=df_t['z'],
                z=[1]*len(df_t),
                mode='markers',
                marker=dict(size=5, color='blue' if "Hider" in name else 'red'),
                name=name
            )
        )

    frames.append(go.Frame(data=[block_trace] + agent_traces, name=str(t)))

# Initial frame (first valid timestamp)
init_t = timestamps[0]
init_blocks = blocks_df[blocks_df['time'] == init_t]
init_block_trace = go.Scatter3d(
    x=init_blocks['x'],
    y=init_blocks['z'],
    z=[1]*len(init_blocks),
    mode='markers',
    marker=dict(size=5, color='brown'),
    name='Blocks'
)

init_agent_traces = []
for name, df in agents.items():
    df_t = df[df['time'] == init_t]
    init_agent_traces.append(
        go.Scatter3d(
            x=df_t['x'],
            y=df_t['z'],
            z=[1]*len(df_t),
            mode='markers',
            marker=dict(size=5, color='blue' if "Hider" in name else 'red'),
            name=name
        )
    )

# Precompute full paths for agents
past_traces = []
for name, df in agents.items():
    past_traces.append(
        go.Scatter3d(
            x=df['x'],
            y=df['z'],
            z=[1]*len(df),
            mode='lines',
            line=dict(color='blue' if "Hider" in name else 'red', width=2),
            name=f"{name}_path",
            visible=False
        )
    )

# Blocks paths
past_block_trace = go.Scatter3d(
    x=blocks_df['x'],
    y=blocks_df['z'],
    z=[1]*len(blocks_df),
    mode='lines',
    line=dict(color='brown', width=2),
    name='Blocks_path',
    visible=False
)

# Build figure
fig = go.Figure(
    data=[init_block_trace] + init_agent_traces + past_traces + [past_block_trace],
    frames=frames
)

# Slider
sliders = [dict(
    steps=[dict(
        method='animate',
        label=str(t),
        args=[[str(t)],
              dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))]
    ) for t in timestamps],
    transition=dict(duration=0),
    x=0, y=0, currentvalue=dict(font=dict(size=12), prefix="Time: ", visible=True),
    len=1.0
)]

# Layout with buttons
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X', range=[-20, 20]),
        yaxis=dict(title='Z (vertical)', range=[-20, 20]),
        zaxis=dict(title='Y (horizontal)', range=[0.8, 2]),
    ),
    sliders=sliders,
    updatemenus=[dict(
        type='buttons', showactive=False,
        y=1, x=0.8, xanchor='left', yanchor='bottom',
        pad=dict(t=50, r=10),
        buttons=[
            dict(label='Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode='immediate')]),
            dict(label='Pause',
                 method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
            dict(label='Show Past Positions',
                 method='update',
                 args=[{'visible': [True]*len(past_traces + [past_block_trace]) + [True]*len(init_agent_traces + [init_block_trace])},
                       {'title': 'Past Positions'}]),
            dict(label='Hide Past Positions',
                 method='update',
                 args=[{'visible': [False]*len(past_traces + [past_block_trace]) + [True]*len(init_agent_traces + [init_block_trace])},
                       {'title': 'No Past Positions'}])
        ]
    )]
)

fig.show()

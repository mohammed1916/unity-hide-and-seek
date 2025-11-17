import pandas as pd
import plotly.graph_objects as go
import glob

# Helper to read only rows with exact columns
def read_csv_filtered(file_path, expected_cols):
    valid_rows = []
    with open(file_path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == expected_cols:
                valid_rows.append(parts)
    df = pd.DataFrame(valid_rows, columns=header[:expected_cols])
    for col in df.columns:
        if col != "block_id":  # keep block_id as string
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Load blocks (7 columns)
blocks_df = read_csv_filtered("trajectory_logs_/default_run/episode_2/blocks.csv", 7)

# Load agents (5 columns)
agent_files = glob.glob("trajectory_logs_/default_run/episode_2/agent_*.csv")
agents = {}
for file in agent_files:
    df = read_csv_filtered(file, 5)
    name = file.split("/")[-1].replace(".csv", "")
    agents[name] = df

# Unique timestamps
timestamps = sorted(blocks_df['time'].unique())

# Create frames
frames = []
for t in timestamps:
    # Blocks positions at time t (y = 1)
    blocks_t = blocks_df[blocks_df['time'] == t]
    block_trace = go.Scatter3d(
        x=blocks_t['x'], y=[1]*len(blocks_t), z=blocks_t['z'],
        mode='markers',
        marker=dict(size=5, color='brown'),
        name='Blocks'
    )

    # Agents positions at time t (y = 1)
    agent_traces = []
    for name, df in agents.items():
        df_t = df[df['time'] == t]
        agent_traces.append(
            go.Scatter3d(
                x=df_t['x'], y=[1]*len(df_t), z=df_t['z'],
                mode='markers',
                marker=dict(size=5, color='blue' if "Hider" in name else 'red'),
                name=name
            )
        )

    frames.append(go.Frame(data=[block_trace] + agent_traces, name=str(t)))

# Initial data (first timestamp)
init_t = timestamps[0]
init_blocks = blocks_df[blocks_df['time'] == init_t]
init_block_trace = go.Scatter3d(
    x=init_blocks['x'], y=[1]*len(init_blocks), z=init_blocks['z'],
    mode='markers', marker=dict(size=5, color='brown'), name='Blocks'
)
init_agent_traces = []
for name, df in agents.items():
    df_t = df[df['time'] == init_t]
    init_agent_traces.append(
        go.Scatter3d(
            x=df_t['x'], y=[1]*len(df_t), z=df_t['z'],
            mode='markers',
            marker=dict(size=5, color='blue' if "Hider" in name else 'red'),
            name=name
        )
    )

fig = go.Figure(
    data=[init_block_trace] + init_agent_traces,
    frames=frames
)

# Slider
sliders = [dict(
    steps=[dict(method='animate', label=str(t),
                args=[[str(t)], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))])
           for t in timestamps],
    transition=dict(duration=0),
    x=0, y=0, currentvalue=dict(font=dict(size=12), prefix="Time: ", visible=True),
    len=1.0
)]

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y (fixed = 1)',
        zaxis_title='Z'
    ),
    width=800, height=600,
    sliders=sliders,
    updatemenus=[dict(type='buttons', showactive=False,
                      y=1,
                      x=0.8,
                      xanchor='left',
                      yanchor='bottom',
                      pad=dict(t=45, r=10),
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode='immediate')]),
                               dict(label='Pause',
                                    method='animate',
                                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                               ])]
)

fig.show()

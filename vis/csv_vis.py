import pandas as pd
import plotly.graph_objects as go
import glob

# ----------------------------
# Load CSVs
# ----------------------------
def read_csv_filtered(file_path, expected_cols):
    """Read CSV and keep only rows with exactly expected_cols columns."""
    valid_rows = []
    with open(file_path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == expected_cols:
                valid_rows.append(parts)
    # Convert to DataFrame
    df = pd.DataFrame(valid_rows, columns=header[:expected_cols])
    # Convert numeric columns to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# -----------------------------
# Blocks (7 columns)
# -----------------------------
blocks_df = read_csv_filtered("trajectory_logs_/default_run/episode_3/blocks.csv", 7)

# -----------------------------
# agent CSVs (5 columns)
# -----------------------------
import glob
agent_files = glob.glob("trajectory_logs_/default_run/episode_3/agent_*.csv")
agents = {}
for file in agent_files:
    df = read_csv_filtered(file, 5)
    name = file.split("/")[-1].replace(".csv", "")
    agents[name] = df

print("Blocks:", blocks_df.shape)
for name, df in agents.items():
    print(name, df.shape)


# ----------------------------
# Create figure
# ----------------------------
fig = go.Figure()

# Blocks as spheres
for i, row in blocks_df.iterrows():
    fig.add_trace(go.Scatter3d(
        x=[row['x']],
        y=[row['y']],
        z=[row['z']],
        mode='markers',
        marker=dict(size=8, color='brown'),
        name=f"Block{row['block_id']}"
    ))

# Agents as colored points
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
for idx, (agent_name, df) in enumerate(agents.items()):
    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='lines+markers',
        line=dict(color=colors[idx % len(colors)], width=3),
        marker=dict(size=4),
        name=agent_name
    ))

# ----------------------------
# Layout
# ----------------------------
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'
    ),
    title="Hide & Seek Trajectory Visualization",
    width=900,
    height=700
)

fig.show()

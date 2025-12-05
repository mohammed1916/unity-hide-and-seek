TrajectoryLogging Plugin
========================

Overview
--------
A small, managed Unity logging plugin to capture per-episode, per-agent, and event-level data as human-readable CSV files and optionally push curated summary scalars/events to ML-Agents `StatsRecorder` (TensorBoard).

Why use this plugin?
- Records raw trajectories and episode metadata in plain CSV files (easy to inspect, version, and process).
- Produces `combine.csv` and `episode_meta.csv` for run-level and episode-level summaries.
- Writes `events.csv` so you can annotate timelines and visualize events in offline tools.
- Optionally pushes a small set of summary scalars and named events to `StatsRecorder` for TensorBoard visualization.

Files
-----
- [Assets/Plugins/TrajectoryLogging/EpisodeLogger.cs](Assets/Plugins/TrajectoryLogging/EpisodeLogger.cs)
  - Main logging API: `Init`, `StartNewEpisode`, `LogAgent`, `LogBlock`, `LogEvent`, `WriteEpisodeSummary`, `FlushWriters`, `Close`.
- [Assets/Plugins/TrajectoryLogging/TrajectoryLogger.cs](Assets/Plugins/TrajectoryLogging/TrajectoryLogger.cs)
  - Backwards-compatible singleton facade: `TrajectoryLogger.Instance.RecordAgentStep(...)`. For compatibility with existing call sites.
- [Assets/Plugins/TrajectoryLogging/README.md](Assets/Plugins/TrajectoryLogging/README.md)
  - This file.

Quick start (examples)
----------------------
Initialize (e.g., in a manager `Awake()`):

```csharp
private EpisodeLogger episodeLogger = new EpisodeLogger();

void Awake()
{
    string basePath = Path.Combine(Directory.GetCurrentDirectory(), "trajectory_logs_myrun");
    // reportToStats toggles whether the logger will push curated scalars/events into StatsRecorder
    episodeLogger.Init(basePath, reportToStats: true);
    int episodeIndex = episodeLogger.StartNewEpisode();
    episodeLogger.LogEvent("EpisodeStarted", episodeIndex);
}
```

Per-step logging (e.g., in `FixedUpdate()`):

```csharp
// per-agent position
episodeLogger.LogAgent("Hider0_name", transform.position, isActive ? 1f : 0f);
// per-block state
episodeLogger.LogBlock(blockId, blockPos, locked ? 1f : 0f, held ? 1f : 0f);
```

Sparse semantic events (recommended):

```csharp
// record a semantic event (also written to events.csv and optionally to StatsRecorder)
episodeLogger.LogEvent("HiderCaptured", hiderIndex);
```

Episode end summary

```csharp
episodeLogger.WriteEpisodeSummary(
    episodeSteps,
    episodeTimer,
    durationSeconds,
    outcome,
    outcomeSeekersCol,
    outcomeHidersCol,
    winner,
    hidersCaptured,
    stepsHidden,
    timeHidden,
    hidersPerfectGame,
    gracePeriodEnded
);
episodeLogger.FlushWriters();
```

Shutdown (e.g., `OnApplicationQuit`):

```csharp
episodeLogger.Close();
```

Backwards compatibility
----------------------
If you have existing code that calls `TrajectoryLogger.Instance.RecordAgentStep(instanceId, position)`, keep `TrajectoryLogger.cs` in place — it forwards to `EpisodeLogger` and requires no changes to callers.

CSV schemas
-----------
- Per-agent CSV (per episode): `agent_<id>.csv`
  - Header: `time,x,y,z,isActive`
  - Rows: per-step floats/timestamps (human-readable, one row per step)
- Blocks CSV (per episode): `blocks.csv`
  - Header: `time,block_id,x,y,z,isLocked,isHeld`
- Events CSV (per episode): `events.csv`
  - Header: `time,event_name,value`
  - Written by `LogEvent(name, value)`; good for timeline annotations and sparse markers
- Episode meta CSV (per episode): `episode_meta.csv`
  - Contains a single-row (plus header) describing the episode
- Combine CSV (run root): `combine.csv`
  - Appends one row per episode summarizing run-level metrics

StatsRecorder integration
------------------------
- `EpisodeLogger.Init(basePath, reportToStats: true)` (default) will cause the logger to push a curated set of summary scalars and events to `Academy.Instance.StatsRecorder`:
  - Episode summaries: `Episode/EpisodeSteps`, `Episode/DurationSeconds`, `Episode/HidersCaptured`, `Episode/TimeHidden`, `Episode/WinnerIsHiders`, etc.
  - Named events via `LogEvent("EventName", value)` are pushed to `Events/<EventName>`.
- Important: avoid pushing high-cardinality per-step/per-agent logs into `StatsRecorder` (TensorBoard) — it leads to huge event files and slows training. Keep `StatsRecorder` for summaries and a few sparse events; keep raw CSVs for full traces and replay/visualization.

Custom metrics
--------------
You can record custom per-episode metrics in two ways: push-style (from anywhere during the episode) and pull-style (provide a function that computes metrics at summary time).

- Push-style API (recommended when multiple systems add metrics):

```csharp
// anywhere in your code
episodeLogger.AddMetric("my/custom_metric", 12.34f);
```

These accumulated metrics are written to `episode_metrics.csv` inside the episode folder when you call `WriteEpisodeSummary(...)` and — if `reportToStats` is enabled — pushed to TensorBoard under `Custom/<metric_name>`.

- Pull-style provider (recommended when metrics are computed only at summary time):

```csharp
episodeLogger.MetricProvider = () => new Dictionary<string,float>
{
  {"score", ComputeScore() },
  {"completion_ratio", ComputeCompletion() }
};
```

WriteEpisodeSummary will invoke `MetricProvider` (if set) and merge any returned metrics with push-style metrics before writing them to `episode_metrics.csv` and (optionally) pushing them to `StatsRecorder`.

You can also clear push-style metrics manually with:

```csharp
episodeLogger.ClearMetrics();
```

Visualization workflow
----------------------
1. Run experiments — plugin writes per-episode CSV folders under the provided `basePath`.
2. Use Python (pandas, Plotly) or your existing tools to load the `episode_<n>` folders, `agent_*.csv`, `blocks.csv` and `events.csv`.
3. Use `events.csv` to overlay event markers when animating trajectories.

Performance & best practices
---------------------------
- Initialize early (in `Awake`) so agents calling `TrajectoryLogger.Instance` or `episodeLogger` find the logger ready.
- Call `FlushWriters()` periodically (the plugin already flushes per-step where used); large flush frequency reduces risk of data loss at the cost of I/O.
- Use `LogEvent(...)` for sparse, high-value events (captures, major state transitions) and rely on CSV trace files for dense per-step data.

Optional: assembly definition (`.asmdef`)
----------------------------------------
To make the plugin a distinct managed assembly, add an assembly definition file in `Assets/Plugins/TrajectoryLogging`. This helps compile order and explicit references between parts of the project. If you want I can scaffold an `.asmdef` for you.

Troubleshooting
---------------
- If you see `TrajectoryLogger` not found errors, ensure `TrajectoryLogger.cs` exists (it is a small facade) or update call sites to use `EpisodeLogger`.
- If TensorBoard event files grow too large, set `reportToStats: false` when calling `Init` or reduce the set of events you push.

Want an example scene or an `.asmdef` file created for the plugin? Reply with `example` or `asmdef` and I will add it.

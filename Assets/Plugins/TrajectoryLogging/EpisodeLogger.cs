using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using Unity.MLAgents;

namespace TrajectoryLogging
{
    /// <summary>
    /// Encapsulates episode/run-level CSV logging (combine.csv, episode_meta.csv, per-agent CSVs, blocks.csv).
    /// Usage:
    ///   var logger = new EpisodeLogger();
    ///   logger.Init(basePath);
    ///   logger.StartNewEpisode();
    ///   logger.LogAgent(agentId, position, isActive);
    ///   logger.LogBlock(id, position, locked, held);
    ///   logger.WriteEpisodeSummary(...);
    ///   logger.FlushWriters();
    ///   logger.Close();
    /// </summary>
    public class EpisodeLogger
    {
        private string rootPath;
        private string episodePath;
        private StreamWriter combineWriter;
        private StreamWriter blockWriter;
        private StreamWriter episodeMetaWriter;
        private StreamWriter eventsWriter;
        private Dictionary<string, StreamWriter> agentWriters = new Dictionary<string, StreamWriter>();
        private int episodeIndex = 0;

        public bool Initialized { get; private set; } = false;

        private bool reportToStats = true;

        public void Init(string basePath, bool reportToStats = true)
        {
            if (string.IsNullOrEmpty(basePath)) throw new ArgumentException("basePath required");
            rootPath = basePath;
            Directory.CreateDirectory(rootPath);

            this.reportToStats = reportToStats;

            // Open/append combine.csv
            try
            {
                string combineFile = Path.Combine(rootPath, "combine.csv");
                bool writeHeader = !File.Exists(combineFile);
                combineWriter = new StreamWriter(new FileStream(combineFile, FileMode.Append, FileAccess.Write, FileShare.ReadWrite));
                if (writeHeader)
                {
                    combineWriter.WriteLine("episode_index,episode_steps,episode_duration_steps,episode_duration_seconds,outcome,outcome_seekers,outcome_hiders,winner,hidersCaptured,stepsHidden,timeHidden,hidersPerfectGame,gracePeriodEnded");
                    combineWriter.Flush();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"EpisodeLogger: Failed to create/open combine CSV: {e.Message}");
                combineWriter = null;
            }

            Initialized = true;
        }

        public int StartNewEpisode()
        {
            if (!Initialized) throw new InvalidOperationException("EpisodeLogger not initialized");
            episodeIndex++;
            episodePath = Path.Combine(rootPath, $"episode_{episodeIndex}");
            Directory.CreateDirectory(episodePath);

            // Close previous writers
            foreach (var w in agentWriters.Values) { try { w.Close(); } catch { } }
            agentWriters.Clear();
            try { blockWriter?.Close(); } catch { }

            // Create block writer and episode meta writer
            try
            {
                blockWriter = new StreamWriter(new FileStream(Path.Combine(episodePath, "blocks.csv"), FileMode.Create, FileAccess.Write, FileShare.ReadWrite));
                blockWriter.WriteLine("time,block_id,x,y,z,isLocked,isHeld");
            }
            catch (Exception e)
            {
                Debug.LogError($"EpisodeLogger: Failed to create block writer: {e.Message}");
                blockWriter = null;
            }

            // events.csv for arbitrary named events (also pushed to StatsRecorder if enabled)
            try
            {
                eventsWriter = new StreamWriter(new FileStream(Path.Combine(episodePath, "events.csv"), FileMode.Create, FileAccess.Write, FileShare.ReadWrite));
                eventsWriter.WriteLine("time,event_name,value");
            }
            catch (Exception e)
            {
                Debug.LogError($"EpisodeLogger: Failed to create events writer: {e.Message}");
                eventsWriter = null;
            }

            try
            {
                episodeMetaWriter = new StreamWriter(new FileStream(Path.Combine(episodePath, "episode_meta.csv"), FileMode.Create, FileAccess.Write, FileShare.ReadWrite));
                episodeMetaWriter.WriteLine("episode_index,episode_steps,episode_duration_steps,episode_duration_seconds,outcome,outcome_seekers,outcome_hiders,winner,hidersCaptured,stepsHidden,timeHidden,hidersPerfectGame,gracePeriodEnded");
            }
            catch (Exception e)
            {
                Debug.LogError($"EpisodeLogger: Failed to create episode_meta writer: {e.Message}");
                episodeMetaWriter = null;
            }

            return episodeIndex;
        }

        public void LogBlock(int id, Vector3 p, float locked, float held)
        {
            try
            {
                blockWriter?.WriteLine($"{Time.time:F4},Block{id},{p.x:F4},{p.y:F4},{p.z:F4},{locked},{held}");
            }
            catch (Exception e)
            {
                Debug.LogError($"EpisodeLogger: Failed to write block: {e.Message}");
            }
            // Optionally emit an event for visualization/summary
            try
            {
                LogEvent("BlockPlaced", 1f);
            }
            catch { }
        }

        public void LogAgent(string agentId, Vector3 p, float active)
        {
            try
            {
                if (!agentWriters.ContainsKey(agentId))
                {
                    string file = Path.Combine(episodePath, $"agent_{agentId}.csv");
                    var writer = new StreamWriter(new FileStream(file, FileMode.Create, FileAccess.Write, FileShare.ReadWrite));
                    writer.WriteLine("time,x,y,z,isActive");
                    agentWriters.Add(agentId, writer);
                }
                agentWriters[agentId].WriteLine($"{Time.time:F4},{p.x:F4},{p.y:F4},{p.z:F4},{active}");
            }
            catch (Exception e)
            {
                Debug.LogError($"EpisodeLogger: Failed to write agent {agentId}: {e.Message}");
            }
            // small optional summary event per agent step (may be noisy)
            try
            {
                if (reportToStats)
                {
                    Academy.Instance.StatsRecorder.Add($"Agent/{agentId}/Active", active);
                }
            }
            catch { }
        }

        public void WriteEpisodeSummary(int episodeSteps, int episodeTimer, float durationSeconds, string outcome, string outcomeSeekersCol, string outcomeHidersCol, string winner, int hidersCaptured, int stepsHidden, float timeHidden, bool hidersPerfectGame, bool gracePeriodEnded)
        {
            // Build row matching previous format
            string row = string.Format("{0},{1},{2},{3:F3},{4},{5},{6},{7},{8},{9},{10:F3},{11},{12}",
                episodeIndex,
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
                gracePeriodEnded);

            if (episodeMetaWriter != null)
            {
                try
                {
                    episodeMetaWriter.WriteLine(row);
                    episodeMetaWriter.Flush();
                }
                catch (Exception e)
                {
                    Debug.LogError($"EpisodeLogger: Failed to write episode metadata: {e.Message}");
                }
                try { episodeMetaWriter?.Close(); } catch { }
                episodeMetaWriter = null;
            }

            if (combineWriter != null)
            {
                try
                {
                    combineWriter.WriteLine(row);
                    combineWriter.Flush();
                }
                catch (Exception e)
                {
                    Debug.LogError($"EpisodeLogger: Failed to append to combine CSV: {e.Message}");
                }
            }

            // Also push summary scalars to StatsRecorder for TensorBoard visualization
            try
            {
                if (reportToStats)
                {
                    var stats = Academy.Instance.StatsRecorder;
                    stats.Add("Episode/EpisodeSteps", episodeSteps);
                    stats.Add("Episode/EpisodeTimerSteps", episodeTimer);
                    stats.Add("Episode/DurationSeconds", durationSeconds);
                    // outcome as numeric flag: 1 = success (hiders), 0 = timeout/failure
                    stats.Add("Episode/HidersCaptured", hidersCaptured);
                    stats.Add("Episode/StepsHidden", stepsHidden);
                    stats.Add("Episode/TimeHidden", timeHidden);
                    stats.Add("Episode/HidersPerfectGame", hidersPerfectGame ? 1f : 0f);
                    // Push a numeric indicator for winner
                    stats.Add("Episode/WinnerIsHiders", winner == "Hiders" ? 1f : 0f);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"EpisodeLogger: failed to push summary stats: {e.Message}");
            }
        }

        public void FlushWriters()
        {
            try { blockWriter?.Flush(); } catch { }
            foreach (var w in agentWriters.Values) { try { w.Flush(); } catch { } }
            try { eventsWriter?.Flush(); } catch { }
        }

        public void Close()
        {
            foreach (var w in agentWriters.Values) { try { w.Close(); } catch { } }
            agentWriters.Clear();
            try { blockWriter?.Close(); } catch { }
            try { episodeMetaWriter?.Close(); } catch { }
            try { eventsWriter?.Close(); } catch { }
            try { combineWriter?.Close(); } catch { }
        }

        /// <summary>
        /// Record a named event (written to events.csv and optionally pushed to StatsRecorder under Events/<name>).
        /// </summary>
        public void LogEvent(string name, float value)
        {
            try
            {
                eventsWriter?.WriteLine($"{Time.time:F4},{name},{value}");
                eventsWriter?.Flush();
            }
            catch (Exception e)
            {
                Debug.LogError($"EpisodeLogger: failed to write event {name}: {e.Message}");
            }

            if (reportToStats)
            {
                try
                {
                    Academy.Instance.StatsRecorder.Add($"Events/{name}", value);
                }
                catch { }
            }
        }
    }
}

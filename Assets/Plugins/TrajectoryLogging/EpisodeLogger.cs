using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

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
        private Dictionary<string, StreamWriter> agentWriters = new Dictionary<string, StreamWriter>();
        private int episodeIndex = 0;

        public bool Initialized { get; private set; } = false;

        public void Init(string basePath)
        {
            if (string.IsNullOrEmpty(basePath)) throw new ArgumentException("basePath required");
            rootPath = basePath;
            Directory.CreateDirectory(rootPath);

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
        }

        public void FlushWriters()
        {
            try { blockWriter?.Flush(); } catch { }
            foreach (var w in agentWriters.Values) { try { w.Flush(); } catch { } }
        }

        public void Close()
        {
            foreach (var w in agentWriters.Values) { try { w.Close(); } catch { } }
            agentWriters.Clear();
            try { blockWriter?.Close(); } catch { }
            try { episodeMetaWriter?.Close(); } catch { }
            try { combineWriter?.Close(); } catch { }
        }
    }
}

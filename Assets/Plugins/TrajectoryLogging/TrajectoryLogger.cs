using System;
using System.IO;
using UnityEngine;

namespace TrajectoryLogging
{
    /// <summary>
    /// Simple managed facade kept for backward compatibility with existing code
    /// that calls `TrajectoryLogger.Instance?.RecordAgentStep(int, Vector3)`.
    /// Forwards calls to an internal `EpisodeLogger` instance.
    /// </summary>
    public class TrajectoryLogger
    {
        public static TrajectoryLogger Instance { get; } = new TrajectoryLogger();

        private EpisodeLogger episodeLogger = new EpisodeLogger();
        private bool initialized = false;

        private TrajectoryLogger()
        {
            try
            {
                // Mirror GameController's default run path so logs remain consistent.
                string runId = "run" + DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string[] args = Environment.GetCommandLineArgs();
                for (int i = 0; i < args.Length; i++)
                {
                    if (args[i] == "--run-id" && i + 1 < args.Length)
                    {
                        runId = args[i + 1];
                    }
                }
                string basePath = Path.Combine(Directory.GetCurrentDirectory(), "trajectory_logs_", runId);
                episodeLogger.Init(basePath);
                episodeLogger.StartNewEpisode();
                initialized = true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"TrajectoryLogger: failed init: {e.Message}");
                initialized = false;
            }
        }

        public void RecordAgentStep(int instanceId, Vector3 position)
        {
            if (!initialized) return;
            try
            {
                string id = instanceId.ToString();
                episodeLogger.LogAgent(id, position, 1f);
            }
            catch (Exception e)
            {
                Debug.LogError($"TrajectoryLogger: RecordAgentStep failed: {e.Message}");
            }
        }

        // Optional convenience methods kept for other callers
        public void RecordAgentStep(string id, Vector3 position)
        {
            if (!initialized) return;
            try { episodeLogger.LogAgent(id, position, 1f); } catch (Exception) { }
        }

        public void RecordBlock(int id, Vector3 pos, float locked, float held)
        {
            if (!initialized) return;
            try { episodeLogger.LogBlock(id, pos, locked, held); } catch (Exception) { }
        }

        public void Flush() => episodeLogger.FlushWriters();
        public void Close() => episodeLogger.Close();
    }
}

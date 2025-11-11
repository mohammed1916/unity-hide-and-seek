using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

[Serializable]
public class SerializableVector3
{
    public float x;
    public float y;
    public float z;

    public SerializableVector3() { }
    public SerializableVector3(Vector3 v) { x = v.x; y = v.y; z = v.z; }
}

[Serializable]
public class AgentTrajectory
{
    public int instanceId;
    public string name;
    public List<SerializableVector3> positions = new List<SerializableVector3>();
    public float cumulativeReward = 0f;
}

[Serializable]
public class EpisodeTrajectory
{
    public int episodeIndex;
    public int steps;
    public List<AgentTrajectory> agents = new List<AgentTrajectory>();
    public float averageEpisodicReward;
}

public class TrajectoryLogger : MonoBehaviour
{
    public GameController gameController;
    public string outputFolder = "trajectories";
    // How many FixedUpdate steps to skip between samples. Set to 1 to log every step.
    public int sampleEveryNSteps = 1;
    // If true, write summary scalars to the ML-Agents StatsRecorder (TensorBoard)
    public bool logToStatsRecorder = true;

    private bool logging = false;
    private int episodeIndex = 0;
    private int stepCount = 0;
    private Dictionary<int, AgentTrajectory> trajectories = new Dictionary<int, AgentTrajectory>();
    private int _sampleCounter = 0;

    // Singleton instance so agents can call RecordAgentStep
    public static TrajectoryLogger Instance { get; private set; }

    private void Start()
    {
        Instance = this;
        if (gameController == null)
        {
            gameController = FindObjectOfType<GameController>();
        }
        if (gameController == null)
        {
            Debug.LogWarning("TrajectoryLogger: GameController not found. Disabling logger.");
            enabled = false;
            return;
        }
        Debug.Log("TrajectoryLogger: Subscribing to GameController episode events.");

        gameController.EpisodeStarted += OnEpisodeStarted;
        gameController.EpisodeEnded += OnEpisodeEnded;

        Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, outputFolder));

        // SideChannel streaming removed — this component writes episode JSON to disk only.
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
        if (gameController != null)
        {
            gameController.EpisodeStarted -= OnEpisodeStarted;
            gameController.EpisodeEnded -= OnEpisodeEnded;
        }
        // No side-channel cleanup required.
    }

    private void OnEpisodeStarted()
    {
        logging = true;
        stepCount = 0;
        _sampleCounter = 0;
        trajectories.Clear();

        // Initialize trajectories for currently active agents
        foreach (AgentActions h in gameController.GetHiders())
        {
            var t = new AgentTrajectory { instanceId = h.gameObject.GetInstanceID(), name = h.gameObject.name };
            trajectories[t.instanceId] = t;
        }
        foreach (AgentActions s in gameController.GetSeekers())
        {
            var t = new AgentTrajectory { instanceId = s.gameObject.GetInstanceID(), name = s.gameObject.name };
            trajectories[t.instanceId] = t;
        }
    }

    private void OnEpisodeEnded()
    {
        logging = false;
        // Fill cumulative rewards and compute average
        float sumRewards = 0f;
        int agentCount = 0;
        foreach (var kv in trajectories)
        {
            AgentActions agent = FindAgentByInstanceId(kv.Key);
            if (agent != null)
            {
                kv.Value.cumulativeReward = agent.HideAndSeekAgent.GetCumulativeReward();
                sumRewards += kv.Value.cumulativeReward;
                agentCount++;
            }
        }
        float avg = agentCount > 0 ? sumRewards / agentCount : 0f;

        EpisodeTrajectory episode = new EpisodeTrajectory();
        episode.episodeIndex = episodeIndex++;
        episode.steps = stepCount;
        episode.averageEpisodicReward = avg;
        episode.agents.AddRange(trajectories.Values);

        string json = JsonUtility.ToJson(episode, true);
        string filename = string.Format("episode_{0}_{1}.json", episode.episodeIndex, DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        string path = Path.Combine(Application.persistentDataPath, outputFolder, filename);
        File.WriteAllText(path, json);
        Debug.LogFormat("TrajectoryLogger: Wrote episode trajectory to {0}", path);

        // Optionally log summary scalars to the ML-Agents StatsRecorder so they
        // appear in TensorBoard under results/<run-id>/summaries when a trainer
        // is connected.
        if (logToStatsRecorder)
        {
            try
            {
                var stats = Academy.Instance.StatsRecorder;
                stats.Add("Trajectories/AverageEpisodicReward", episode.averageEpisodicReward);
                stats.Add("Trajectories/TeamReward", sumRewards);
                foreach (var kv in trajectories)
                {
                    var safeName = kv.Value.name != null ? kv.Value.name.Replace("/", "_") : kv.Key.ToString();
                    stats.Add($"Trajectories/Agent/{safeName}/CumulativeReward", kv.Value.cumulativeReward);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogFormat("TrajectoryLogger: StatsRecorder add failed: {0}", e.Message);
            }
        }

        // Streaming removed: payload is written to disk only.
    }

    private AgentActions FindAgentByInstanceId(int id)
    {
        // Search among hiders and seekers
        foreach (AgentActions h in gameController.GetHiders())
        {
            if (h.gameObject.GetInstanceID() == id) return h;
        }
        foreach (AgentActions s in gameController.GetSeekers())
        {
            if (s.gameObject.GetInstanceID() == id) return s;
        }
        return null;
    }

    private void FixedUpdate()
    {
        if (!logging) return;
        // Advance counters; actual sampling is performed when agents call RecordAgentStep
        stepCount++;
        _sampleCounter++;
    }

    /// <summary>
    /// Called by agents (or other code) to record a per-step sample.
    /// Respects the global sampling rate configured on the logger.
    /// </summary>
    /// <param name="instanceId">Agent GameObject instance id</param>
    /// <param name="position">World position to record</param>
    public void RecordAgentStep(int instanceId, Vector3 position)
    {
        if (!logging) return;
        // Only record when sample counter matches sampling policy
        if (sampleEveryNSteps > 1 && (_sampleCounter % sampleEveryNSteps) != 0) return;
        if (trajectories.TryGetValue(instanceId, out var t))
        {
            t.positions.Add(new SerializableVector3(position));
        }
    }


    // No side-channel reference — file-based logging only.
}

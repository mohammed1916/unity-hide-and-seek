using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
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

    private bool logging = false;
    private int episodeIndex = 0;
    private int stepCount = 0;
    private Dictionary<int, AgentTrajectory> trajectories = new Dictionary<int, AgentTrajectory>();

    private void Start()
    {
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

        gameController.EpisodeStarted += OnEpisodeStarted;
        gameController.EpisodeEnded += OnEpisodeEnded;

        Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, outputFolder));

        // Try to register a SideChannel for streaming (optional). Use SideChannelManager which
        // is available in this ML-Agents runtime version. If registration fails, keep working
        // with file-based logging only.
        try
        {
            var channel = new TrajectorySideChannel();
            SideChannelManager.RegisterSideChannel(channel);
            sideChannel = channel;
        }
        catch (System.Exception e)
        {
            Debug.LogFormat("TrajectoryLogger: SideChannel registration failed: {0}", e.Message);
            sideChannel = null;
        }
    }

    private void OnDestroy()
    {
        if (gameController != null)
        {
            gameController.EpisodeStarted -= OnEpisodeStarted;
            gameController.EpisodeEnded -= OnEpisodeEnded;
        }
        // Unregister the side channel if we registered one.
        try
        {
            if (sideChannel != null)
            {
                SideChannelManager.UnregisterSideChannel(sideChannel);
                sideChannel = null;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogFormat("TrajectoryLogger: SideChannel unregistration failed: {0}", e.Message);
        }
    }

    private void OnEpisodeStarted()
    {
        logging = true;
        stepCount = 0;
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

        // Also try to stream the payload to any connected Python listener via SideChannel.
        try
        {
            if (sideChannel != null)
            {
                sideChannel.SendJson(json);
                Debug.Log("TrajectoryLogger: Sent episode payload via SideChannel.");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogFormat("TrajectoryLogger: Failed to send SideChannel payload: {0}", e.Message);
        }
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
        stepCount++;

        // Record positions for all tracked agents
        foreach (var kv in trajectories)
        {
            AgentActions agent = FindAgentByInstanceId(kv.Key);
            if (agent != null)
            {
                kv.Value.positions.Add(new SerializableVector3(agent.transform.position));
            }
        }
    }

    // Reference to the registered side channel (if available)
    private TrajectorySideChannel sideChannel = null;
}

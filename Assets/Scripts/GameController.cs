using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Unity.Barracuda;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.UIElements;

public class GameController : MonoBehaviour
{
    [SerializeField] private int episodeSteps = 1000;
    [SerializeField] private float gracePeriodFraction = 0.4f;
    [SerializeField] private float coneAngle = 0.375f * 180f;

    [SerializeField] private MapGenerator mapGenerator = null;


    [Serializable]
    public struct RewardInfo
    {
        public enum Type { VisibilityIndividual, VisibilityTeam, Capture, OobPenalty };
        public Type type;
        public float weight;
    };

    public enum WinCondition { None, LineOfSight, Capture };

    [Header("Game Rules")]
    [SerializeField] private List<RewardInfo> rewards = null;
    [SerializeField] private WinCondition winCondition = WinCondition.None;
    [SerializeField] private float winConditionRewardMultiplier = 1.0f;
    [SerializeField] private float arenaSize = 20f;

    [SerializeField] private BoxHolding[] boxes;
    [SerializeField] private bool allowCapture = false;
    [SerializeField] private float captureDistance = 1.2f;
    [SerializeField] private int seekersCaptureGoal = 2;

    [Header("Coplay")]
    [SerializeField] private bool useCoplay = false;
    [SerializeField] private int numberOfCoplayAgents = 1;
    [SerializeField] private float selfPlayRatio = 0.5f;

    [Header("Inference")]
    [SerializeField] private bool inferenceMode = false;
    [SerializeField] private NNModel[] hidersCheckpoints = null;
    [SerializeField] private NNModel[] seekersCheckpoints = null;

    [Header("Debug")]
    [SerializeField] private bool debugDrawBoxHold = true;
    [SerializeField] private bool debugDrawVisibility = true;
    [SerializeField] private bool debugDrawIndividualReward = true;
    [SerializeField] private bool debugDrawPlayAreaBounds = true;
    [SerializeField] private bool debugLogPlatformParams = true;
    [SerializeField] private bool debugLogMatchResult = false;
    [SerializeField] private bool debugLogCoplay = false;


    private int episodeTimer = 0;
    private List<AgentActions> hiders;
    private List<AgentActions> seekers;
    private List<AgentActions> hiderInstances;
    private List<AgentActions> seekerInstances;
    private SimpleMultiAgentGroup hidersGroup;
    private SimpleMultiAgentGroup seekersGroup;
    private List<BoxHolding> holdObjects;

    private bool[,] visibilityMatrix;
    private bool[] visibilityHiders;
    private bool[] visibilitySeekers;
    private bool allHidden;

    private int stepsHidden = 0;
    private int hidersCaptured = 0;
    private bool hidersPerfectGame = true;
    private StatsRecorder statsRecorder = null;

    // Events for external systems to subscribe to (e.g., trajectory logger)
    public event Action EpisodeStarted;
    public event Action EpisodeEnded;

    public bool GracePeriodEnded
    {
        get { return episodeTimer >= episodeSteps * gracePeriodFraction; }
    }

    public bool DebugDrawBoxHold => debugDrawBoxHold;
    public bool DebugDrawIndividualReward => debugDrawIndividualReward;


    private void Awake()
    {
        // Ensure Unity keeps running even when the editor window loses focus.
        // This prevents the environment from pausing when you switch to another app
        // (helps avoid trainer worker restarts / hangs while training).
        Application.runInBackground = true;

        if (SystemArgs.GameParamsPath != null)
        {
            Debug.Log("Game controller configuration file: " + SystemArgs.GameParamsPath);
            string content = File.ReadAllText(SystemArgs.GameParamsPath);
            JsonUtility.FromJsonOverwrite(content, this);
        }

        // Notify subscribers that a new episode has started
        EpisodeStarted?.Invoke();
    }

    private void Start()
    {
        if (debugLogPlatformParams)
        {
            Console.WriteLine(gameObject.name);
            Console.WriteLine(JsonUtility.ToJson(this, true));
            if (mapGenerator != null)
            {
                Console.WriteLine(JsonUtility.ToJson(mapGenerator, true));
            }
            Console.WriteLine();
        }

        mapGenerator.Initialize();
        hiderInstances = mapGenerator.GetInstantiatedHiders();
        seekerInstances = mapGenerator.GetInstantiatedSeekers();
        boxes = mapGenerator.GetInstantiatedBoxes();
        hiderInstances.ForEach(hider => hider.GameController = this);
        seekerInstances.ForEach(seeker => seeker.GameController = this);

        if (inferenceMode)
        {
            for (int i = 0; i < hiderInstances.Count; i++)
            {
                if (i < hidersCheckpoints.Length && hidersCheckpoints[i] != null)
                {
                    hiderInstances[i].SwitchToInference(hidersCheckpoints[i]);
                }
            }
            for (int i = 0; i < seekerInstances.Count; i++)
            {
                if (i < seekersCheckpoints.Length && seekersCheckpoints[i] != null)
                {
                    seekerInstances[i].SwitchToInference(seekersCheckpoints[i]);
                }
            }
        }

        hidersGroup = new SimpleMultiAgentGroup();
        seekersGroup = new SimpleMultiAgentGroup();

        holdObjects = FindObjectsOfType<BoxHolding>().ToList();

        statsRecorder = Academy.Instance.StatsRecorder;

        ResetScene();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.P))
        {
            ResetScene();
        }
    }

    private void FixedUpdate()
    {
        episodeTimer++;

        Vector3 seekersMeanPosition = Vector3.zero;
        foreach (AgentActions seeker in seekers)
        {
            seekersMeanPosition += seeker.transform.position - transform.position;
        }
        seekersMeanPosition /= Mathf.Max(1, seekers.Count);
        statsRecorder.Add("Environment/SeekersMeanX", seekersMeanPosition.x);
        statsRecorder.Add("Environment/SeekersMeanZ", seekersMeanPosition.z);

        Vector3 hidersMeanPosition = Vector3.zero;
        foreach (AgentActions hider in hiders)
        {
            hidersMeanPosition += hider.transform.position - transform.position;
        }
        hidersMeanPosition /= Mathf.Max(1, hiders.Count);
        statsRecorder.Add("Environment/HidersMeanX", hidersMeanPosition.x);
        statsRecorder.Add("Environment/HidersMeanZ", hidersMeanPosition.z);

        // Log block positions (unchanged)
        for (int i = 0; i < boxes.Length; i++)
        {
            if (boxes[i] == null) 
            {
                print("Box " + i + " is null, box reference is destroyed or missing");
                continue;
            }
            Vector3 p = boxes[i].transform.position;

            // Log each axis separately so they appear correctly in TensorBoard / CSV
            statsRecorder.Add("Blocks/Block" + i + "_X", p.x, StatAggregationMethod.Average);
            statsRecorder.Add("Blocks/Block" + i + "_Y", p.y, StatAggregationMethod.Average);
            statsRecorder.Add("Blocks/Block" + i + "_Z", p.z, StatAggregationMethod.Average);
            statsRecorder.Add("Blocks/Block" + i + "_IsLocked", boxes[i].LockOwner != null ? 1 : 0);
            statsRecorder.Add("Blocks/Block" + i + "_IsHeld", boxes[i].Owner != null ? 1 : 0);
        }

        // -----------------------
        // NEW: Log agent positions + agent name in stat key
        // -----------------------
        // Hiders
        for (int i = 0; i < hiders.Count; i++)
        {
            var agent = hiders[i];
            if (agent == null) continue;
            Vector3 p = agent.transform.position;
            string nameSafe = SanitizeName(agent.gameObject.name);
            string prefix = $"Agents/Hider_{i}_{nameSafe}";
            statsRecorder.Add(prefix + "_X", p.x, StatAggregationMethod.Average);
            statsRecorder.Add(prefix + "_Y", p.y, StatAggregationMethod.Average);
            statsRecorder.Add(prefix + "_Z", p.z, StatAggregationMethod.Average);
            statsRecorder.Add(prefix + "_IsActive", agent.gameObject.activeInHierarchy ? 1 : 0);
        }

        // Seekers
        for (int i = 0; i < seekers.Count; i++)
        {
            var agent = seekers[i];
            if (agent == null) continue;
            Vector3 p = agent.transform.position;
            string nameSafe = SanitizeName(agent.gameObject.name);
            string prefix = $"Agents/Seeker_{i}_{nameSafe}";
            statsRecorder.Add(prefix + "_X", p.x, StatAggregationMethod.Average);
            statsRecorder.Add(prefix + "_Y", p.y, StatAggregationMethod.Average);
            statsRecorder.Add(prefix + "_Z", p.z, StatAggregationMethod.Average);
            statsRecorder.Add(prefix + "_IsActive", agent.gameObject.activeInHierarchy ? 1 : 0);
        }
        // -----------------------

        UpdateRewards();

        if (episodeTimer >= episodeSteps)
        {
            EndEpisode();
        }
        else if (GracePeriodEnded)
        {
            FillVisibilityMatrix();
            stepsHidden += allHidden ? 1 : 0;
            if (!allHidden)
            {
                hidersPerfectGame = false;
            }

            if (allowCapture)
            {
                for (int i = 0; i < seekers.Count; i++)
                {
                    for (int j = 0; j < hiders.Count; j++)
                    {
                        if (!hiders[j].WasCaptured && Vector3.Distance(seekers[i].transform.position, hiders[j].transform.position) < captureDistance)
                        {
                            hidersCaptured++;
                            hiders[j].WasCaptured = true;
                            foreach (RewardInfo rewardInfo in rewards)
                            {
                                if (rewardInfo.type == RewardInfo.Type.Capture)
                                {
                                    seekers[i].HideAndSeekAgent.AddReward(rewardInfo.weight);
                                    hiders[j].HideAndSeekAgent.AddReward(-rewardInfo.weight);
                                }
                            }

                            if (hidersCaptured < hiders.Count)
                            {
                                hiders[j].gameObject.SetActive(false);
                            }
                            else
                            {
                                EndEpisode();
                            }
                        }
                    }
                }
            }
        }
    }


    public IEnumerable<AgentActions> GetHiders()
    {
        foreach (AgentActions agent in hiders)
        {
            yield return agent;
        }
    }

    public IEnumerable<AgentActions> GetSeekers()
    {
        foreach (AgentActions agent in seekers)
        {
            yield return agent;
        }
    }


    private void EndEpisode()
    {
        float timeHidden = GracePeriodEnded ? stepsHidden / Mathf.Ceil(episodeTimer - episodeSteps * gracePeriodFraction) : 0.0f;
        statsRecorder.Add("Environment/TimeHidden", timeHidden);

        if (allowCapture)
        {
            statsRecorder.Add("Environment/HidersCaptured", (float)hidersCaptured / hiders.Count());
        }

        if (winCondition != WinCondition.None)
        {
            bool hidersWon = false;
            if (winCondition == WinCondition.LineOfSight && hidersPerfectGame)
            {
                hidersWon = true;
            }
            if (winCondition == WinCondition.Capture && hidersCaptured < seekersCaptureGoal)
            {
                hidersWon = true;
            }

            hidersGroup.AddGroupReward(hidersWon ? winConditionRewardMultiplier : -winConditionRewardMultiplier);
            seekersGroup.AddGroupReward(hidersWon ? -winConditionRewardMultiplier : winConditionRewardMultiplier);
            statsRecorder.Add("Environment/HiderWinRatio", hidersWon ? 1 : 0);

            if (debugLogMatchResult)
            {
                switch (winCondition)
                {
                    case WinCondition.LineOfSight:
                        Debug.LogFormat("Team {0} won; Time percentage hidden - {1}", hidersWon ? "hiders" : "seekers", timeHidden * 100f);
                        break;

                    case WinCondition.Capture:
                        Debug.LogFormat("Team {0} won; Hiders captured - {1} / {2}", hidersWon ? "hiders" : "seekers", hidersCaptured, hiders.Count());
                        break;
                }
            }
        }

        // Notify subscribers before actually ending the group episodes (so cumulative rewards are still available)
        EpisodeEnded?.Invoke();

        hidersGroup.EndGroupEpisode();
        seekersGroup.EndGroupEpisode();
        ResetScene();
    }

    private void ResetScene()
    {
        stepsHidden = 0;
        hidersPerfectGame = true;
        hidersCaptured = 0;
        episodeTimer = 0;

        if (!mapGenerator.InstantiatesBoxesOnReset())
        {
            foreach (BoxHolding holdObject in holdObjects)
            {
                holdObject.Reset();
            }
        }

        mapGenerator.Generate();
        hiders = hiderInstances.Take(mapGenerator.NumHiders).ToList();
        seekers = seekerInstances.Take(mapGenerator.NumSeekers).ToList();
        foreach (AgentActions hider in hiders)
        {
            hider.ResetAgent();
            hidersGroup.RegisterAgent(hider.HideAndSeekAgent);
        }
        foreach (AgentActions seeker in seekers)
        {
            seeker.ResetAgent();
            seekersGroup.RegisterAgent(seeker.HideAndSeekAgent);
        }

        for (int i = 0; i < hiderInstances.Count; i++)
        {
            hiderInstances[i].gameObject.SetActive(i < hiders.Count);
        }
        for (int i = 0; i < seekerInstances.Count; i++)
        {
            seekerInstances[i].gameObject.SetActive(i < seekers.Count);
        }

        visibilityMatrix = new bool[hiders.Count, seekers.Count];
        visibilityHiders = new bool[hiders.Count];
        visibilitySeekers = new bool[seekers.Count];
        allHidden = false;

        if (!inferenceMode && useCoplay)
        {
            CoplayManager.Instance.Rescan();
            foreach (AgentActions agent in hiders)
            {
                agent.SwitchToTraining();
            }
            foreach (AgentActions agent in seekers)
            {
                agent.SwitchToTraining();
            }

            if (UnityEngine.Random.Range(0f, 1f) < selfPlayRatio)
            {
                int maxBound = 0;
                if (CoplayManager.TrainedTeamID == 0)
                {
                    maxBound = hiders.Count;
                }
                else
                {
                    maxBound = seekers.Count;
                }
                var numbers = Enumerable.Range(0, maxBound).ToList();
                //shuffle
                for (int i = numbers.Count - 1; i > 0; i--)
                {
                    int j = UnityEngine.Random.Range(0, i + 1);
                    int tmp = numbers[i];
                    numbers[i] = numbers[j];
                    numbers[j] = tmp;
                }
                var randomPos = numbers.Take(numberOfCoplayAgents).ToList();
                for (int i = 0; i < numberOfCoplayAgents; i++)
                {
                    NNModel model = CoplayManager.Instance.GetRandomModel();
                    if (model != null)
                    {
                        (CoplayManager.TrainedTeamID == 0 ? hiders[randomPos[i]] : seekers[randomPos[i]]).SwitchToInference(model);
                        if (debugLogCoplay)
                        {
                            Debug.LogFormat("{0}, team {1}, agent {2}, set model = {3}",
                                             gameObject.name, CoplayManager.TrainedTeamID, randomPos[i],
                                             model.name.Split("/").Last());
                        }
                    }
                }
            }
        }
    }


    private bool AgentSeesAgent(AgentActions agent1, AgentActions agent2, out RaycastHit hit)
    {
        Vector3 direction = agent2.transform.position - agent1.transform.position;
        if (Vector3.Angle(direction, agent1.transform.forward) > coneAngle)
        {
            hit = new RaycastHit();
            return false;
        }

        Ray ray = new Ray(agent1.transform.position, direction);
        if (Physics.Raycast(ray, out hit))
        {
            if (hit.collider.gameObject == agent2.gameObject)
            {
                return true;
            }
        }
        return false;
    }

    private void FillVisibilityMatrix()
    {
        allHidden = true;
        Array.Fill(visibilityHiders, false);
        Array.Fill(visibilitySeekers, false);

        for (int i = 0; i < hiders.Count; i++)
        {
            for (int j = 0; j < seekers.Count; j++)
            {
                if (AgentSeesAgent(seekers[j], hiders[i], out RaycastHit hit))
                {
                    visibilityMatrix[i, j] = true;
                    visibilityHiders[i] = true;
                    visibilitySeekers[j] = true;
                    allHidden = false;
                    if (debugDrawVisibility)
                    {
                        Debug.DrawLine(seekers[j].transform.position, hit.point, Color.red);
                    }
                }
                else
                {
                    visibilityMatrix[i, j] = false;
                }
            }
        }
    }

    private void UpdateRewards()
    {
        foreach (RewardInfo rewardInfo in rewards)
        {
            switch (rewardInfo.type)
            {
                case RewardInfo.Type.VisibilityIndividual:
                    if (!GracePeriodEnded) break;
                    for (int i = 0; i < hiders.Count(); i++)
                    {
                        float reward = visibilityHiders[i] ? -rewardInfo.weight : rewardInfo.weight;
                        if (allowCapture && hiders[i].WasCaptured)
                        {
                            reward = 0f;
                        }
                        hiders[i].HideAndSeekAgent.AddReward(reward);
                    }
                    for (int i = 0; i < seekers.Count(); i++)
                    {
                        float reward = visibilitySeekers[i] ? rewardInfo.weight : -rewardInfo.weight;
                        if (allowCapture && hidersCaptured == hiders.Count())
                        {
                            reward = 0f;
                        }
                        seekers[i].HideAndSeekAgent.AddReward(reward);
                    }
                    break;

                case RewardInfo.Type.VisibilityTeam:
                    if (!GracePeriodEnded) break;
                    float hidersReward = allHidden ? rewardInfo.weight : -rewardInfo.weight;
                    hiders.ForEach((AgentActions hider) => hider.HideAndSeekAgent.AddReward(hidersReward));
                    seekers.ForEach((AgentActions seeker) => seeker.HideAndSeekAgent.AddReward(-hidersReward));
                    break;

                case RewardInfo.Type.OobPenalty:
                    foreach (AgentActions hider in hiders.Where((AgentActions agent) => IsOoB(agent)))
                    {
                        hider.HideAndSeekAgent.AddReward(-rewardInfo.weight);
                    }
                    foreach (AgentActions seeker in seekers.Where((AgentActions agent) => IsOoB(agent)))
                    {
                        seeker.HideAndSeekAgent.AddReward(-rewardInfo.weight);
                    }
                    break;
            }
        }
    }

    private bool IsOoB(AgentActions agent)
    {
        return Mathf.Max(Mathf.Abs(agent.transform.position.x - transform.position.x),
                         Mathf.Abs(agent.transform.position.z - transform.position.z)) > arenaSize * 0.5f;
    }


    private void OnDrawGizmos()
    {
        if (debugDrawPlayAreaBounds)
        {
            Vector3 center = transform.position;
            Color c = Gizmos.color;
            Gizmos.color = new Color(0.3f, 1f, 0.3f, 0.5f);
            Gizmos.DrawCube(center + new Vector3(-arenaSize * 0.5f, 1f, 0f), new Vector3(0.25f, 2f, arenaSize));
            Gizmos.DrawCube(center + new Vector3(+arenaSize * 0.5f, 1f, 0f), new Vector3(0.25f, 2f, arenaSize));
            Gizmos.DrawCube(center + new Vector3(0f, 1f, -arenaSize * 0.5f), new Vector3(arenaSize, 2f, 0.25f));
            Gizmos.DrawCube(center + new Vector3(0f, 1f, +arenaSize * 0.5f), new Vector3(arenaSize, 2f, 0.25f));
            Gizmos.color = c;
        }
    }

    /// <summary>
    /// Replace any characters that could cause odd stat keys. Keep letters, digits, underscore and dash.
    /// Collapses whitespace and removes other punctuation.
    /// </summary>
    private string SanitizeName(string name)
    {
        if (string.IsNullOrEmpty(name)) return "unnamed";
        // replace whitespace with underscore
        string s = Regex.Replace(name, @"\s+", "_");
        // keep letters, digits, underscore and dash only
        s = Regex.Replace(s, @"[^A-Za-z0-9_\-]", "");
        // safety: trim length
        if (s.Length > 32) s = s.Substring(0, 32);
        return s;
    }
}

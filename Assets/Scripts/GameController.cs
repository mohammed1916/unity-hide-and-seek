using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Unity.Barracuda;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.UIElements;
// using System.Globalization;
public class GameController : MonoBehaviour
{
    [SerializeField] private int episodeSteps = 5000;
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
    public enum SuccessPerspective { Seekers, Hiders };

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

    [Header("Outcome")]
    [SerializeField] private SuccessPerspective successPerspective = SuccessPerspective.Seekers;


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

    // For CSV logging
    private string runId;
    private string basePath;
    private string episodePath;
    private Dictionary<string, StreamWriter> agentWriters = new();
    private StreamWriter blockWriter;
    private StreamWriter episodeMetaWriter;
    private StreamWriter combineWriter;
    private int episodeIndex = 0;

    // private bool shouldLogEpisode = false;
    private bool shouldLogEpisode = true;


    private void Awake()
    {
        Application.runInBackground = true;

        // Read run-id from command-line
        // runId = "default_run";
        runId = "run41_10_5000_steps";
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++){
            if (args[i] == "--run-id" && i + 1 < args.Length)
                runId = args[i + 1];
        }
        print("runId: " + runId);
        print("shouldLogEpisode: " + shouldLogEpisode);

        // Base folder using Directory.GetCurrentDirectory()
        basePath = Path.Combine(Directory.GetCurrentDirectory(), "trajectory_logs_", runId);
        Directory.CreateDirectory(basePath);

        // Create or open combined CSV file in run root
        if (shouldLogEpisode)
        {
            try
            {
                string combineFile = Path.Combine(basePath, "combine.csv");
                bool writeHeader = !File.Exists(combineFile);
                combineWriter = new StreamWriter(new FileStream(
                    combineFile,
                    FileMode.Append,
                    FileAccess.Write,
                    FileShare.ReadWrite
                ));
                if (writeHeader)
                {
                    combineWriter.WriteLine("episode_index,episode_steps,episode_duration_steps,episode_duration_seconds,outcome,outcome_seekers,outcome_hiders,winner,hidersCaptured,stepsHidden,timeHidden,hidersPerfectGame,gracePeriodEnded");
                    combineWriter.Flush();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to create/open combine CSV: {e.Message}");
                combineWriter = null;
            }
        }

        // Next episode folder (episodeIndex will be incremented on episode start)
        if (shouldLogEpisode)
        {
            StartNewEpisode();
        }

        if (SystemArgs.GameParamsPath != null)
        {
            Debug.Log("Game configuration file: " + SystemArgs.GameParamsPath);
            JsonUtility.FromJsonOverwrite(File.ReadAllText(SystemArgs.GameParamsPath), this);
        }

        EpisodeStarted?.Invoke();
    }

    private void StartNewEpisode()
    {
        episodeIndex++;
        episodePath = Path.Combine(basePath, $"episode_{episodeIndex}");
        Directory.CreateDirectory(episodePath);

        // Close previous writers
        foreach (var w in agentWriters.Values) w.Close();
        agentWriters.Clear();
        blockWriter?.Close();

        try
        {
            blockWriter = new StreamWriter(new FileStream(
                Path.Combine(episodePath, "blocks.csv"),
                FileMode.Create,          // overwrite existing file
                FileAccess.Write,
                FileShare.ReadWrite       // allow other readers/writers temporarily
            ));
            blockWriter.WriteLine("time,block_id,x,y,z,isLocked,isHeld");
            // Episode meta file - contains a single line describing the episode
            try
            {
                episodeMetaWriter = new StreamWriter(new FileStream(
                    Path.Combine(episodePath, "episode_meta.csv"),
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.ReadWrite
                ));
                episodeMetaWriter.WriteLine("episode_index,episode_steps,episode_duration_steps,episode_duration_seconds,outcome,outcome_seekers,outcome_hiders,winner,hidersCaptured,stepsHidden,timeHidden,hidersPerfectGame,gracePeriodEnded");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to create episode_meta writer: {e.Message}");
                episodeMetaWriter = null;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to create block writer: {e.Message}");
            blockWriter = null;
        }
    }
    
    private void LogBlock(int id, Vector3 p, float locked, float held)
    {
        // blockWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
        // "{0:F4},Block{1},{2:F4},{3:F4},{4:F4},{5},{6}",
        // Time.time, id, p.x, p.y, p.z, locked, held));
        blockWriter.WriteLine($"{Time.time:F4},Block{id},{p.x:F4},{p.y:F4},{p.z:F4},{locked},{held}");

    }

    private void LogAgent(string agentId, Vector3 p, float active)
    {
        if (!agentWriters.ContainsKey(agentId))
        {
            try
            {
                string file = Path.Combine(episodePath, $"agent_{agentId}.csv");
                var writer = new StreamWriter(new FileStream(
                    file,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.ReadWrite
                ));
                writer.WriteLine("time,x,y,z,isActive");
                agentWriters.Add(agentId, writer);
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to create agent writer for {agentId}: {e.Message}");
                return;
            }
        }

        try
        {
            // agentWriters[agentId].WriteLine(
            //     string.Format(CultureInfo.InvariantCulture,
            //     "{0:F4},{1:F4},{2:F4},{3:F4},{4}",
            //     Time.time, p.x, p.y, p.z, active));
            agentWriters[agentId].WriteLine($"{Time.time:F4},{p.x:F4},{p.y:F4},{p.z:F4},{active}");

        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to write to agent {agentId}: {e.Message}");
        }
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

        holdObjects = FindObjectsByType<BoxHolding>(FindObjectsSortMode.None).ToList();

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

        // // Log block positions (unchanged)
        // for (int i = 0; i < boxes.Length; i++)
        // {
        //     if (boxes[i] == null) 
        //     {
        //         print("Box " + i + " is null, box reference is destroyed or missing");
        //         continue;
        //     }
        //     Vector3 p = boxes[i].transform.position;

        //     // Log each axis separately so they appear correctly in TensorBoard / CSV
        //     statsRecorder.Add("Blocks/Block" + i + "_X", p.x, StatAggregationMethod.Average);
        //     statsRecorder.Add("Blocks/Block" + i + "_Y", p.y, StatAggregationMethod.Average);
        //     statsRecorder.Add("Blocks/Block" + i + "_Z", p.z, StatAggregationMethod.Average);
        //     statsRecorder.Add("Blocks/Block" + i + "_IsLocked", boxes[i].LockOwner != null ? 1 : 0);
        //     statsRecorder.Add("Blocks/Block" + i + "_IsHeld", boxes[i].Owner != null ? 1 : 0);
        // }

        // // -----------------------
        // // Log agent positions + agent name in stat key
        // // -----------------------
        // // Hiders
        // for (int i = 0; i < hiders.Count; i++)
        // {
        //     var agent = hiders[i];
        //     if (agent == null) continue;
        //     Vector3 p = agent.transform.position;
        //     string nameSafe = SanitizeName(agent.gameObject.name);
        //     string prefix = $"Agents/Hider_{i}_{nameSafe}";
        //     statsRecorder.Add(prefix + "_X", p.x, StatAggregationMethod.Average);
        //     statsRecorder.Add(prefix + "_Y", p.y, StatAggregationMethod.Average);
        //     statsRecorder.Add(prefix + "_Z", p.z, StatAggregationMethod.Average);
        //     statsRecorder.Add(prefix + "_IsActive", agent.gameObject.activeInHierarchy ? 1 : 0);
        // }

        // // Seekers
        // for (int i = 0; i < seekers.Count; i++)
        // {
        //     var agent = seekers[i];
        //     if (agent == null) continue;
        //     Vector3 p = agent.transform.position;
        //     string nameSafe = SanitizeName(agent.gameObject.name);
        //     string prefix = $"Agents/Seeker_{i}_{nameSafe}";
        //     statsRecorder.Add(prefix + "_X", p.x, StatAggregationMethod.Average);
        //     statsRecorder.Add(prefix + "_Y", p.y, StatAggregationMethod.Average);
        //     statsRecorder.Add(prefix + "_Z", p.z, StatAggregationMethod.Average);
        //     statsRecorder.Add(prefix + "_IsActive", agent.gameObject.activeInHierarchy ? 1 : 0);
        // }
        // // -----------------------

        // // -----------------------
        // Dictionary<string, float> frame = new Dictionary<string, float>();

        // // Blocks
        // for (int i = 0; i < boxes.Length; i++)
        // {
        //     if (boxes[i] == null) continue;
        //     Vector3 p = boxes[i].transform.position;
        //     frame[$"Block{i}_X"] = p.x;
        //     frame[$"Block{i}_Y"] = p.y;
        //     frame[$"Block{i}_Z"] = p.z;
        //     frame[$"Block{i}_IsLocked"] = boxes[i].LockOwner != null ? 1f : 0f;
        //     frame[$"Block{i}_IsHeld"] = boxes[i].Owner != null ? 1f : 0f;
        // }

        // // Hiders
        // for (int i = 0; i < hiders.Count; i++)
        // {
        //     var agent = hiders[i];
        //     if (agent == null) continue;
        //     Vector3 p = agent.transform.position;
        //     string prefix = $"Hider{i}";
        //     frame[$"{prefix}_X"] = p.x;
        //     frame[$"{prefix}_Y"] = p.y;
        //     frame[$"{prefix}_Z"] = p.z;
        //     frame[$"{prefix}_IsActive"] = agent.gameObject.activeInHierarchy ? 1f : 0f;
        // }

        // // Seekers
        // for (int i = 0; i < seekers.Count; i++)
        // {
        //     var agent = seekers[i];
        //     if (agent == null) continue;
        //     Vector3 p = agent.transform.position;
        //     string prefix = $"Seeker{i}";
        //     frame[$"{prefix}_X"] = p.x;
        //     frame[$"{prefix}_Y"] = p.y;
        //     frame[$"{prefix}_Z"] = p.z;
        //     frame[$"{prefix}_IsActive"] = agent.gameObject.activeInHierarchy ? 1f : 0f;
        // }

        // // Commit this frame
        // csvLogger.LogFrame(frame);

        // // -----------------------
        if (shouldLogEpisode)
       { 
            // -----------------------
            // Log blocks
            // -----------------------
            if (boxes != null)
            {
                for (int i = 0; i < boxes.Length; i++)
                {
                    if (boxes[i] == null) continue;
                    Vector3 p = boxes[i].transform.position;
                    LogBlock(i, p, boxes[i].LockOwner != null ? 1f : 0f, boxes[i].Owner != null ? 1f : 0f);
                }
            }

            // -----------------------
            // Log hiders
            // -----------------------
            if (hiders != null)
            {
                for (int i = 0; i < hiders.Count; i++)
                {
                    var agent = hiders[i];
                    if (agent == null) continue;
                    string agentId = $"Hider{i}_{SanitizeName(agent.gameObject.name)}";
                    LogAgent(agentId, agent.transform.position, agent.gameObject.activeInHierarchy ? 1f : 0f);
                }
            }

            // -----------------------
            // Log seekers
            // -----------------------
            if (seekers != null)
            {
                for (int i = 0; i < seekers.Count; i++)
                {
                    var agent = seekers[i];
                    if (agent == null) continue;
                    string agentId = $"Seeker{i}_{SanitizeName(agent.gameObject.name)}";
                    LogAgent(agentId, agent.transform.position, agent.gameObject.activeInHierarchy ? 1f : 0f);
                }
            }

            // -----------------------
            // Flush writers to ensure data is written to disk
            // -----------------------
            try
            {
                blockWriter?.Flush();
            }
            catch (Exception e)
            {
                Debug.LogError($"Error flushing block writer: {e.Message}");
            }
            foreach (var w in agentWriters.Values)
            {
                try
                {
                    w.Flush();
                }
                catch (Exception e)
                {
                    Debug.LogError($"Error flushing agent writer: {e.Message}");
                }
            }
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
                                    float captureReward = rewardInfo.weight;
                                    statsRecorder.Add($"Reward/Capture_Seeker{i}_Hider{j}", captureReward);
                                    seekers[i].HideAndSeekAgent.AddReward(captureReward);
                                    statsRecorder.Add($"Reward/Capture_Hider{j}_Seeker{i}", -captureReward);
                                    hiders[j].HideAndSeekAgent.AddReward(-captureReward);
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

        // Prepare episode metadata summary (row) for both episode_meta and combine
        string winner = "None";
        string outcome = "Timeout"; // default if time limited
        bool timedOut = episodeTimer >= episodeSteps;
        if (timedOut)
        {
            outcome = "Timeout";
        }
        else
        {
            // reuse the logic we used above to compute who won
            bool hidersWon = false;
            if (winCondition == WinCondition.LineOfSight && hidersPerfectGame)
            {
                hidersWon = true;
            }
            if (winCondition == WinCondition.Capture && hidersCaptured < seekersCaptureGoal)
            {
                hidersWon = true;
            }
            // Also if capture is allowed and all hiders captured -> seekers win
            if (allowCapture && hidersCaptured == hiders.Count())
            {
                hidersWon = false;
            }
            winner = hidersWon ? "Hiders" : "Seekers";
        }

        float durationSeconds = Time.fixedDeltaTime * (float)episodeTimer;
        // recompute final per-team outcomes and use configured outcome as the legacy outcome column
        string outcomeSeekersCol = "Timeout";
        string outcomeHidersCol = "Timeout";
        if (!timedOut)
        {
            outcomeSeekersCol = (winner == "Seekers") ? "Success" : "Failure";
            outcomeHidersCol = (winner == "Hiders") ? "Success" : "Failure";
        }
        // compute configured outcome depending on successPerspective
        switch (successPerspective)
        {
            case SuccessPerspective.Hiders:
                outcome = outcomeHidersCol;
                break;
            case SuccessPerspective.Seekers:
            default:
                outcome = outcomeSeekersCol;
                break;
        }
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
            (GracePeriodEnded ? (stepsHidden / Mathf.Ceil(episodeTimer - episodeSteps * gracePeriodFraction)) : 0.0f),
            hidersPerfectGame,
            GracePeriodEnded);

        if (shouldLogEpisode && episodeMetaWriter != null)
        {
            try
            {
                episodeMetaWriter.WriteLine(row);
                episodeMetaWriter.Flush();
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to write episode metadata: {e.Message}");
            }
            try
            {
                episodeMetaWriter?.Close();
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to close episode meta writer: {e.Message}");
            }
            episodeMetaWriter = null;
        }

        // Also append the row into combined CSV (run-level)
        if (shouldLogEpisode && combineWriter != null)
        {
            try
            {
                combineWriter.WriteLine(row);
                combineWriter.Flush();
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to append to combine CSV: {e.Message}");
            }
        }

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
        boxes = mapGenerator.GetInstantiatedBoxes();
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
        if (shouldLogEpisode)
        {
            StartNewEpisode();
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
                        if (reward != 0f)
                        {
                            statsRecorder.Add($"Reward/VisibilityIndividual_Hider{i}", reward);
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
                        if (reward != 0f)
                        {
                            statsRecorder.Add($"Reward/VisibilityIndividual_Seeker{i}", reward);
                        }
                        seekers[i].HideAndSeekAgent.AddReward(reward);
                    }
                    break;

                case RewardInfo.Type.VisibilityTeam:
                    if (!GracePeriodEnded) break;
                    float hidersReward = allHidden ? rewardInfo.weight : -rewardInfo.weight;
                    if (hidersReward != 0f)
                    {
                        statsRecorder.Add("Reward/VisibilityTeam", hidersReward);
                    }
                    hiders.ForEach((AgentActions hider) => hider.HideAndSeekAgent.AddReward(hidersReward));
                    seekers.ForEach((AgentActions seeker) => seeker.HideAndSeekAgent.AddReward(-hidersReward));
                    break;

                case RewardInfo.Type.OobPenalty:
                    foreach (AgentActions hider in hiders.Where((AgentActions agent) => IsOoB(agent)))
                    {
                        float penalty = -rewardInfo.weight;
                        statsRecorder.Add($"Reward/OobPenalty_Hider{hiders.IndexOf(hider)}", penalty);
                        hider.HideAndSeekAgent.AddReward(penalty);
                    }
                    foreach (AgentActions seeker in seekers.Where((AgentActions agent) => IsOoB(agent)))
                    {
                        float penalty = -rewardInfo.weight;
                        statsRecorder.Add($"Reward/OobPenalty_Seeker{seekers.IndexOf(seeker)}", penalty);
                        seeker.HideAndSeekAgent.AddReward(penalty);
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
    private void OnApplicationQuit()
    {
        // Close all writers on application quit to ensure data is flushed
        foreach (var w in agentWriters.Values)
        {
            try
            {
                w.Close();
            }
            catch (Exception e)
            {
                Debug.LogError($"Error closing agent writer: {e.Message}");
            }
        }
        agentWriters.Clear();
        
        try
        {
            blockWriter?.Close();
        }
        catch (Exception e)
        {
            Debug.LogError($"Error closing block writer: {e.Message}");
        }
        try
        {
            episodeMetaWriter?.Close();
        }
        catch (Exception e)
        {
            Debug.LogError($"Error closing episode meta writer: {e.Message}");
        }
        try
        {
            combineWriter?.Close();
        }
        catch (Exception e)
        {
            Debug.LogError($"Error closing combine writer: {e.Message}");
        }
    }
}

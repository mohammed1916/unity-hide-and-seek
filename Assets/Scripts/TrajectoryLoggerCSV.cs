using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class TrajectoryLoggerCSV : MonoBehaviour
{
    private string runId;
    private string rootPath;
    private string episodeFolder;
    private StreamWriter writer;

    private int episodeIndex = 0;
    private bool isReady = false;

    private GameController gameController;

    void Awake()
    {
        // Detect --run-id from command line
        runId = "default_run";
        foreach (string arg in Environment.GetCommandLineArgs())
        {
            if (arg.StartsWith("--run-id="))
            {
                runId = arg.Replace("--run-id=", "");
                break;
            }
        }

        // Create root folder next to Unity Editor or build executable working directory
        rootPath = Path.Combine(Directory.GetCurrentDirectory(), "trajectory_logs", runId);
        Directory.CreateDirectory(rootPath);

        gameController = FindObjectOfType<GameController>();
        gameController.EpisodeStarted += OnEpisodeStart;
        gameController.EpisodeEnded += OnEpisodeEnd;

        isReady = true;
    }

    private void OnEpisodeStart()
    {
        if (!isReady) return;

        episodeFolder = Path.Combine(rootPath, $"episode_{episodeIndex}");
        Directory.CreateDirectory(episodeFolder);

        string csvPath = Path.Combine(episodeFolder, "trajectory.csv");
        writer = new StreamWriter(csvPath, false);

        // CSV headers
        writer.WriteLine(
            "step,team,agent_id,pos_x,pos_y,pos_z,rot_y,vel_x,vel_y,vel_z,was_captured," +
            "box_id,box_pos_x,box_pos_y,box_pos_z,is_locked,is_held"
        );
    }

    public void LogStep(int step, string team, int agentId,
        Vector3 pos, float rotY, Vector3 vel, float wasCaptured,
        int boxId, Vector3 boxPos, int locked, int held)
    {
        if (!isReady || writer == null) return;

        writer.WriteLine(
            $"{step},{team},{agentId},{pos.x},{pos.y},{pos.z},{rotY},{vel.x},{vel.y},{vel.z},{wasCaptured}," +
            $"{boxId},{boxPos.x},{boxPos.y},{boxPos.z},{locked},{held}"
        );
    }

    private void OnEpisodeEnd()
    {
        if (writer != null)
        {
            writer.Flush();
            writer.Close();
        }

        episodeIndex++;
    }

    private void OnApplicationQuit()
    {
        if (writer != null)
        {
            writer.Flush();
            writer.Close();
        }
    }
}

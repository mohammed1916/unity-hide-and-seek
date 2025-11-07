using System;
using Unity.MLAgents.SideChannels;

// Small SideChannel to send JSON payloads from Unity to a Python listener.
public class TrajectorySideChannel : SideChannel
{
    // Use a stable GUID. Change this only if you also update the Python listener.
    public static readonly Guid ChannelId = new Guid("11111111-2222-3333-4444-555555555555");

    public TrajectorySideChannel() : base(ChannelId) { }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // This channel is primarily for Unity->Python, so ignore incoming messages for now.
        // If needed, read strings/bytes from msg here.
    }

    // Convenience sender for JSON strings
    public void SendJson(string json)
    {
        using (var msg = new OutgoingMessage())
        {
            msg.WriteString(json);
            QueueMessageToSend(msg);
        }
    }
}

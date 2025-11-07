using System;
using Unity.MLAgents.SideChannels;

// Small SideChannel to send JSON payloads from Unity to a Python listener.
public class TrajectorySideChannel : SideChannel
{
    // Use a stable GUID. Change this only if you also update the Python listener.
    public static readonly Guid ChannelId = new Guid("11111111-2222-3333-4444-555555555555");

    public TrajectorySideChannel() : base()
    {
        // Some ML-Agents versions expose a SideChannel(Guid) constructor.
        // Others only have a parameterless constructor and keep the GUID in a private field.
        // To be compatible with both, try to set a private Guid field via reflection if present.
        try
        {
            var scType = typeof(SideChannel);
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            System.Reflection.FieldInfo guidField = null;
            // common field name guesses
            string[] candidates = new[] { "m_ChannelId", "channelId", "m_Guid", "m_guid", "m_Id" };
            foreach (var name in candidates)
            {
                guidField = scType.GetField(name, flags);
                if (guidField != null && guidField.FieldType == typeof(Guid)) break;
            }
            if (guidField == null)
            {
                // fallback: find any non-public Guid field
                foreach (var f in scType.GetFields(flags))
                {
                    if (f.FieldType == typeof(Guid)) { guidField = f; break; }
                }
            }
            if (guidField != null)
            {
                guidField.SetValue(this, ChannelId);
            }
            else
            {
                // Try property setter if available
                var prop = scType.GetProperty("ChannelId", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (prop != null && prop.CanWrite && prop.PropertyType == typeof(Guid))
                {
                    prop.SetValue(this, ChannelId);
                }
            }
        }
        catch (System.Exception)
        {
            // If reflection fails, ignore; SideChannel registration may still work for some versions.
        }
    }

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

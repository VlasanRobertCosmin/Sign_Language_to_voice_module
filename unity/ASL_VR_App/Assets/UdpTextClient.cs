using UnityEngine;
using System.Net.Sockets;
using System.Text;

public class UdpTextClient : MonoBehaviour
{
    [Header("Python UDP Server")]
    public string pythonIp = "127.0.0.1";
    public int pythonPort = 5005;

    private UdpClient udpClient;

    void Start()
    {
        udpClient = new UdpClient();
        Debug.Log("UDP Text Client started");
    }

    public void SendTextToPython(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            Debug.LogWarning("Empty text, nothing sent.");
            return;
        }

        string safeText = text.Replace("\\", "\\\\").Replace("\"", "\\\"");
        string json = "{\"type\":\"text\",\"text\":\"" + safeText + "\"}";

        byte[] data = Encoding.UTF8.GetBytes(json);
        udpClient.Send(data, data.Length, pythonIp, pythonPort);

        Debug.Log("Sent to Python: " + json);
    }

    public void SendSignToPython(string label)
    {
        if (string.IsNullOrWhiteSpace(label))
        {
            Debug.LogWarning("Empty label, nothing sent.");
            return;
        }

        string safeLabel = label.Replace("\\", "\\\\").Replace("\"", "\\\"");
        string json = "{\"type\":\"sign\",\"label\":\"" + safeLabel + "\"}";

        byte[] data = Encoding.UTF8.GetBytes(json);
        udpClient.Send(data, data.Length, pythonIp, pythonPort);

        Debug.Log("Sent sign to Python: " + json);
    }

    void OnApplicationQuit()
    {
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }
    }
}
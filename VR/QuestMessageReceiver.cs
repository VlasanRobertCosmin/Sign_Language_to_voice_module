using UnityEngine;
using TMPro;
using NativeWebSocket;

public class QuestMessageReceiver : MonoBehaviour
{
    [Header("Connection")]
    public string serverIp = "192.168.0.106";
    public int serverPort = 8765;

    [Header("UI")]
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI receivedText;

    private WebSocket websocket;

    async void Start()
    {
        string url = $"ws://{serverIp}:{serverPort}";
        websocket = new WebSocket(url);

        websocket.OnOpen += () =>
        {
            Debug.Log("WebSocket connected");
            if (statusText != null)
                statusText.text = "Connected";
        };

        websocket.OnError += (e) =>
        {
            Debug.LogError("WebSocket error: " + e);
            if (statusText != null)
                statusText.text = "Error";
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log("WebSocket closed");
            if (statusText != null)
                statusText.text = "Disconnected";
        };

        websocket.OnMessage += (bytes) =>
        {
            string message = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log("Received: " + message);

            if (receivedText != null)
                receivedText.text = "Received: " + message;
        };

        if (statusText != null)
            statusText.text = "Connecting...";

        await websocket.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
#endif
    }

    async void OnApplicationQuit()
    {
        if (websocket != null)
            await websocket.Close();
    }
}
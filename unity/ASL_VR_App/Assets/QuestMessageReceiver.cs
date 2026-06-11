using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using TMPro;
using UnityEngine;

public class QuestUdpReceiver : MonoBehaviour
{
    [Header("UDP Settings")]
    public int listenPort = 5051;

    [Header("UI")]
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI receivedText;

    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool isRunning = false;

    private readonly object messageLock = new object();
    private string latestMessage = "Waiting for UDP message...";
    private string latestStatus = "Starting UDP listener...";

    void Start()
    {
        try
        {
            udpClient = new UdpClient(listenPort);
            isRunning = true;

            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            latestStatus = $"Listening on UDP port {listenPort}";
            Debug.Log($"[QuestUdpReceiver] Listening on UDP port {listenPort}");
        }
        catch (Exception ex)
        {
            latestStatus = $"UDP start error: {ex.Message}";
            Debug.LogError("[QuestUdpReceiver] " + ex);
        }
    }

    void Update()
    {
        lock (messageLock)
        {
            if (statusText != null)
                statusText.text = latestStatus;

            if (receivedText != null)
                receivedText.text = latestMessage;
        }
    }

    private void ReceiveLoop()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

        while (isRunning)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);

                lock (messageLock)
                {
                    latestMessage = "Received: " + message;
                    latestStatus = $"UDP OK ({remoteEndPoint.Address}:{remoteEndPoint.Port})";
                }

                Debug.Log($"[QuestUdpReceiver] Received from {remoteEndPoint.Address}:{remoteEndPoint.Port} -> {message}");
            }
            catch (SocketException)
            {
                // Happens when closing socket, safe to ignore if shutting down
                if (isRunning)
                {
                    lock (messageLock)
                    {
                        latestStatus = "UDP socket exception";
                    }
                }
            }
            catch (Exception ex)
            {
                lock (messageLock)
                {
                    latestStatus = "UDP error: " + ex.Message;
                }

                Debug.LogError("[QuestUdpReceiver] " + ex);
            }
        }
    }

    void OnApplicationQuit()
    {
        Shutdown();
    }

    void OnDestroy()
    {
        Shutdown();
    }

    private void Shutdown()
    {
        isRunning = false;

        try
        {
            udpClient?.Close();
            udpClient = null;
        }
        catch { }

        try
        {
            if (receiveThread != null && receiveThread.IsAlive)
                receiveThread.Join(200);
        }
        catch { }
    }
}
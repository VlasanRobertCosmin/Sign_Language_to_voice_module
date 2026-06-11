using UnityEngine;

public class ControllerSendTest : MonoBehaviour
{
    public UdpTextClient udpClient;

    public string messageToSend = "hello thank you";

    void Update()
    {
        // Right controller trigger
        if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
        {
            SendMessageToPython();
        }

        // A button
        if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch))
        {
            SendMessageToPython();
        }
    }

    public void SendMessageToPython()
    {
        if (udpClient == null)
        {
            Debug.LogError("UdpTextClient is not assigned.");
            return;
        }

        udpClient.SendTextToPython(messageToSend);
        Debug.Log("Controller sent: " + messageToSend);
    }
}
using UnityEngine;
using TMPro;

public class VrTextInputController : MonoBehaviour
{
    [Header("UI")]
    public TMP_InputField inputField;

    [Header("UDP Client")]
    public UdpTextClient udpClient;

    public void SendInputText()
    {
        if (inputField == null)
        {
            Debug.LogError("TMP InputField is not assigned.");
            return;
        }

        if (udpClient == null)
        {
            Debug.LogError("UdpTextClient is not assigned.");
            return;
        }

        string text = inputField.text;

        udpClient.SendTextToPython(text);

        inputField.text = "";
        inputField.ActivateInputField();
    }

    public void SendHello()
    {
        if (udpClient != null)
        {
            udpClient.SendSignToPython("hello");
        }
    }

    public void SendThankYou()
    {
        if (udpClient != null)
        {
            udpClient.SendTextToPython("thank you");
        }
    }
}
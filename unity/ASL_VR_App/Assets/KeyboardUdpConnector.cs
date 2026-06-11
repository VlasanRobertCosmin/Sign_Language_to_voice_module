using System;
using Microsoft.MixedReality.Toolkit.Experimental.UI;
using UnityEngine;

/// <summary>
/// Bridges the NonNativeKeyboard with the UdpTextClient.
/// Call OpenKeyboard() (e.g. from a UI button) to present the VR keyboard.
/// When the user presses Enter the typed text is sent via UDP and the keyboard closes.
/// </summary>
public class KeyboardUdpConnector : MonoBehaviour
{
    [Header("Dependencies")]
    [Tooltip("Reference to the UdpTextClient in the scene.")]
    public UdpTextClient udpClient;

    [Tooltip("Optional: pre-fill the keyboard with this text when opened.")]
    public string initialText = "";

    private NonNativeKeyboard _keyboard;

    private void Awake()
    {
        _keyboard = NonNativeKeyboard.Instance;

        if (_keyboard == null)
        {
            Debug.LogError("[KeyboardUdpConnector] NonNativeKeyboard.Instance is null. Make sure the NonNativeKeyboard prefab is in the scene.");
            enabled = false;
            return;
        }

        _keyboard.OnTextSubmitted += HandleTextSubmitted;
    }

    private void OnDestroy()
    {
        if (_keyboard != null)
        {
            _keyboard.OnTextSubmitted -= HandleTextSubmitted;
        }
    }

    /// <summary>
    /// Opens the VR keyboard. Wire this to a world-space UI button's OnClick event.
    /// </summary>
    public void OpenKeyboard()
    {
        if (_keyboard == null)
        {
            Debug.LogError("[KeyboardUdpConnector] Cannot open keyboard — NonNativeKeyboard.Instance is null.");
            return;
        }

        if (string.IsNullOrEmpty(initialText))
        {
            _keyboard.PresentKeyboard();
        }
        else
        {
            _keyboard.PresentKeyboard(initialText);
        }
    }

    /// <summary>
    /// Handles the Enter key press on the keyboard: sends the text via UDP and closes the keyboard.
    /// </summary>
    private void HandleTextSubmitted(object sender, EventArgs e)
    {
        var keyboard = sender as NonNativeKeyboard;
        if (keyboard == null) return;

        string text = keyboard.InputField.text;

        if (udpClient == null)
        {
            Debug.LogError("[KeyboardUdpConnector] UdpTextClient is not assigned.");
            return;
        }

        udpClient.SendTextToPython(text);
        Debug.Log($"[KeyboardUdpConnector] Sent via UDP: {text}");

        keyboard.gameObject.SetActive(false);
    }
}

using UnityEngine;
using TMPro;

public class HandDebugManager : MonoBehaviour
{
    [Header("Hand References")]
    public OVRHand leftHand;
    public OVRHand rightHand;

    [Header("UI")]
    public TextMeshProUGUI debugText;

    void Start()
    {
        Debug.Log("[HandDebugManager] Start");

        if (debugText != null)
        {
            debugText.text = "STARTED\n";
        }
        else
        {
            Debug.LogWarning("[HandDebugManager] debugText is NULL");
        }

        DebugReferenceState();
    }

    void Update()
    {
        if (debugText == null)
            return;

        string leftInfo = GetHandInfo("Left", leftHand);
        string rightInfo = GetHandInfo("Right", rightHand);

        debugText.text =
            $"FRAME: {Time.frameCount}\n" +
            $"{leftInfo}\n\n" +
            $"{rightInfo}";
    }

    string GetHandInfo(string label, OVRHand hand)
    {
        if (hand == null)
            return $"{label}: NULL";

        string tracked = hand.IsTracked ? "True" : "False";
        string confidence = hand.HandConfidence.ToString();
        string pointerValid = hand.IsPointerPoseValid ? "True" : "False";

        return
            $"{label} Hand\n" +
            $"Tracked: {tracked}\n" +
            $"Confidence: {confidence}\n" +
            $"PointerValid: {pointerValid}\n" +
            $"IndexPinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Index)}\n" +
            $"MiddlePinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Middle)}\n" +
            $"RingPinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Ring)}\n" +
            $"PinkyPinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Pinky)}";
    }

    void DebugReferenceState()
    {
        Debug.Log(
            "[HandDebugManager] References -> " +
            $"leftHand={(leftHand != null ? leftHand.gameObject.name : "NULL")}, " +
            $"rightHand={(rightHand != null ? rightHand.gameObject.name : "NULL")}, " +
            $"debugText={(debugText != null ? debugText.gameObject.name : "NULL")}"
        );
    }
}
using UnityEngine;
using TMPro;

public class QuestHandLandmarkExtractor : MonoBehaviour
{
    [Header("References")]
    public OVRHand leftHand;
    public OVRHand rightHand;

    [Header("Optional Debug UI")]
    public TextMeshProUGUI debugText;

    [Header("Debug")]
    public bool logToConsole = true;
    public float logInterval = 0.5f;

    private float nextLogTime;

    void Update()
    {
        if (Time.time < nextLogTime)
            return;

        nextLogTime = Time.time + logInterval;

        string leftInfo = BuildHandInfo("Left", leftHand);
        string rightInfo = BuildHandInfo("Right", rightHand);
        string fullText = leftInfo + "\n\n" + rightInfo;

        if (debugText != null)
            debugText.text = fullText;

        if (logToConsole)
            Debug.Log(fullText);
    }

    private string BuildHandInfo(string handName, OVRHand hand)
    {
        if (hand == null)
            return $"{handName} Hand\nReference missing";

        bool isTracked = hand.IsTracked;
        bool isPointerValid = hand.IsPointerPoseValid;
        bool isHighConfidence = hand.HandConfidence == OVRHand.TrackingConfidence.High;

        string pinchState = "";
        pinchState += $"IndexPinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Index)}\n";
        pinchState += $"MiddlePinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Middle)}\n";
        pinchState += $"RingPinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Ring)}\n";
        pinchState += $"PinkyPinch: {hand.GetFingerIsPinching(OVRHand.HandFinger.Pinky)}\n";

        string pointerPoseText = "PointerPose: not valid";
        if (hand.PointerPose != null)
        {
            Vector3 p = hand.PointerPose.position;
            pointerPoseText = $"PointerPose: ({p.x:F3}, {p.y:F3}, {p.z:F3})";
        }

        return
            $"{handName} Hand\n" +
            $"Tracked: {isTracked}\n" +
            $"High confidence: {isHighConfidence}\n" +
            $"Pointer pose valid: {isPointerValid}\n" +
            $"{pointerPoseText}\n" +
            pinchState;
    }
}
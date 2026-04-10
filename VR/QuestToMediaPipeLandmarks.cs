// QuestToMediaPipeLandmarks.cs
// ================================
// Extracts hand bone positions from Meta Quest hand tracking
// and remaps them into the 225-feature vector format that
// the ASL Transformer-LSTM model expects.
//
// Attach this to a GameObject in your VR scene.
// Requires: OVRCameraRig with hand tracking enabled.
//
// Output: float[225] per frame, matching:
//   [0:63]   = Left hand  (21 landmarks × 3 coords)
//   [63:126] = Right hand (21 landmarks × 3 coords)
//   [126:225] = Pose       (33 landmarks × 3 coords)

using UnityEngine;
using System.Collections.Generic;

namespace ASLRecognitionVR
{
    public class QuestToMediaPipeLandmarks : MonoBehaviour
    {
        [Header("Hand References")]
        [Tooltip("Assign the OVRHand for the left hand (from OVRCameraRig > TrackingSpace > LeftHandAnchor)")]
        public OVRHand leftHand;

        [Tooltip("Assign the OVRHand for the right hand (from OVRCameraRig > TrackingSpace > RightHandAnchor)")]
        public OVRHand rightHand;

        [Tooltip("Assign the OVRSkeleton for the left hand")]
        public OVRSkeleton leftSkeleton;

        [Tooltip("Assign the OVRSkeleton for the right hand")]
        public OVRSkeleton rightSkeleton;

        [Header("Head Tracking")]
        [Tooltip("Assign the CenterEyeAnchor from OVRCameraRig")]
        public Transform headTransform;

        [Header("Settings")]
        [Tooltip("Estimated fingertip extension distance in meters")]
        public float fingertipExtension = 0.015f;

        // Constants
        private const int LANDMARKS_PER_HAND = 21;
        private const int COORDS_PER_LANDMARK = 3;
        private const int HAND_FEATURES = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK; // 63
        private const int POSE_LANDMARKS = 33;
        private const int POSE_FEATURES = POSE_LANDMARKS * COORDS_PER_LANDMARK; // 99
        private const int TOTAL_FEATURES = HAND_FEATURES * 2 + POSE_FEATURES; // 225

        /// <summary>
        /// MediaPipe landmark indices and their corresponding Quest bone IDs.
        /// Quest OVR skeleton has 24 bones per hand; MediaPipe has 21 landmarks.
        /// Fingertips (indices 4, 8, 12, 16, 20) are computed by extrapolation.
        /// </summary>
        private static readonly int[] QuestBoneToMediaPipe = new int[]
        {
            // MediaPipe idx => Quest OVRSkeleton.BoneId
            // 0: WRIST
            (int)OVRSkeleton.BoneId.Hand_WristRoot,

            // 1: THUMB_CMC
            (int)OVRSkeleton.BoneId.Hand_Thumb0,

            // 2: THUMB_MCP
            (int)OVRSkeleton.BoneId.Hand_Thumb1,

            // 3: THUMB_IP
            (int)OVRSkeleton.BoneId.Hand_Thumb2,

            // 4: THUMB_TIP — will be computed, placeholder
            (int)OVRSkeleton.BoneId.Hand_Thumb3,

            // 5: INDEX_MCP
            (int)OVRSkeleton.BoneId.Hand_Index1,

            // 6: INDEX_PIP
            (int)OVRSkeleton.BoneId.Hand_Index2,

            // 7: INDEX_DIP
            (int)OVRSkeleton.BoneId.Hand_Index3,

            // 8: INDEX_TIP — will be computed, placeholder
            (int)OVRSkeleton.BoneId.Hand_Index3,

            // 9: MIDDLE_MCP
            (int)OVRSkeleton.BoneId.Hand_Middle1,

            // 10: MIDDLE_PIP
            (int)OVRSkeleton.BoneId.Hand_Middle2,

            // 11: MIDDLE_DIP
            (int)OVRSkeleton.BoneId.Hand_Middle3,

            // 12: MIDDLE_TIP — will be computed, placeholder
            (int)OVRSkeleton.BoneId.Hand_Middle3,

            // 13: RING_MCP
            (int)OVRSkeleton.BoneId.Hand_Ring1,

            // 14: RING_PIP
            (int)OVRSkeleton.BoneId.Hand_Ring2,

            // 15: RING_DIP
            (int)OVRSkeleton.BoneId.Hand_Ring3,

            // 16: RING_TIP — will be computed, placeholder
            (int)OVRSkeleton.BoneId.Hand_Ring3,

            // 17: PINKY_MCP
            (int)OVRSkeleton.BoneId.Hand_Pinky0,

            // 18: PINKY_PIP
            (int)OVRSkeleton.BoneId.Hand_Pinky1,

            // 19: PINKY_DIP
            (int)OVRSkeleton.BoneId.Hand_Pinky2,

            // 20: PINKY_TIP — will be computed, placeholder
            (int)OVRSkeleton.BoneId.Hand_Pinky2,
        };

        // MediaPipe indices that are fingertips (must be extrapolated)
        private static readonly int[] FingertipIndices = { 4, 8, 12, 16, 20 };

        // For each fingertip, the DIP bone index (the bone before the tip)
        // Used to compute the extension direction
        private static readonly int[] FingertipDIPIndices = { 3, 7, 11, 15, 19 };

        // For each fingertip, the PIP bone index (two bones before tip)
        // Used with DIP to get the direction vector
        private static readonly int[] FingertipPIPIndices = { 2, 6, 10, 14, 18 };

        // Reference position for normalization (wrist center)
        private Vector3 _normalizationCenter;
        private float _normalizationScale = 1f;

        // =====================================================
        // PUBLIC API
        // =====================================================

        /// <summary>
        /// Extract a complete 225-feature landmark vector for the current frame.
        /// Returns null if both hands are untracked.
        /// </summary>
        public float[] ExtractLandmarks()
        {
            float[] features = new float[TOTAL_FEATURES];

            // Extract left hand landmarks (features 0-62)
            bool leftTracked = ExtractHandLandmarks(
                leftHand, leftSkeleton, features, 0
            );

            // Extract right hand landmarks (features 63-125)
            bool rightTracked = ExtractHandLandmarks(
                rightHand, rightSkeleton, features, HAND_FEATURES
            );

            // Extract pose landmarks (features 126-224)
            ExtractPoseLandmarks(features, HAND_FEATURES * 2);

            // Return null only if neither hand is tracked
            if (!leftTracked && !rightTracked)
                return null;

            return features;
        }

        /// <summary>
        /// Check if at least one hand is currently tracked.
        /// </summary>
        public bool IsAnyHandTracked()
        {
            return (leftHand != null && leftHand.IsTracked) ||
                   (rightHand != null && rightHand.IsTracked);
        }

        // =====================================================
        // HAND LANDMARK EXTRACTION
        // =====================================================

        private bool ExtractHandLandmarks(
            OVRHand hand, OVRSkeleton skeleton,
            float[] features, int offset)
        {
            // If hand is not tracked, leave features as zero (same as training padding)
            if (hand == null || !hand.IsTracked ||
                skeleton == null || skeleton.Bones == null ||
                skeleton.Bones.Count == 0)
            {
                return false;
            }

            // Compute normalization: center on wrist, scale to consistent range
            Transform wristBone = skeleton.Bones[
                (int)OVRSkeleton.BoneId.Hand_WristRoot
            ].Transform;
            _normalizationCenter = wristBone.position;

            // Estimate hand scale from wrist to middle MCP distance
            if (skeleton.Bones.Count > (int)OVRSkeleton.BoneId.Hand_Middle1)
            {
                Transform middleMCP = skeleton.Bones[
                    (int)OVRSkeleton.BoneId.Hand_Middle1
                ].Transform;
                float wristToMiddle = Vector3.Distance(
                    wristBone.position, middleMCP.position
                );
                // MediaPipe normalizes coords roughly to [0, 1]
                // A typical wrist-to-middle-MCP distance is ~8cm
                _normalizationScale = (wristToMiddle > 0.01f)
                    ? 1f / (wristToMiddle * 5f)
                    : 1f;
            }

            // Extract each of the 21 MediaPipe landmarks
            for (int mpIdx = 0; mpIdx < LANDMARKS_PER_HAND; mpIdx++)
            {
                Vector3 pos;

                if (IsFingertip(mpIdx))
                {
                    // Fingertips: extrapolate from DIP bone
                    pos = ComputeFingertipPosition(skeleton, mpIdx);
                }
                else
                {
                    // Direct bone mapping
                    int questBoneId = QuestBoneToMediaPipe[mpIdx];
                    if (questBoneId < skeleton.Bones.Count)
                    {
                        pos = skeleton.Bones[questBoneId].Transform.position;
                    }
                    else
                    {
                        pos = Vector3.zero;
                    }
                }

                // Normalize relative to wrist
                Vector3 normalized = (pos - _normalizationCenter) * _normalizationScale;

                // Write to feature array (x, y, z)
                int featureIdx = offset + mpIdx * COORDS_PER_LANDMARK;
                features[featureIdx] = normalized.x;
                features[featureIdx + 1] = normalized.y;
                features[featureIdx + 2] = normalized.z;
            }

            return true;
        }

        private bool IsFingertip(int mediaPipeIndex)
        {
            for (int i = 0; i < FingertipIndices.Length; i++)
            {
                if (FingertipIndices[i] == mediaPipeIndex) return true;
            }
            return false;
        }

        /// <summary>
        /// Compute fingertip position by extending the DIP->last-bone direction.
        /// This approximates where the fingertip would be.
        /// </summary>
        private Vector3 ComputeFingertipPosition(OVRSkeleton skeleton, int mpIdx)
        {
            int fingertipArrayIdx = -1;
            for (int i = 0; i < FingertipIndices.Length; i++)
            {
                if (FingertipIndices[i] == mpIdx)
                {
                    fingertipArrayIdx = i;
                    break;
                }
            }

            if (fingertipArrayIdx < 0) return Vector3.zero;

            int dipMPIdx = FingertipDIPIndices[fingertipArrayIdx];
            int pipMPIdx = FingertipPIPIndices[fingertipArrayIdx];

            int dipBoneId = QuestBoneToMediaPipe[dipMPIdx];
            int pipBoneId = QuestBoneToMediaPipe[pipMPIdx];

            if (dipBoneId >= skeleton.Bones.Count ||
                pipBoneId >= skeleton.Bones.Count)
            {
                return Vector3.zero;
            }

            Vector3 dipPos = skeleton.Bones[dipBoneId].Transform.position;
            Vector3 pipPos = skeleton.Bones[pipBoneId].Transform.position;

            // Direction from PIP to DIP, extended by fingertipExtension
            Vector3 direction = (dipPos - pipPos).normalized;
            return dipPos + direction * fingertipExtension;
        }

        // =====================================================
        // POSE LANDMARK EXTRACTION
        // =====================================================

        /// <summary>
        /// Extract approximate pose landmarks from Quest tracking data.
        /// Quest only provides head + hand positions, so most pose landmarks
        /// are set to zero (which matches how the model was trained — many
        /// pose landmarks were NaN/zero in the original dataset).
        /// </summary>
        private void ExtractPoseLandmarks(float[] features, int offset)
        {
            // All 33 pose landmarks × 3 coords = 99 features
            // Most are left as zero. We fill in what Quest can provide:

            if (headTransform != null)
            {
                Vector3 headPos = headTransform.position;

                // Landmark 0: Nose (approximate from head position)
                SetPoseLandmark(features, offset, 0, headPos + headTransform.forward * 0.05f);

                // Landmarks 11, 12: Left/Right shoulder (estimate from head)
                Vector3 leftShoulder = headPos + Vector3.down * 0.35f
                    + headTransform.right * (-0.2f);
                Vector3 rightShoulder = headPos + Vector3.down * 0.35f
                    + headTransform.right * 0.2f;
                SetPoseLandmark(features, offset, 11, leftShoulder);
                SetPoseLandmark(features, offset, 12, rightShoulder);
            }

            // Landmarks 15, 16: Left/Right wrist (from hand tracking)
            if (leftSkeleton != null && leftSkeleton.Bones != null &&
                leftSkeleton.Bones.Count > 0)
            {
                Vector3 leftWrist = leftSkeleton.Bones[
                    (int)OVRSkeleton.BoneId.Hand_WristRoot
                ].Transform.position;
                SetPoseLandmark(features, offset, 15, leftWrist);

                // Landmark 13: Left elbow (rough midpoint estimate)
                if (headTransform != null)
                {
                    Vector3 leftShoulder = headTransform.position
                        + Vector3.down * 0.35f
                        + headTransform.right * (-0.2f);
                    Vector3 leftElbow = Vector3.Lerp(leftShoulder, leftWrist, 0.5f);
                    SetPoseLandmark(features, offset, 13, leftElbow);
                }
            }

            if (rightSkeleton != null && rightSkeleton.Bones != null &&
                rightSkeleton.Bones.Count > 0)
            {
                Vector3 rightWrist = rightSkeleton.Bones[
                    (int)OVRSkeleton.BoneId.Hand_WristRoot
                ].Transform.position;
                SetPoseLandmark(features, offset, 16, rightWrist);

                // Landmark 14: Right elbow (rough midpoint estimate)
                if (headTransform != null)
                {
                    Vector3 rightShoulder = headTransform.position
                        + Vector3.down * 0.35f
                        + headTransform.right * 0.2f;
                    Vector3 rightElbow = Vector3.Lerp(rightShoulder, rightWrist, 0.5f);
                    SetPoseLandmark(features, offset, 14, rightElbow);
                }
            }

            // All other pose landmarks (hips, knees, ankles, etc.) remain 0
            // This is fine — the model was trained with these often being NaN/0
        }

        private void SetPoseLandmark(float[] features, int offset,
            int landmarkIdx, Vector3 position)
        {
            int idx = offset + landmarkIdx * COORDS_PER_LANDMARK;
            if (idx + 2 < features.Length)
            {
                features[idx] = position.x;
                features[idx + 1] = position.y;
                features[idx + 2] = position.z;
            }
        }
    }
}

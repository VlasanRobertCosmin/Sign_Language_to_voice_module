"""
Diagnostic script to understand your landmark data format.
Run this and share the output!
"""

import numpy as np

CACHE_FILE = "asl_signs_cache.npz"

print("=" * 60)
print("LANDMARK DATA DIAGNOSTIC")
print("=" * 60)

cache = np.load(CACHE_FILE, allow_pickle=True)
X = cache['X']
y = cache['y']

print(f"\nDataset shape: {X.shape}")
print(f"Sample labels: {y[:5]}")

# Get a sample with movement (not all zeros)
sample_idx = 0
for i in range(min(100, len(X))):
    if np.abs(X[i]).sum() > 10:
        sample_idx = i
        break

sample = X[sample_idx]
print(f"\nUsing sample: '{y[sample_idx]}' (index {sample_idx})")
print(f"Sample shape: {sample.shape}")

# Frame 0 and middle frame
for frame_idx in [0, len(sample)//2]:
    print(f"\n--- Frame {frame_idx} ---")
    
    # Left hand (21 landmarks * 3 coords = 63)
    left_hand = sample[frame_idx, 0:63].reshape(21, 3)
    print(f"Left hand: min={left_hand.min():.3f}, max={left_hand.max():.3f}, mean={left_hand.mean():.3f}")
    print(f"  Wrist (0): {left_hand[0]}")
    print(f"  Index tip (8): {left_hand[8]}")
    
    # Right hand
    right_hand = sample[frame_idx, 63:126].reshape(21, 3)
    print(f"Right hand: min={right_hand.min():.3f}, max={right_hand.max():.3f}, mean={right_hand.mean():.3f}")
    print(f"  Wrist (0): {right_hand[0]}")
    print(f"  Index tip (8): {right_hand[8]}")
    
    # Pose
    pose = sample[frame_idx, 126:225].reshape(33, 3)
    print(f"Pose: min={pose.min():.3f}, max={pose.max():.3f}, mean={pose.mean():.3f}")
    print(f"  Left shoulder (11): {pose[11]}")
    print(f"  Right shoulder (12): {pose[12]}")
    print(f"  Left wrist (15): {pose[15]}")
    print(f"  Right wrist (16): {pose[16]}")

# Check if hands have data
print("\n--- Hand visibility check ---")
for i in range(min(10, len(sample))):
    left = sample[i, 0:63]
    right = sample[i, 63:126]
    left_visible = np.abs(left).sum() > 0.5
    right_visible = np.abs(right).sum() > 0.5
    print(f"Frame {i}: Left={'YES' if left_visible else 'no'}, Right={'YES' if right_visible else 'no'}")

print("\n" + "=" * 60)
print("Copy everything above and share it!")
print("=" * 60)
"""
Quick test - verify hand data
"""
import numpy as np

CACHE_FILE = "asl_signs_cache.npz"

cache = np.load(CACHE_FILE, allow_pickle=True)
X = cache['X']
y = cache['y']

# Find "blow" or another sign with hand data
test_words = ['blow', 'hello', 'thank', 'you', 'please']

for word in test_words:
    idx = np.where(y == word)[0]
    if len(idx) > 0:
        sample = X[idx[0]]
        
        # Count valid frames
        valid_frames = 0
        has_left = 0
        has_right = 0
        
        for f in range(len(sample)):
            if np.abs(sample[f]).sum() > 0.1:
                valid_frames += 1
                left = sample[f, 0:63]
                right = sample[f, 63:126]
                if np.abs(left).sum() > 0.5:
                    has_left += 1
                if np.abs(right).sum() > 0.5:
                    has_right += 1
        
        print(f"{word:10}: {valid_frames:2} frames, left_hand={has_left:2}, right_hand={has_right:2}")
        
        # Show first frame hand data
        if valid_frames > 0:
            frame = sample[0]
            right_hand = frame[63:126].reshape(21, 3)
            print(f"           Right hand wrist: {right_hand[0]}")
            print(f"           Right index tip:  {right_hand[8]}")
            print()
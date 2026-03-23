"""
Speech to Sign Language - Animated Avatar (FIXED)
==================================================
Correctly handles the landmark coordinate system from your dataset.
"""

import os
import sys
import numpy as np
import cv2
import time
from collections import deque
import math

# Suppress ALSA warnings on Linux
try:
    from ctypes import *
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# Speech recognition (optional)
SPEECH_AVAILABLE = False
try:
    import speech_recognition as sr
    sr.Microphone()
    SPEECH_AVAILABLE = True
    print("Speech recognition: ENABLED")
except Exception as e:
    print(f"Speech recognition: DISABLED")
    print("Use ENTER key to type text instead")


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_FILE = "asl_signs_cache.npz"

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 30
ANIMATION_SPEED = 0.5  # Slower animation (was 1.0)

# Avatar colors
SKIN_COLOR = (180, 160, 150)
SKIN_DARK = (140, 120, 110)
SHIRT_COLOR = (180, 100, 80)
HAIR_COLOR = (50, 35, 25)
BG_COLOR = (45, 42, 38)


# ============================================================
# SIGN DATABASE
# ============================================================

class SignDatabase:
    def __init__(self, cache_file):
        self.signs = {}
        self.classes = []
        self.load_data(cache_file)
    
    def interpolate_frames(self, frames, factor=2):
        """Interpolate between frames for smoother animation."""
        if len(frames) < 2:
            return frames
        
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            # Add interpolated frames between each pair
            for j in range(1, factor):
                t = j / factor
                interp = frames[i] * (1 - t) + frames[i + 1] * t
                interpolated.append(interp)
        interpolated.append(frames[-1])
        
        return np.array(interpolated)
    
    def load_data(self, cache_file):
        if not os.path.exists(cache_file):
            print(f"Cache not found: {cache_file}")
            self._create_demo()
            return
        
        print(f"Loading {cache_file}...")
        cache = np.load(cache_file, allow_pickle=True)
        X = cache['X']
        y = cache['y']
        
        self.classes = sorted(list(set(y)))
        
        # Store one example per class
        for i, label in enumerate(y):
            if label not in self.signs:
                # Find the actual length (non-zero frames)
                seq = X[i]
                valid_frames = []
                for f in range(len(seq)):
                    if np.abs(seq[f]).sum() > 0.1:
                        valid_frames.append(seq[f])
                
                if valid_frames:
                    # Interpolate for smoother animation
                    frames = np.array(valid_frames)
                    frames = self.interpolate_frames(frames, factor=3)
                    self.signs[label] = frames
                else:
                    self.signs[label] = seq[:10]  # Fallback
        
        print(f"Loaded {len(self.signs)} signs (with interpolation)")
    
    def _create_demo(self):
        for word in ['hello', 'thank', 'you']:
            self.signs[word] = np.random.rand(30, 225).astype(np.float32) * 0.5
            self.classes.append(word)
    
    def get_sign(self, word):
        return self.signs.get(word.lower().strip())
    
    def find_similar(self, word):
        word = word.lower().strip()
        if word in self.signs:
            return [word]
        matches = [s for s in self.classes if word in s or s in word]
        if not matches:
            matches = [s for s in self.classes if s[:2] == word[:2]]
        return matches[:3]


# ============================================================
# AVATAR RENDERER
# ============================================================

class AvatarRenderer:
    """Renders avatar using actual landmark data from dataset."""
    
    def __init__(self, width=700, height=650):
        self.width = width
        self.height = height
        
        # Avatar position
        self.center_x = width // 2
        self.center_y = height // 2 + 50
        self.scale = 350
        
        # Smoothing - higher = more responsive, lower = smoother
        self.prev_landmarks = None
        self.smooth = 0.25  # More fluid (was 0.4)
        
        # Hand connections for drawing fingers
        self.finger_chains = [
            [0, 1, 2, 3, 4],       # Thumb
            [0, 5, 6, 7, 8],       # Index  
            [0, 9, 10, 11, 12],    # Middle
            [0, 13, 14, 15, 16],   # Ring
            [0, 17, 18, 19, 20],   # Pinky
        ]
        
        self.palm_indices = [0, 5, 9, 13, 17]
    
    def smooth_landmarks(self, landmarks):
        """Apply smoothing between frames."""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            return landmarks
        
        smoothed = self.smooth * landmarks + (1 - self.smooth) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        return smoothed
    
    def draw(self, frame, raw_landmarks):
        """Draw the complete avatar."""
        landmarks = self.smooth_landmarks(raw_landmarks)
        
        # Parse landmark groups
        left_hand = landmarks[0:63].reshape(21, 3)
        right_hand = landmarks[63:126].reshape(21, 3)
        pose = landmarks[126:225].reshape(33, 3)
        
        # Check which hands are visible
        left_visible = np.abs(left_hand).sum() > 0.5
        right_visible = np.abs(right_hand).sum() > 0.5
        
        # Draw body first
        self.draw_body(frame, pose)
        
        # Draw arms and hands
        self.draw_arm_and_hand(frame, pose, right_hand, is_left=False, hand_visible=right_visible)
        self.draw_arm_and_hand(frame, pose, left_hand, is_left=True, hand_visible=left_visible)
        
        # Draw head last (on top)
        self.draw_head(frame)
        
        return frame
    
    def pose_to_screen(self, pose_point):
        """Convert pose landmark to screen coordinates.
        Pose data ranges roughly from -2 to 2, with center around 0.5
        """
        x = pose_point[0]
        y = pose_point[1]
        
        # Normalize: pose x is roughly 0-1 for body, y is 0-1
        # But can extend beyond for arms
        screen_x = self.center_x + (x - 0.5) * self.scale
        screen_y = self.center_y + (y - 0.5) * self.scale * 0.8
        
        return int(screen_x), int(screen_y)
    
    def hand_to_screen(self, hand_point, wrist_screen_pos):
        """Convert hand landmark to screen coordinates relative to wrist."""
        # Hand coordinates are 0-1, relative positions
        x = hand_point[0]
        y = hand_point[1]
        
        # Scale factor for hand size
        hand_scale = 250
        
        # Offset from wrist
        dx = (x - 0.5) * hand_scale
        dy = (y - 0.5) * hand_scale
        
        screen_x = wrist_screen_pos[0] + dx
        screen_y = wrist_screen_pos[1] + dy
        
        return int(screen_x), int(screen_y)
    
    def draw_body(self, frame, pose):
        """Draw torso and neck."""
        # Get shoulder positions
        left_shoulder = self.pose_to_screen(pose[11])
        right_shoulder = self.pose_to_screen(pose[12])
        
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2
        )
        
        # Body rectangle
        body_width = abs(left_shoulder[0] - right_shoulder[0]) + 40
        body_top = shoulder_center[1]
        body_bottom = shoulder_center[1] + 200
        
        # Draw body (trapezoid shape)
        pts = np.array([
            [shoulder_center[0] - body_width//2, body_top],
            [shoulder_center[0] + body_width//2, body_top],
            [shoulder_center[0] + body_width//2 + 30, body_bottom],
            [shoulder_center[0] - body_width//2 - 30, body_bottom],
        ], np.int32)
        
        cv2.fillPoly(frame, [pts], SHIRT_COLOR)
        cv2.polylines(frame, [pts], True, (120, 70, 50), 3)
        
        # Neck
        neck_top = shoulder_center[1] - 30
        cv2.rectangle(frame,
                     (shoulder_center[0] - 25, neck_top),
                     (shoulder_center[0] + 25, body_top + 10),
                     SKIN_COLOR, -1)
        
        # Store for head position
        self.head_center = (shoulder_center[0], neck_top - 60)
    
    def draw_head(self, frame):
        """Draw head and face."""
        cx, cy = self.head_center
        radius = 55
        
        # Head
        cv2.circle(frame, (cx, cy), radius, SKIN_COLOR, -1)
        cv2.circle(frame, (cx, cy), radius, SKIN_DARK, 2)
        
        # Hair
        hair_pts = []
        for angle in range(160, 381, 8):
            rad = math.radians(angle)
            x = cx + int((radius + 5) * math.cos(rad))
            y = cy + int((radius + 5) * math.sin(rad))
            hair_pts.append([x, y])
        if hair_pts:
            cv2.fillPoly(frame, [np.array(hair_pts, np.int32)], HAIR_COLOR)
        
        # Eyes
        eye_y = cy - 5
        eye_spacing = 22
        
        # Eye whites
        cv2.ellipse(frame, (cx - eye_spacing, eye_y), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx + eye_spacing, eye_y), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        
        # Pupils
        cv2.circle(frame, (cx - eye_spacing, eye_y), 5, (40, 30, 20), -1)
        cv2.circle(frame, (cx + eye_spacing, eye_y), 5, (40, 30, 20), -1)
        
        # Eyebrows
        cv2.line(frame, (cx - eye_spacing - 12, eye_y - 15), 
                (cx - eye_spacing + 10, eye_y - 12), HAIR_COLOR, 3)
        cv2.line(frame, (cx + eye_spacing - 10, eye_y - 12),
                (cx + eye_spacing + 12, eye_y - 15), HAIR_COLOR, 3)
        
        # Nose
        cv2.line(frame, (cx, cy + 5), (cx + 8, cy + 20), SKIN_DARK, 2)
        
        # Mouth
        cv2.ellipse(frame, (cx, cy + 35), (15, 8), 0, 10, 170, (120, 80, 80), 2)
    
    def draw_arm_and_hand(self, frame, pose, hand, is_left, hand_visible):
        """Draw arm from shoulder to wrist, then hand."""
        # Pose indices
        shoulder_idx = 11 if is_left else 12
        elbow_idx = 13 if is_left else 14
        wrist_idx = 15 if is_left else 16
        
        shoulder = self.pose_to_screen(pose[shoulder_idx])
        elbow = self.pose_to_screen(pose[elbow_idx])
        wrist = self.pose_to_screen(pose[wrist_idx])
        
        # Clamp to reasonable positions
        wrist = (
            max(50, min(self.width - 50, wrist[0])),
            max(50, min(self.height - 50, wrist[1]))
        )
        elbow = (
            max(50, min(self.width - 50, elbow[0])),
            max(50, min(self.height - 50, elbow[1]))
        )
        
        # Draw upper arm
        cv2.line(frame, shoulder, elbow, SKIN_COLOR, 22)
        cv2.line(frame, shoulder, elbow, SKIN_DARK, 2)
        
        # Draw lower arm
        cv2.line(frame, elbow, wrist, SKIN_COLOR, 18)
        cv2.line(frame, elbow, wrist, SKIN_DARK, 2)
        
        # Joint circles
        cv2.circle(frame, elbow, 12, SKIN_COLOR, -1)
        cv2.circle(frame, elbow, 12, SKIN_DARK, 2)
        
        # Draw hand
        if hand_visible:
            self.draw_hand(frame, hand, wrist, is_left)
        else:
            # Draw simple fist
            cv2.circle(frame, wrist, 25, SKIN_COLOR, -1)
            cv2.circle(frame, wrist, 25, SKIN_DARK, 2)
    
    def draw_hand(self, frame, hand_landmarks, wrist_pos, is_left):
        """Draw detailed hand with fingers."""
        # Hand landmarks are in ABSOLUTE coordinates (0-1), not relative to wrist
        # We need to convert them directly to screen space
        
        # Get screen positions for all hand landmarks
        points = []
        for i in range(21):
            x = hand_landmarks[i][0]
            y = hand_landmarks[i][1]
            
            # Convert from normalized (0-1) to screen coordinates
            # Center the hand area and scale appropriately
            screen_x = self.center_x + (x - 0.5) * self.scale * 1.5
            screen_y = self.center_y + (y - 0.5) * self.scale * 1.2
            
            points.append((int(screen_x), int(screen_y)))
        
        # Draw palm (filled polygon connecting base of fingers)
        palm_pts = [points[0], points[5], points[9], points[13], points[17]]
        if len(palm_pts) >= 3:
            cv2.fillPoly(frame, [np.array(palm_pts, np.int32)], SKIN_COLOR)
            cv2.polylines(frame, [np.array(palm_pts, np.int32)], True, SKIN_DARK, 2)
        
        # Draw fingers
        for finger in self.finger_chains:
            finger_pts = [points[i] for i in finger]
            
            # Draw finger segments as thick lines
            for i in range(len(finger_pts) - 1):
                thickness = 14 - i * 3  # Thinner toward tip
                thickness = max(5, thickness)
                cv2.line(frame, finger_pts[i], finger_pts[i + 1], SKIN_COLOR, thickness)
                cv2.line(frame, finger_pts[i], finger_pts[i + 1], SKIN_DARK, 2)
            
            # Draw joints as circles
            for i, pt in enumerate(finger_pts):
                radius = 10 - i * 2  # Smaller toward tip
                radius = max(4, radius)
                cv2.circle(frame, pt, radius, SKIN_COLOR, -1)
                cv2.circle(frame, pt, radius, SKIN_DARK, 1)
            
            # Fingertip
            cv2.circle(frame, finger_pts[-1], 6, SKIN_DARK, 2)
        
        # Wrist connection
        cv2.circle(frame, points[0], 12, SKIN_COLOR, -1)


# ============================================================
# SPEECH RECOGNIZER
# ============================================================

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = None
        if SPEECH_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
    
    def listen(self, timeout=5):
        if not self.recognizer:
            return None, "No microphone"
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            text = self.recognizer.recognize_google(audio)
            return text.lower(), None
        except sr.WaitTimeoutError:
            return None, "No speech detected"
        except sr.UnknownValueError:
            return None, "Could not understand"
        except Exception as e:
            return None, str(e)


# ============================================================
# MAIN APPLICATION
# ============================================================

class SpeechToSignApp:
    def __init__(self):
        self.db = SignDatabase(CACHE_FILE)
        self.avatar = AvatarRenderer(WINDOW_WIDTH - 280, WINDOW_HEIGHT)
        self.speech = SpeechRecognizer()
        
        self.current_word = ""
        self.current_sign = None
        self.frame_idx = 0
        self.is_playing = False
        self.word_queue = deque()
        self.status = "Press SPACE to speak"
        
        # Neutral pose
        self.neutral = self._create_neutral()
        self.current_landmarks = self.neutral.copy()
    
    def _create_neutral(self):
        """Create neutral standing pose."""
        lm = np.zeros(225, dtype=np.float32)
        
        # Pose - shoulders and arms down
        pose_start = 126
        # Left shoulder
        lm[pose_start + 11*3] = 0.7
        lm[pose_start + 11*3 + 1] = 0.4
        # Right shoulder  
        lm[pose_start + 12*3] = 0.3
        lm[pose_start + 12*3 + 1] = 0.4
        # Left elbow
        lm[pose_start + 13*3] = 0.8
        lm[pose_start + 13*3 + 1] = 0.6
        # Right elbow
        lm[pose_start + 14*3] = 0.2
        lm[pose_start + 14*3 + 1] = 0.6
        # Left wrist
        lm[pose_start + 15*3] = 0.85
        lm[pose_start + 15*3 + 1] = 0.8
        # Right wrist
        lm[pose_start + 16*3] = 0.15
        lm[pose_start + 16*3 + 1] = 0.8
        
        return lm
    
    def process_text(self, text):
        words = text.lower().split()
        found = False
        
        for word in words:
            word = ''.join(c for c in word if c.isalpha())
            if not word:
                continue
            
            if self.db.get_sign(word) is not None:
                self.word_queue.append(word)
                found = True
                print(f"  ✓ Found: {word}")
            else:
                similar = self.db.find_similar(word)
                if similar:
                    self.word_queue.append(similar[0])
                    found = True
                    print(f"  ~ Using '{similar[0]}' for '{word}'")
                else:
                    print(f"  ✗ Not found: {word}")
        
        return found
    
    def play_next(self):
        if self.word_queue:
            self.current_word = self.word_queue.popleft()
            self.current_sign = self.db.get_sign(self.current_word)
            self.frame_idx = 0
            self.is_playing = True
            self.status = f"Signing: {self.current_word.upper()}"
            print(f"Playing: {self.current_word} ({len(self.current_sign)} frames)")
    
    def update(self):
        if self.is_playing and self.current_sign is not None:
            if self.frame_idx < len(self.current_sign):
                self.current_landmarks = self.current_sign[self.frame_idx]
                self.frame_idx += 1
            else:
                self.is_playing = False
                if self.word_queue:
                    self.play_next()
                else:
                    self.status = "Ready"
                    self.current_word = ""
        else:
            # Return to neutral
            self.current_landmarks = 0.9 * self.current_landmarks + 0.1 * self.neutral
    
    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        panel_x = w - 260
        
        # Panel
        cv2.rectangle(frame, (panel_x, 0), (w, h), (35, 32, 30), -1)
        cv2.line(frame, (panel_x, 0), (panel_x, h), (60, 55, 50), 2)
        
        # Title
        cv2.putText(frame, "SPEECH TO SIGN", (panel_x + 15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        cv2.line(frame, (panel_x + 15, 50), (w - 15, 50), (60, 55, 50), 1)
        
        # Current word
        cv2.putText(frame, "Now signing:", (panel_x + 15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        word = self.current_word.upper() if self.current_word else "---"
        cv2.putText(frame, word, (panel_x + 15, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 180), 2)
        
        # Progress
        if self.is_playing and self.current_sign is not None:
            progress = self.frame_idx / len(self.current_sign)
            bar_w = 220
            cv2.rectangle(frame, (panel_x + 15, 145), (panel_x + 15 + bar_w, 160), (50, 50, 50), -1)
            cv2.rectangle(frame, (panel_x + 15, 145), (panel_x + 15 + int(bar_w * progress), 160), (100, 255, 180), -1)
        
        # Queue
        cv2.putText(frame, f"Queue ({len(self.word_queue)}):", (panel_x + 15, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        y = 225
        for w in list(self.word_queue)[:5]:
            cv2.putText(frame, f"• {w}", (panel_x + 25, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y += 22
        
        # Controls
        cv2.line(frame, (panel_x + 15, 350), (w - 15, 350), (60, 55, 50), 1)
        cv2.putText(frame, "CONTROLS", (panel_x + 15, 380),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        controls = [
            "SPACE  Voice input",
            "ENTER  Type text", 
            "R      Replay",
            "C      Clear",
            "L      List signs",
            "Q      Quit"
        ]
        
        y = 410
        for ctrl in controls:
            cv2.putText(frame, ctrl, (panel_x + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
            y += 25
        
        # Status
        cv2.putText(frame, self.status[:30], (panel_x + 15, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 180), 1)
    
    def run(self):
        print("\n" + "=" * 50)
        print("SPEECH TO SIGN - AVATAR")
        print("=" * 50)
        print(f"Signs loaded: {len(self.db.classes)}")
        print("\nSPACE=speak, ENTER=type, Q=quit")
        print("=" * 50 + "\n")
        
        cv2.namedWindow("Speech to Sign", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Speech to Sign", WINDOW_WIDTH, WINDOW_HEIGHT)
        
        frame_time = 1.0 / FPS
        last_update = time.time()
        
        while True:
            # Create frame
            frame = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BG_COLOR, dtype=np.uint8)
            
            # Update animation
            now = time.time()
            if now - last_update >= frame_time / ANIMATION_SPEED:
                self.update()
                last_update = now
            
            # Draw avatar
            avatar_area = frame[:, :WINDOW_WIDTH - 260]
            self.avatar.draw(avatar_area, self.current_landmarks)
            
            # Draw UI
            self.draw_ui(frame)
            
            cv2.imshow("Speech to Sign", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):
                self.status = "Listening..."
                cv2.imshow("Speech to Sign", frame)
                cv2.waitKey(1)
                
                text, err = self.speech.listen()
                if text:
                    print(f"\nHeard: '{text}'")
                    if self.process_text(text) and not self.is_playing:
                        self.play_next()
                else:
                    self.status = err or "Try again"
            
            elif key == 13:  # Enter
                print("\nType text: ", end="", flush=True)
                text = input()
                if text:
                    print(f"Processing: '{text}'")
                    if self.process_text(text) and not self.is_playing:
                        self.play_next()
            
            elif key == ord('r') and self.current_sign is not None:
                self.frame_idx = 0
                self.is_playing = True
            
            elif key == ord('c'):
                self.word_queue.clear()
                self.is_playing = False
                self.current_word = ""
                self.status = "Cleared"
            
            elif key == ord('l'):
                print("\nAvailable signs:")
                for i, s in enumerate(sorted(self.db.classes)):
                    print(f"{s:15}", end="")
                    if (i + 1) % 6 == 0:
                        print()
                print("\n")
            
            if not self.is_playing and self.word_queue:
                self.play_next()
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SpeechToSignApp()
    app.run()
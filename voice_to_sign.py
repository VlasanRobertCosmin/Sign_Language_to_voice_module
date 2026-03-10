"""
Speech to Sign Language - Animated Avatar
==========================================
Creates an animated 2D avatar that performs ASL signs
using the landmark data from your training dataset.

Features:
- Cartoon-style avatar with body, arms, hands
- Smooth animation interpolation
- Hand shape visualization with fingers
- Real-time speech recognition

Requirements:
    pip install SpeechRecognition pyaudio numpy opencv-python
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

# Speech recognition (optional - works without it)
SPEECH_AVAILABLE = False
try:
    import speech_recognition as sr
    # Test if pyaudio is available
    sr.Microphone()
    SPEECH_AVAILABLE = True
    print("Speech recognition: ENABLED")
except Exception as e:
    print(f"Speech recognition: DISABLED ({e})")
    print("You can still type text with ENTER key")
    print("To enable voice: sudo apt-get install portaudio19-dev && pip install pyaudio")


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_FILE = "asl_signs_cache.npz"

# Window settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
AVATAR_CENTER_X = 400
AVATAR_CENTER_Y = 400

# Animation
FPS = 30
ANIMATION_SPEED = 1.0
SMOOTH_FACTOR = 0.3  # For interpolation between frames

# Avatar colors
SKIN_COLOR = (200, 180, 170)
SKIN_OUTLINE = (150, 130, 120)
SHIRT_COLOR = (100, 80, 180)  # Purple
SHIRT_OUTLINE = (70, 50, 140)
HAIR_COLOR = (60, 40, 30)
EYE_COLOR = (80, 60, 50)
HAND_LEFT_COLOR = (210, 190, 180)
HAND_RIGHT_COLOR = (210, 190, 180)


# ============================================================
# SIGN DATABASE
# ============================================================

class SignDatabase:
    """Loads ASL signs from training cache."""
    
    def __init__(self, cache_file):
        self.signs = {}
        self.classes = []
        self.load_data(cache_file)
    
    def load_data(self, cache_file):
        if not os.path.exists(cache_file):
            print(f"Cache not found: {cache_file}")
            self._create_demo_signs()
            return
        
        print(f"Loading signs from {cache_file}...")
        cache = np.load(cache_file, allow_pickle=True)
        X = cache['X']
        y = cache['y']
        
        self.classes = sorted(list(set(y)))
        
        # Store best example per class (one with most movement)
        class_examples = {}
        for i, label in enumerate(y):
            if label not in class_examples:
                class_examples[label] = []
            class_examples[label].append(X[i])
        
        # Pick example with most hand movement
        for label, examples in class_examples.items():
            best_idx = 0
            best_movement = 0
            for idx, ex in enumerate(examples):
                # Calculate hand movement
                hand_data = ex[:, :126]  # Left + right hand
                movement = np.abs(np.diff(hand_data, axis=0)).sum()
                if movement > best_movement:
                    best_movement = movement
                    best_idx = idx
            self.signs[label] = examples[best_idx]
        
        print(f"Loaded {len(self.signs)} signs")
    
    def _create_demo_signs(self):
        """Create demo signs for testing."""
        demo_words = ['hello', 'thank', 'you', 'please', 'yes', 'no']
        for word in demo_words:
            frames = np.random.rand(64, 225).astype(np.float32) * 0.3 + 0.35
            self.signs[word] = frames
            self.classes.append(word)
    
    def get_sign(self, word):
        word = word.lower().strip()
        return self.signs.get(word)
    
    def find_similar(self, word, max_results=5):
        word = word.lower().strip()
        if word in self.signs:
            return [word]
        
        matches = []
        for sign in self.classes:
            if word in sign or sign in word:
                matches.append(sign)
        
        if not matches:
            for sign in self.classes:
                if len(word) >= 3 and len(sign) >= 3:
                    if sign[:3] == word[:3]:
                        matches.append(sign)
        
        return matches[:max_results]


# ============================================================
# 2D AVATAR RENDERER
# ============================================================

class Avatar2D:
    """
    Renders a 2D cartoon avatar that performs sign language.
    Uses MediaPipe landmark positions to animate arms and hands.
    """
    
    # MediaPipe pose landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # Hand landmark indices (within 21-point hand)
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    def __init__(self, center_x=400, center_y=350, scale=300):
        self.center_x = center_x
        self.center_y = center_y
        self.scale = scale
        
        # Smoothed positions for animation
        self.smooth_landmarks = None
        
        # Avatar body proportions
        self.head_radius = int(scale * 0.18)
        self.body_width = int(scale * 0.25)
        self.body_height = int(scale * 0.35)
        self.arm_thickness = int(scale * 0.06)
        self.hand_scale = scale * 0.4
    
    def parse_landmarks(self, raw_landmarks):
        """Parse raw landmark array into structured format."""
        # raw_landmarks: 225 features
        # [0:63] = left hand (21 * 3)
        # [63:126] = right hand (21 * 3)
        # [126:225] = pose (33 * 3)
        
        left_hand = raw_landmarks[0:63].reshape(21, 3)
        right_hand = raw_landmarks[63:126].reshape(21, 3)
        pose = raw_landmarks[126:225].reshape(33, 3)
        
        return {
            'left_hand': left_hand,
            'right_hand': right_hand,
            'pose': pose
        }
    
    def smooth_update(self, landmarks):
        """Smoothly interpolate to new landmarks."""
        if self.smooth_landmarks is None:
            self.smooth_landmarks = landmarks.copy()
        else:
            self.smooth_landmarks = (
                SMOOTH_FACTOR * landmarks + 
                (1 - SMOOTH_FACTOR) * self.smooth_landmarks
            )
        return self.smooth_landmarks
    
    def landmark_to_screen(self, x, y, offset_x=0, offset_y=0):
        """Convert normalized landmark to screen coordinates."""
        # Landmarks are normalized 0-1, center around avatar
        screen_x = self.center_x + (x - 0.5) * self.scale * 2 + offset_x
        screen_y = self.center_y + (y - 0.5) * self.scale * 1.5 + offset_y
        return int(screen_x), int(screen_y)
    
    def draw(self, frame, raw_landmarks):
        """Draw the complete avatar with current pose."""
        # Smooth the landmarks
        smoothed = self.smooth_update(raw_landmarks)
        lm = self.parse_landmarks(smoothed)
        
        pose = lm['pose']
        left_hand = lm['left_hand']
        right_hand = lm['right_hand']
        
        # Draw order: body -> arms -> hands -> head (back to front)
        self.draw_body(frame)
        self.draw_arms(frame, pose)
        self.draw_hands(frame, pose, left_hand, right_hand)
        self.draw_head(frame)
        
        return frame
    
    def draw_body(self, frame):
        """Draw the avatar's torso."""
        # Shirt/body
        body_top = self.center_y - int(self.body_height * 0.3)
        body_bottom = self.center_y + int(self.body_height * 0.7)
        
        # Draw rounded rectangle body
        pts = np.array([
            [self.center_x - self.body_width, body_top],
            [self.center_x + self.body_width, body_top],
            [self.center_x + self.body_width + 20, body_bottom],
            [self.center_x - self.body_width - 20, body_bottom]
        ], np.int32)
        
        cv2.fillPoly(frame, [pts], SHIRT_COLOR)
        cv2.polylines(frame, [pts], True, SHIRT_OUTLINE, 3)
        
        # Neck
        neck_width = int(self.head_radius * 0.5)
        neck_top = self.center_y - self.head_radius - int(self.body_height * 0.4)
        neck_bottom = body_top + 10
        
        cv2.rectangle(frame,
                     (self.center_x - neck_width, neck_top),
                     (self.center_x + neck_width, neck_bottom),
                     SKIN_COLOR, -1)
    
    def draw_head(self, frame):
        """Draw the avatar's head and face."""
        head_y = self.center_y - self.head_radius - int(self.body_height * 0.5)
        
        # Head circle
        cv2.circle(frame, (self.center_x, head_y), self.head_radius, SKIN_COLOR, -1)
        cv2.circle(frame, (self.center_x, head_y), self.head_radius, SKIN_OUTLINE, 3)
        
        # Hair (simple arc on top)
        hair_pts = []
        for angle in range(150, 391, 10):
            rad = math.radians(angle)
            x = self.center_x + int(self.head_radius * 1.05 * math.cos(rad))
            y = head_y + int(self.head_radius * 1.05 * math.sin(rad))
            hair_pts.append([x, y])
        
        if len(hair_pts) > 2:
            cv2.fillPoly(frame, [np.array(hair_pts, np.int32)], HAIR_COLOR)
        
        # Eyes
        eye_y = head_y - int(self.head_radius * 0.1)
        eye_offset = int(self.head_radius * 0.35)
        eye_size = int(self.head_radius * 0.15)
        
        # White of eyes
        cv2.circle(frame, (self.center_x - eye_offset, eye_y), eye_size + 3, (255, 255, 255), -1)
        cv2.circle(frame, (self.center_x + eye_offset, eye_y), eye_size + 3, (255, 255, 255), -1)
        
        # Pupils
        cv2.circle(frame, (self.center_x - eye_offset, eye_y), eye_size, EYE_COLOR, -1)
        cv2.circle(frame, (self.center_x + eye_offset, eye_y), eye_size, EYE_COLOR, -1)
        
        # Eyebrows
        brow_y = eye_y - int(self.head_radius * 0.2)
        cv2.line(frame, 
                (self.center_x - eye_offset - 10, brow_y),
                (self.center_x - eye_offset + 10, brow_y - 3),
                HAIR_COLOR, 3)
        cv2.line(frame,
                (self.center_x + eye_offset - 10, brow_y - 3),
                (self.center_x + eye_offset + 10, brow_y),
                HAIR_COLOR, 3)
        
        # Nose
        nose_y = head_y + int(self.head_radius * 0.1)
        cv2.line(frame,
                (self.center_x, nose_y - 5),
                (self.center_x + 5, nose_y + 5),
                SKIN_OUTLINE, 2)
        
        # Mouth (slight smile)
        mouth_y = head_y + int(self.head_radius * 0.4)
        cv2.ellipse(frame, 
                   (self.center_x, mouth_y - 5),
                   (15, 8), 0, 20, 160, (150, 100, 100), 2)
    
    def draw_arms(self, frame, pose):
        """Draw arms based on pose landmarks."""
        # Get pose positions
        l_shoulder = pose[self.LEFT_SHOULDER]
        r_shoulder = pose[self.RIGHT_SHOULDER]
        l_elbow = pose[self.LEFT_ELBOW]
        r_elbow = pose[self.RIGHT_ELBOW]
        l_wrist = pose[self.LEFT_WRIST]
        r_wrist = pose[self.RIGHT_WRIST]
        
        # Shoulder anchor points (fixed to body)
        shoulder_y = self.center_y - int(self.body_height * 0.25)
        l_shoulder_screen = (self.center_x - self.body_width - 5, shoulder_y)
        r_shoulder_screen = (self.center_x + self.body_width + 5, shoulder_y)
        
        # Calculate elbow and wrist positions from landmarks
        def get_arm_pos(shoulder_screen, elbow_lm, wrist_lm, is_left):
            # Use landmark positions to determine arm angle
            # Scale the movement
            dx_elbow = (elbow_lm[0] - 0.5) * self.scale * 1.5
            dy_elbow = (elbow_lm[1] - 0.3) * self.scale * 1.2
            
            dx_wrist = (wrist_lm[0] - 0.5) * self.scale * 1.8
            dy_wrist = (wrist_lm[1] - 0.3) * self.scale * 1.5
            
            elbow_pos = (
                int(shoulder_screen[0] + dx_elbow * (1 if is_left else 1)),
                int(shoulder_screen[1] + dy_elbow)
            )
            
            wrist_pos = (
                int(shoulder_screen[0] + dx_wrist * (1 if is_left else 1)),
                int(shoulder_screen[1] + dy_wrist)
            )
            
            return elbow_pos, wrist_pos
        
        l_elbow_pos, l_wrist_pos = get_arm_pos(l_shoulder_screen, l_elbow, l_wrist, True)
        r_elbow_pos, r_wrist_pos = get_arm_pos(r_shoulder_screen, r_elbow, r_wrist, False)
        
        # Store wrist positions for hand drawing
        self.left_wrist_pos = l_wrist_pos
        self.right_wrist_pos = r_wrist_pos
        
        # Draw arms (thick lines with circles at joints)
        # Left arm
        cv2.line(frame, l_shoulder_screen, l_elbow_pos, SKIN_COLOR, self.arm_thickness + 4)
        cv2.line(frame, l_elbow_pos, l_wrist_pos, SKIN_COLOR, self.arm_thickness + 2)
        cv2.circle(frame, l_elbow_pos, self.arm_thickness // 2 + 2, SKIN_COLOR, -1)
        
        # Right arm
        cv2.line(frame, r_shoulder_screen, r_elbow_pos, SKIN_COLOR, self.arm_thickness + 4)
        cv2.line(frame, r_elbow_pos, r_wrist_pos, SKIN_COLOR, self.arm_thickness + 2)
        cv2.circle(frame, r_elbow_pos, self.arm_thickness // 2 + 2, SKIN_COLOR, -1)
        
        # Sleeve cuffs
        cv2.circle(frame, l_shoulder_screen, self.arm_thickness // 2 + 5, SHIRT_COLOR, -1)
        cv2.circle(frame, r_shoulder_screen, self.arm_thickness // 2 + 5, SHIRT_COLOR, -1)
    
    def draw_hands(self, frame, pose, left_hand, right_hand):
        """Draw detailed hands with fingers."""
        # Use stored wrist positions from arm drawing
        if hasattr(self, 'left_wrist_pos'):
            self.draw_single_hand(frame, left_hand, self.left_wrist_pos, is_left=True)
        if hasattr(self, 'right_wrist_pos'):
            self.draw_single_hand(frame, right_hand, self.right_wrist_pos, is_left=False)
    
    def draw_single_hand(self, frame, hand_landmarks, wrist_pos, is_left=True):
        """Draw a single hand with all fingers."""
        # Check if hand data is valid
        if np.abs(hand_landmarks).sum() < 0.1:
            # Draw simple fist if no data
            cv2.circle(frame, wrist_pos, int(self.hand_scale * 0.15), SKIN_COLOR, -1)
            return
        
        # Hand landmark connections (finger chains)
        fingers = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20],  # Pinky
        ]
        
        palm = [0, 5, 9, 13, 17, 0]  # Palm outline
        
        # Scale hand landmarks relative to wrist position
        def hand_to_screen(lm_idx):
            lm = hand_landmarks[lm_idx]
            # Center around wrist, scale appropriately
            dx = (lm[0] - hand_landmarks[0][0]) * self.hand_scale
            dy = (lm[1] - hand_landmarks[0][1]) * self.hand_scale
            
            # Mirror x for left hand
            if is_left:
                dx = -dx
            
            return (int(wrist_pos[0] + dx), int(wrist_pos[1] + dy))
        
        # Draw palm
        palm_pts = [hand_to_screen(i) for i in palm]
        cv2.fillPoly(frame, [np.array(palm_pts, np.int32)], SKIN_COLOR)
        
        # Draw fingers
        for finger in fingers:
            pts = [hand_to_screen(i) for i in finger]
            
            # Draw finger segments
            for i in range(len(pts) - 1):
                thickness = max(2, int(6 - i * 1.2))  # Thinner toward tip
                cv2.line(frame, pts[i], pts[i + 1], SKIN_COLOR, thickness + 2)
            
            # Draw joints
            for i, pt in enumerate(pts):
                radius = max(2, int(5 - i))
                cv2.circle(frame, pt, radius, SKIN_COLOR, -1)
            
            # Fingertip
            cv2.circle(frame, pts[-1], 4, SKIN_OUTLINE, -1)
        
        # Palm center circle
        cv2.circle(frame, wrist_pos, int(self.hand_scale * 0.08), SKIN_OUTLINE, 1)


# ============================================================
# SPEECH RECOGNIZER
# ============================================================

class SpeechRecognizer:
    def __init__(self):
        if not SPEECH_AVAILABLE:
            self.recognizer = None
            return
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready!")
    
    def listen(self, timeout=5):
        if not self.recognizer:
            return None, "Speech recognition not available"
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            text = self.recognizer.recognize_google(audio)
            return text.lower(), None
        
        except sr.WaitTimeoutError:
            return None, "No speech detected"
        except sr.UnknownValueError:
            return None, "Could not understand"
        except sr.RequestError as e:
            return None, f"API error: {e}"
        except Exception as e:
            return None, str(e)


# ============================================================
# MAIN APPLICATION
# ============================================================

class SpeechToSignApp:
    def __init__(self):
        self.db = SignDatabase(CACHE_FILE)
        self.avatar = Avatar2D(AVATAR_CENTER_X, AVATAR_CENTER_Y, scale=280)
        self.speech = SpeechRecognizer()
        
        # State
        self.current_word = ""
        self.current_sign = None
        self.frame_index = 0
        self.is_playing = False
        self.word_queue = deque()
        self.status = "Press SPACE to speak"
        
        # Default pose (neutral standing)
        self.neutral_pose = self._create_neutral_pose()
        self.current_landmarks = self.neutral_pose.copy()
    
    def _create_neutral_pose(self):
        """Create neutral standing pose landmarks."""
        landmarks = np.zeros(225, dtype=np.float32)
        
        # Set neutral hand positions (by sides)
        # Left hand near left hip
        for i in range(21):
            landmarks[i * 3] = 0.3      # x
            landmarks[i * 3 + 1] = 0.7  # y
            landmarks[i * 3 + 2] = 0    # z
        
        # Right hand near right hip
        for i in range(21):
            landmarks[63 + i * 3] = 0.7     # x
            landmarks[63 + i * 3 + 1] = 0.7 # y
            landmarks[63 + i * 3 + 2] = 0   # z
        
        # Pose landmarks
        pose_base = 126
        # Shoulders
        landmarks[pose_base + 11 * 3] = 0.4      # left shoulder x
        landmarks[pose_base + 11 * 3 + 1] = 0.3  # left shoulder y
        landmarks[pose_base + 12 * 3] = 0.6      # right shoulder x
        landmarks[pose_base + 12 * 3 + 1] = 0.3  # right shoulder y
        
        # Elbows
        landmarks[pose_base + 13 * 3] = 0.3
        landmarks[pose_base + 13 * 3 + 1] = 0.5
        landmarks[pose_base + 14 * 3] = 0.7
        landmarks[pose_base + 14 * 3 + 1] = 0.5
        
        # Wrists
        landmarks[pose_base + 15 * 3] = 0.3
        landmarks[pose_base + 15 * 3 + 1] = 0.7
        landmarks[pose_base + 16 * 3] = 0.7
        landmarks[pose_base + 16 * 3 + 1] = 0.7
        
        return landmarks
    
    def process_text(self, text):
        words = text.lower().split()
        found_any = False
        
        for word in words:
            word = ''.join(c for c in word if c.isalpha())
            if not word:
                continue
            
            sign_data = self.db.get_sign(word)
            if sign_data is not None:
                self.word_queue.append(word)
                found_any = True
            else:
                similar = self.db.find_similar(word)
                if similar:
                    self.word_queue.append(similar[0])
                    found_any = True
                    print(f"'{word}' → '{similar[0]}'")
        
        return found_any
        
        return found_any
    
    def play_next_word(self):
        if self.word_queue:
            self.current_word = self.word_queue.popleft()
            self.current_sign = self.db.get_sign(self.current_word)
            self.frame_index = 0
            self.is_playing = True
            self.status = f"Signing: {self.current_word.upper()}"
    
    def update_animation(self):
        """Update current frame of animation."""
        if not self.is_playing or self.current_sign is None:
            # Smoothly return to neutral
            self.current_landmarks = (
                0.95 * self.current_landmarks + 
                0.05 * self.neutral_pose
            )
            return
        
        self.frame_index += 1
        
        if self.frame_index >= len(self.current_sign):
            self.is_playing = False
            self.frame_index = 0
            
            if self.word_queue:
                self.play_next_word()
            else:
                self.status = "Ready - SPACE to speak"
                self.current_word = ""
        else:
            self.current_landmarks = self.current_sign[self.frame_index]
    
    def draw_ui(self, frame):
        """Draw UI panel."""
        h, w = frame.shape[:2]
        panel_x = w - 220
        
        # Panel background
        cv2.rectangle(frame, (panel_x, 0), (w, h), (35, 35, 40), -1)
        cv2.line(frame, (panel_x, 0), (panel_x, h), (80, 80, 90), 2)
        
        # Title
        cv2.putText(frame, "SPEECH TO SIGN", (panel_x + 15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2)
        
        # Current word
        cv2.putText(frame, "Signing:", (panel_x + 15, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        word_display = self.current_word.upper() if self.current_word else "---"
        cv2.putText(frame, word_display, (panel_x + 15, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 200), 2)
        
        # Progress bar
        if self.is_playing and self.current_sign is not None:
            progress = self.frame_index / len(self.current_sign)
            bar_w = 190
            cv2.rectangle(frame, (panel_x + 15, 135), (panel_x + 15 + bar_w, 150), (60, 60, 70), -1)
            cv2.rectangle(frame, (panel_x + 15, 135), (panel_x + 15 + int(bar_w * progress), 150), (100, 255, 200), -1)
        
        # Queue
        cv2.putText(frame, f"Queue: {len(self.word_queue)}", (panel_x + 15, 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y = 210
        for word in list(self.word_queue)[:6]:
            cv2.putText(frame, f"• {word}", (panel_x + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 160), 1)
            y += 22
        
        # Controls
        cv2.putText(frame, "─── Controls ───", (panel_x + 15, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 160), 1)
        
        controls = [
            ("SPACE", "Voice input"),
            ("ENTER", "Type text"),
            ("R", "Replay"),
            ("C", "Clear queue"),
            ("S", "Slower"),
            ("F", "Faster"),
            ("L", "List signs"),
            ("Q", "Quit")
        ]
        
        y = 390
        for key, desc in controls:
            cv2.putText(frame, f"{key}", (panel_x + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 220, 255), 1)
            cv2.putText(frame, f"- {desc}", (panel_x + 70, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1)
            y += 22
        
        # Status
        cv2.putText(frame, self.status[:28], (panel_x + 15, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1)
        
        # Speed indicator
        cv2.putText(frame, f"Speed: {ANIMATION_SPEED:.1f}x", (panel_x + 15, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1)
        
        return frame
    
    def run(self):
        global ANIMATION_SPEED
        
        print("\n" + "=" * 50)
        print("SPEECH TO SIGN LANGUAGE - AVATAR MODE")
        print("=" * 50)
        print(f"Available signs: {len(self.db.classes)}")
        print("\nControls: SPACE=speak, ENTER=type, Q=quit")
        print("=" * 50 + "\n")
        
        cv2.namedWindow("Speech to Sign Avatar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Speech to Sign Avatar", WINDOW_WIDTH, WINDOW_HEIGHT)
        
        frame_delay = 1.0 / FPS
        last_time = time.time()
        
        while True:
            # Create frame with gradient background
            frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
            
            # Gradient background
            for y in range(WINDOW_HEIGHT):
                ratio = y / WINDOW_HEIGHT
                color = (
                    int(40 + ratio * 20),
                    int(45 + ratio * 25),
                    int(55 + ratio * 30)
                )
                frame[y, :WINDOW_WIDTH - 220] = color
            
            # Update animation
            current_time = time.time()
            if current_time - last_time >= frame_delay / ANIMATION_SPEED:
                self.update_animation()
                last_time = current_time
            
            # Draw avatar
            self.avatar.draw(frame, self.current_landmarks)
            
            # Draw UI
            self.draw_ui(frame)
            
            cv2.imshow("Speech to Sign Avatar", frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):
                self.status = "Listening..."
                cv2.imshow("Speech to Sign Avatar", frame)
                cv2.waitKey(1)
                
                text, error = self.speech.listen(timeout=5)
                if text:
                    print(f"Heard: {text}")
                    if self.process_text(text):
                        if not self.is_playing:
                            self.play_next_word()
                    else:
                        self.status = "No matching signs found"
                else:
                    self.status = error or "Try again"
            
            elif key == 13:  # Enter
                self.status = "Check console..."
                print("\nType your text:")
                text = input("> ")
                if text and self.process_text(text):
                    if not self.is_playing:
                        self.play_next_word()
            
            elif key == ord('r'):
                if self.current_sign is not None:
                    self.frame_index = 0
                    self.is_playing = True
            
            elif key == ord('c'):
                self.word_queue.clear()
                self.is_playing = False
                self.current_word = ""
                self.status = "Cleared"
            
            elif key == ord('s'):
                ANIMATION_SPEED = max(0.25, ANIMATION_SPEED - 0.25)
            
            elif key == ord('f'):
                ANIMATION_SPEED = min(3.0, ANIMATION_SPEED + 0.25)
            
            elif key == ord('l'):
                print("\n=== Available Signs ===")
                for i, sign in enumerate(sorted(self.db.classes)):
                    print(f"{sign:15}", end="")
                    if (i + 1) % 6 == 0:
                        print()
                print("\n")
            
            # Auto-play next
            if not self.is_playing and self.word_queue:
                self.play_next_word()
        
        cv2.destroyAllWindows()
        print("Goodbye!")


if __name__ == "__main__":
    app = SpeechToSignApp()
    app.run()
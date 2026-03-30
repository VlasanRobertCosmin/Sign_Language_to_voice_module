"""
Sign Language Translator Application (OpenCV UI)
=================================================
A complete application with two modules:
1. Sign to Voice - Recognizes ASL signs and speaks them
2. Voice to Sign - Converts speech to animated sign language

Requirements:
    pip install numpy opencv-python torch mediapipe Pillow
    pip install SpeechRecognition pyaudio pyttsx3
"""

import os
import sys
import threading
import time
import numpy as np
import cv2
from collections import deque
import math
import pickle

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

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - Sign to Voice disabled")

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except:
    SPEECH_AVAILABLE = False
    print("SpeechRecognition not available - Voice input disabled")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except:
    TTS_AVAILABLE = False
    print("pyttsx3 not available - Voice output disabled")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - Sign to Voice disabled")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "asl_signs_model_v3.pth"
ENCODER_PATH = "asl_signs_encoder_v3.pkl"
CACHE_FILE = "asl_signs_cache.npz"

# UI dimensions
WIN_W = 1280
WIN_H = 720
VIDEO_W = 820
PANEL_W = WIN_W - VIDEO_W  # 460

# Colors (BGR for OpenCV)
BG_COLOR       = (43, 43, 43)
PANEL_COLOR    = (54, 54, 54)
ACCENT_COLOR   = (255, 158, 74)   # orange-ish
SUCCESS_COLOR  = (127, 255, 74)   # green
ERROR_COLOR    = (107, 107, 255)  # red-ish
TEXT_COLOR     = (240, 240, 240)
DIM_COLOR      = (140, 140, 140)
DARK_COLOR     = (30, 30, 30)
HIGHLIGHT      = (255, 200, 80)


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_rect(img, x, y, w, h, color, radius=8, alpha=1.0):
    """Filled rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in [(x+radius, y+radius), (x+w-radius, y+radius),
                   (x+radius, y+h-radius), (x+w-radius, y+h-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_text(img, text, x, y, scale=0.6, color=TEXT_COLOR, thickness=1, center=False, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = x - tw // 2 if center else x
    cv2.putText(img, text, (tx, y), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def draw_button(img, label, x, y, w, h, color, text_color=DARK_COLOR, active=False):
    border_color = HIGHLIGHT if active else color
    draw_rect(img, x, y, w, h, color)
    if active:
        cv2.rectangle(img, (x, y), (x+w, y+h), border_color, 2)
    draw_text(img, label, x + w//2, y + h//2 + 6, scale=0.55, color=text_color,
              thickness=1, center=True)
    return (x, y, w, h)  # hit area


def draw_badge(img, text, x, y, color):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    pad = 6
    draw_rect(img, x, y - 14, tw + pad*2, 20, color, radius=5)
    draw_text(img, text, x + pad, y, scale=0.45, color=DARK_COLOR)


# ============================================================
# MODEL DEFINITION
# ============================================================

if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class ConvSubsampling(nn.Module):
        def __init__(self, d_model, dropout=0.1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            residual = x
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
            return self.norm(x + residual)

    class ASLModelV3(nn.Module):
        def __init__(self, input_size, num_classes, d_model=384, n_heads=8,
                     n_layers=6, dim_ff=1536, dropout=0.4, max_frames=64):
            super().__init__()
            self.d_model = d_model
            self.input_norm = nn.LayerNorm(input_size)
            self.input_proj1 = nn.Linear(input_size, d_model)
            self.input_proj2 = nn.Sequential(nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(d_model, d_model))
            self.input_ln = nn.LayerNorm(d_model)
            self.conv_subsample = ConvSubsampling(d_model, dropout * 0.5)
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames, dropout=dropout * 0.5)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.trans_ln = nn.LayerNorm(d_model)
            self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
            self.lstm_ln = nn.LayerNorm(d_model)
            self.n_queries = 4
            self.pool_queries = nn.Parameter(torch.randn(1, self.n_queries, d_model * 2))
            self.pool_attention = nn.MultiheadAttention(embed_dim=d_model * 2, num_heads=n_heads, dropout=dropout, batch_first=True)
            self.pool_ln = nn.LayerNorm(d_model * 2)
            self.classifier = nn.Sequential(
                nn.Linear(d_model * 2 * self.n_queries, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout * 0.7),
                nn.Linear(384, num_classes))

        def forward(self, x):
            batch_size = x.size(0)
            x = self.input_norm(x)
            x = self.input_proj1(x)
            x = self.input_proj2(x) + x
            x = self.input_ln(x)
            x = self.conv_subsample(x)
            x_pos = self.pos_encoder(x)
            x_trans = self.transformer(x_pos)
            x_trans = self.trans_ln(x_trans)
            x_lstm, _ = self.lstm(x)
            x_lstm = self.lstm_ln(x_lstm)
            combined = torch.cat([x_trans, x_lstm], dim=-1)
            queries = self.pool_queries.expand(batch_size, -1, -1)
            pooled, _ = self.pool_attention(queries, combined, combined)
            pooled = self.pool_ln(pooled)
            pooled = pooled.view(batch_size, -1)
            return self.classifier(pooled)


# ============================================================
# TEXT TO SPEECH
# ============================================================

class TextToSpeech:
    def __init__(self):
        self.engine = None
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
            except:
                self.engine = None

    def speak(self, text):
        if self.engine:
            def _speak():
                self.engine.say(text)
                self.engine.runAndWait()
            threading.Thread(target=_speak, daemon=True).start()
        else:
            print(f"[TTS]: {text}")


# ============================================================
# SPEECH RECOGNIZER
# ============================================================

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        if SPEECH_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except:
                self.recognizer = None

    def listen(self, timeout=5):
        if not self.recognizer:
            return None, "Microphone not available"
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
# SIGN DATABASE
# ============================================================

class SignDatabase:
    def __init__(self, cache_file):
        self.signs = {}
        self.classes = []
        self.load_data(cache_file)

    def interpolate_frames(self, frames, factor=3):
        if len(frames) < 2:
            return frames
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            for j in range(1, factor):
                t = j / factor
                interpolated.append(frames[i] * (1 - t) + frames[i + 1] * t)
        interpolated.append(frames[-1])
        return np.array(interpolated)

    def load_data(self, cache_file):
        if not os.path.exists(cache_file):
            return
        cache = np.load(cache_file, allow_pickle=True)
        X, y = cache['X'], cache['y']
        self.classes = sorted(list(set(y)))
        for i, label in enumerate(y):
            if label not in self.signs:
                seq = X[i]
                valid = [seq[f] for f in range(len(seq)) if np.abs(seq[f]).sum() > 0.1]
                if valid:
                    self.signs[label] = self.interpolate_frames(np.array(valid), factor=3)

    def get_sign(self, word):
        return self.signs.get(word.lower().strip())

    def find_similar(self, word):
        word = word.lower().strip()
        if word in self.signs:
            return [word]
        return [s for s in self.classes if word in s or s in word][:3]


# ============================================================
# AVATAR DRAWING
# ============================================================

def draw_body(img, pose, cx, cy, scale):
    body_color = (120, 130, 150)
    shirt_color = (180, 100, 80)
    shoulder_y = cy - int(scale * 0.1)
    shoulder_w = int(scale * 0.25)
    pts = np.array([
        [cx - shoulder_w, shoulder_y],
        [cx + shoulder_w, shoulder_y],
        [cx + shoulder_w + 20, cy + int(scale * 0.3)],
        [cx - shoulder_w - 20, cy + int(scale * 0.3)]
    ], np.int32)
    cv2.fillPoly(img, [pts], shirt_color)
    cv2.rectangle(img, (cx - 15, shoulder_y - 30), (cx + 15, shoulder_y), body_color, -1)
    l_wrist = pose[15]
    r_wrist = pose[16]
    lw = (int(cx + (l_wrist[0] - 0.5) * scale * 1.5), int(cy + (l_wrist[1] - 0.5) * scale))
    rw = (int(cx + (r_wrist[0] - 0.5) * scale * 1.5), int(cy + (r_wrist[1] - 0.5) * scale))
    cv2.line(img, (cx - shoulder_w, shoulder_y), lw, body_color, 18)
    cv2.circle(img, lw, 20, body_color, -1)
    cv2.line(img, (cx + shoulder_w, shoulder_y), rw, body_color, 18)
    cv2.circle(img, rw, 20, body_color, -1)


def draw_hand(img, hand, cx, cy, scale):
    body_color = (120, 130, 150)
    finger_chains = [
        [0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]
    ]
    points = [(int(cx + (hand[i][0] - 0.5)*scale*1.5), int(cy + (hand[i][1] - 0.5)*scale)) for i in range(21)]
    for finger in finger_chains:
        for i in range(len(finger) - 1):
            cv2.line(img, points[finger[i]], points[finger[i+1]], body_color, max(4, 12 - i*2))
        for f in finger:
            cv2.circle(img, points[f], 6, body_color, -1)


def draw_head(img, cx, cy, radius):
    body_color = (120, 130, 150)
    hair_color = (40, 30, 25)
    cv2.circle(img, (cx, cy), radius, body_color, -1)
    cv2.ellipse(img, (cx, cy - radius//4), (radius, radius//2), 0, 180, 360, hair_color, -1)
    eye_y = cy - radius // 6
    for ex in [cx - radius//3, cx + radius//3]:
        cv2.circle(img, (ex, eye_y), 8, (255, 255, 255), -1)
        cv2.circle(img, (ex, eye_y), 4, (40, 30, 20), -1)
    cv2.ellipse(img, (cx, cy + radius//3), (10, 5), 0, 0, 180, (100, 80, 80), 2)


def render_avatar(landmarks, w, h):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cx, cy = w // 2, h // 2 + 30
    scale = min(w, h) * 0.5
    left_hand  = landmarks[0:63].reshape(21, 3)
    right_hand = landmarks[63:126].reshape(21, 3)
    pose       = landmarks[126:225].reshape(33, 3)
    draw_body(img, pose, cx, cy, scale)
    if np.abs(right_hand).sum() > 0.5:
        draw_hand(img, right_hand, cx, cy, scale)
    if np.abs(left_hand).sum() > 0.5:
        draw_hand(img, left_hand, cx, cy, scale)
    draw_head(img, cx, cy - int(scale * 0.4), int(scale * 0.15))
    return img


def create_neutral_pose():
    lm = np.zeros(225, dtype=np.float32)
    ps = 126
    for idx, x, y in [(11, 0.7, 0.4), (12, 0.3, 0.4), (13, 0.8, 0.6),
                       (14, 0.2, 0.6), (15, 0.85, 0.8), (16, 0.15, 0.8)]:
        lm[ps + idx*3], lm[ps + idx*3 + 1] = x, y
    return lm


# ============================================================
# MAIN APPLICATION
# ============================================================

class SignLanguageApp:
    def __init__(self):
        self.tts   = TextToSpeech()
        self.speech = SpeechRecognizer()
        self.sign_db = SignDatabase(CACHE_FILE)

        # Model
        self.model = None
        self.label_encoder = None
        self.max_frames = 64
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if TORCH_AVAILABLE else None)

        # MediaPipe
        self.mp_holistic = None
        self.holistic = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # State
        self.mode = "sign_to_voice"   # or "voice_to_sign"
        self.is_running = False
        self.cap = None
        self.frame_buffer = deque(maxlen=64)
        self.status_msg   = "Ready"
        self.status_color = SUCCESS_COLOR
        self.detected     = "---"
        self.confidence   = ""
        self.history      = []
        self.listening    = False
        self.text_input   = ""
        self.text_focused = False

        # Avatar
        self.avatar_lm      = create_neutral_pose()
        self.sign_queue     = deque()
        self.current_sign   = None
        self.sign_frame_idx = 0
        self.is_signing     = False

        # Latest camera frame
        self._cam_frame = None
        self._lock = threading.Lock()

        self.load_model()
        self.set_status(f"Model: {self.status_msg}", self.status_color)

        # Define button hit regions (x, y, w, h)
        self.btn_sign_to_voice = (VIDEO_W + 10,  60, 210, 38)
        self.btn_voice_to_sign = (VIDEO_W + 230, 60, 210, 38)
        self.btn_start_stop    = (VIDEO_W + 50, 390, 150, 44)
        self.btn_speak         = (VIDEO_W + 260, 390, 150, 44)
        self.btn_submit_text   = (VIDEO_W + 360, 560, 80, 32)
        self.text_box_region   = (VIDEO_W + 10, 540, 340, 34)

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------

    def load_model(self):
        if not TORCH_AVAILABLE:
            self.set_status("PyTorch not available", ERROR_COLOR); return
        if not os.path.exists(MODEL_PATH):
            self.set_status(f"Model not found: {MODEL_PATH}", ERROR_COLOR); return
        try:
            ck = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            self.max_frames = ck.get('max_frames', 64)
            self.model = ASLModelV3(ck['input_size'], ck['num_classes'],
                                    max_frames=self.max_frames).to(self.device)
            self.model.load_state_dict(ck['model_state_dict'])
            self.model.eval()
            with open(ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.set_status(f"Model loaded ({ck['num_classes']} signs)", SUCCESS_COLOR)
        except Exception as e:
            self.set_status(f"Error: {str(e)[:40]}", ERROR_COLOR)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def set_status(self, msg, color=SUCCESS_COLOR):
        self.status_msg   = msg
        self.status_color = color

    def add_history(self, text):
        self.history.append(text)
        if len(self.history) > 20:
            self.history.pop(0)

    # ----------------------------------------------------------
    # Camera thread
    # ----------------------------------------------------------

    def _camera_thread(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.set_status("Cannot open camera", ERROR_COLOR)
            self.is_running = False
            return
        self.frame_buffer.clear()
        while self.is_running and self.mode == "sign_to_voice":
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.holistic:
                results = self.holistic.process(frame_rgb)
                lm = self.extract_landmarks(results)
                self.frame_buffer.append(lm)
                frame = self.draw_landmarks(frame, results)
                if len(self.frame_buffer) >= self.max_frames:
                    self.predict_sign()
                    self.frame_buffer.clear()
            with self._lock:
                self._cam_frame = frame.copy()
            time.sleep(0.03)
        if self.cap:
            self.cap.release()
            self.cap = None

    def extract_landmarks(self, results):
        lm = []
        for attr, n in [('left_hand_landmarks', 63), ('right_hand_landmarks', 63),
                        ('pose_landmarks', 99)]:
            src = getattr(results, attr)
            if src:
                for p in src.landmark:
                    lm.extend([p.x, p.y, p.z])
            else:
                lm.extend([0.0] * n)
        return np.array(lm, dtype=np.float32)

    def draw_landmarks(self, frame, results):
        if MEDIAPIPE_AVAILABLE:
            mp_draw = mp.solutions.drawing_utils
            for attr in ('left_hand_landmarks', 'right_hand_landmarks'):
                src = getattr(results, attr)
                if src:
                    mp_draw.draw_landmarks(frame, src, self.mp_holistic.HAND_CONNECTIONS)
        return frame

    def predict_sign(self):
        if not self.model or not TORCH_AVAILABLE:
            return
        seq = np.array(list(self.frame_buffer), dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        if len(seq) < self.max_frames:
            pad = np.zeros((self.max_frames - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out   = self.model(t)
            probs = torch.softmax(out, dim=1)[0]
            conf, idx = torch.max(probs, 0)
            word = self.label_encoder.inverse_transform([idx.item()])[0]
            confidence = conf.item()
        if confidence > 0.4:
            self.detected   = word.upper()
            self.confidence = f"Confidence: {confidence*100:.1f}%"
            self.add_history(f"[Sign→Voice] {word}")
            self.tts.speak(word)

    # ----------------------------------------------------------
    # Avatar animation
    # ----------------------------------------------------------

    def step_avatar(self):
        if self.is_signing and self.current_sign is not None:
            if self.sign_frame_idx < len(self.current_sign):
                target = self.current_sign[self.sign_frame_idx]
                self.avatar_lm = 0.3 * target + 0.7 * self.avatar_lm
                self.sign_frame_idx += 1
            else:
                self.is_signing = False
                self.sign_frame_idx = 0
                if self.sign_queue:
                    self.play_next_sign()
                else:
                    self.detected = "---"
        else:
            neutral = create_neutral_pose()
            self.avatar_lm = 0.95 * self.avatar_lm + 0.05 * neutral

    def play_next_sign(self):
        if self.sign_queue:
            word = self.sign_queue.popleft()
            self.current_sign   = self.sign_db.get_sign(word)
            self.sign_frame_idx = 0
            self.is_signing     = True
            self.detected       = word.upper()

    # ----------------------------------------------------------
    # Speech thread
    # ----------------------------------------------------------

    def _listen_thread(self):
        self.set_status("Listening...", ACCENT_COLOR)
        text, error = self.speech.listen()
        self.listening = False
        if text:
            self.process_text(text)
        else:
            self.set_status(error, ERROR_COLOR)

    def process_text(self, text):
        self.set_status(f"Heard: {text}", SUCCESS_COLOR)
        self.add_history(f"[Voice→Sign] {text}")
        for word in text.lower().split():
            word = ''.join(c for c in word if c.isalpha())
            if not word:
                continue
            if self.sign_db.get_sign(word) is not None:
                self.sign_queue.append(word)
            else:
                similar = self.sign_db.find_similar(word)
                if similar:
                    self.sign_queue.append(similar[0])
        if self.sign_queue and not self.is_signing:
            self.play_next_sign()

    # ----------------------------------------------------------
    # UI rendering
    # ----------------------------------------------------------

    def render_panel(self, panel):
        """Draw the right control panel onto a (WIN_H x PANEL_W) image."""
        h, w = panel.shape[:2]
        panel[:] = np.array(PANEL_COLOR, dtype=np.uint8)

        # ---- Title ----
        draw_text(panel, "Sign Language", w//2, 28, scale=0.75,
                  color=TEXT_COLOR, thickness=2, center=True)
        draw_text(panel, "Translator", w//2, 52, scale=0.65,
                  color=ACCENT_COLOR, thickness=1, center=True)

        # ---- Mode buttons ----
        bx1, bx2 = 10, 230
        by = 70
        bw, bh = 210, 38
        draw_button(panel, "Sign -> Voice", bx1, by, bw, bh,
                    ACCENT_COLOR if self.mode == "sign_to_voice" else (80,80,80),
                    active=(self.mode == "sign_to_voice"))
        draw_button(panel, "Voice -> Sign", bx2, by, bw, bh,
                    ACCENT_COLOR if self.mode == "voice_to_sign" else (80,80,80),
                    active=(self.mode == "voice_to_sign"))

        # ---- Separator ----
        cv2.line(panel, (10, 120), (w-10, 120), (70,70,70), 1)

        # ---- Status ----
        draw_text(panel, "STATUS", 10, 145, scale=0.42, color=DIM_COLOR, thickness=1)
        msg = self.status_msg[:48]
        draw_text(panel, msg, 10, 165, scale=0.5, color=self.status_color)

        # ---- Detected sign ----
        cv2.line(panel, (10, 182), (w-10, 182), (70,70,70), 1)
        draw_text(panel, "DETECTED SIGN", 10, 202, scale=0.42, color=DIM_COLOR)
        det_scale = min(1.4, 14.0 / max(len(self.detected), 1))
        draw_text(panel, self.detected, w//2, 255, scale=det_scale,
                  color=HIGHLIGHT, thickness=2, center=True)
        draw_text(panel, self.confidence, w//2, 280, scale=0.45,
                  color=DIM_COLOR, center=True)

        # ---- Start/Stop + Speak buttons ----
        cv2.line(panel, (10, 300), (w-10, 300), (70,70,70), 1)
        sx, sy, sw, sh = 50, 320, 150, 44
        running_lbl = "[ STOP ]" if self.is_running else "[ START ]"
        running_col = ERROR_COLOR if self.is_running else SUCCESS_COLOR
        draw_button(panel, running_lbl, sx, sy, sw, sh, running_col)

        if self.mode == "voice_to_sign":
            lx, ly, lw2, lh = 260, 320, 150, 44
            listen_col = ACCENT_COLOR if not self.listening else (200, 200, 50)
            listen_lbl = "LISTENING..." if self.listening else "MIC SPEAK"
            draw_button(panel, listen_lbl, lx, ly, lw2, lh, listen_col)

        # ---- Text input (Voice→Sign mode) ----
        if self.mode == "voice_to_sign":
            cv2.line(panel, (10, 375), (w-10, 375), (70,70,70), 1)
            draw_text(panel, "TYPE TEXT  (Enter to send):", 10, 395, scale=0.42, color=DIM_COLOR)
            tbx, tby, tbw, tbh = 10, 408, 420, 34
            box_color = (60, 70, 80) if self.text_focused else (50, 50, 50)
            draw_rect(panel, tbx, tby, tbw, tbh, box_color, radius=5)
            if self.text_focused:
                cv2.rectangle(panel, (tbx, tby), (tbx+tbw, tby+tbh), ACCENT_COLOR, 1)
            display_text = self.text_input[-38:] + ("|" if self.text_focused else "")
            draw_text(panel, display_text, tbx+8, tby+23, scale=0.52, color=TEXT_COLOR)

        # ---- History ----
        hist_y = 460 if self.mode == "voice_to_sign" else 380
        cv2.line(panel, (10, hist_y - 18), (w-10, hist_y - 18), (70,70,70), 1)
        draw_text(panel, "HISTORY", 10, hist_y - 2, scale=0.42, color=DIM_COLOR)
        for i, entry in enumerate(self.history[-10:][::-1]):
            col = ACCENT_COLOR if "Sign" in entry else SUCCESS_COLOR
            draw_text(panel, entry[:50], 10, hist_y + 18 + i*20, scale=0.42, color=col)

    def render_frame(self):
        canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

        # ---- Left: video or avatar ----
        if self.mode == "sign_to_voice":
            with self._lock:
                frame = self._cam_frame.copy() if self._cam_frame is not None else None
            if frame is not None:
                resized = cv2.resize(frame, (VIDEO_W, WIN_H))
                canvas[:, :VIDEO_W] = resized
            else:
                draw_text(canvas, "Camera not started", VIDEO_W//2, WIN_H//2,
                          scale=0.8, color=DIM_COLOR, center=True)
                draw_text(canvas, "Press START", VIDEO_W//2, WIN_H//2 + 35,
                          scale=0.6, color=DIM_COLOR, center=True)
        else:
            self.step_avatar()
            avatar = render_avatar(self.avatar_lm, VIDEO_W, WIN_H)
            canvas[:, :VIDEO_W] = avatar

        # Divider
        cv2.line(canvas, (VIDEO_W, 0), (VIDEO_W, WIN_H), (70,70,70), 2)

        # ---- Right: control panel ----
        panel = canvas[:, VIDEO_W:]
        self.render_panel(panel)

        return canvas

    # ----------------------------------------------------------
    # Mouse callback
    # ----------------------------------------------------------

    def _in_btn(self, x, y, btn):
        bx, by, bw, bh = btn
        return bx <= x <= bx + bw and by <= y <= by + bh

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Translate panel coordinates
        px = x - VIDEO_W   # x within panel

        # Mode buttons (panel-relative)
        if self._in_btn(px, y, (10, 70, 210, 38)):
            self.switch_mode("sign_to_voice")
        elif self._in_btn(px, y, (230, 70, 210, 38)):
            self.switch_mode("voice_to_sign")
        # Start/stop
        elif self._in_btn(px, y, (50, 320, 150, 44)):
            self.toggle_running()
        # Speak (voice_to_sign mode only)
        elif self.mode == "voice_to_sign" and self._in_btn(px, y, (260, 320, 150, 44)):
            if not self.listening:
                self.listening = True
                threading.Thread(target=self._listen_thread, daemon=True).start()
        # Text box
        elif self.mode == "voice_to_sign" and self._in_btn(px, y, (10, 408, 420, 34)):
            self.text_focused = True
        else:
            self.text_focused = False

    # ----------------------------------------------------------
    # Keyboard handling
    # ----------------------------------------------------------

    def on_key(self, key):
        if key == -1:
            return
        key = key & 0xFF

        if self.text_focused and self.mode == "voice_to_sign":
            if key == 13:  # Enter
                if self.text_input.strip():
                    self.process_text(self.text_input.strip())
                    self.text_input = ""
            elif key == 8:  # Backspace
                self.text_input = self.text_input[:-1]
            elif 32 <= key <= 126:
                self.text_input += chr(key)
        else:
            if key == ord('q'):
                return True   # signal quit
            elif key == ord('s'):
                self.toggle_running()
            elif key == ord('1'):
                self.switch_mode("sign_to_voice")
            elif key == ord('2'):
                self.switch_mode("voice_to_sign")
            elif key == ord('m') and self.mode == "voice_to_sign":
                if not self.listening:
                    self.listening = True
                    threading.Thread(target=self._listen_thread, daemon=True).start()
        return False

    # ----------------------------------------------------------
    # Mode switching / start/stop
    # ----------------------------------------------------------

    def switch_mode(self, mode):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._cam_frame = None
        self.mode = mode
        self.detected   = "---"
        self.confidence = ""
        self.set_status(f"Mode: {mode.replace('_', ' ').title()}", SUCCESS_COLOR)

    def toggle_running(self):
        if self.is_running:
            self.is_running = False
            self.set_status("Stopped", DIM_COLOR)
        else:
            self.is_running = True
            self.set_status("Running...", SUCCESS_COLOR)
            if self.mode == "sign_to_voice":
                threading.Thread(target=self._camera_thread, daemon=True).start()
            else:
                # Avatar mode just ticks in render loop
                pass

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------

    def run(self):
        cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sign Language Translator", WIN_W, WIN_H)
        cv2.setMouseCallback("Sign Language Translator", self.on_mouse)

        print("Controls: Q=quit | S=start/stop | 1=Sign->Voice | 2=Voice->Sign | M=mic")

        while True:
            frame = self.render_frame()
            cv2.imshow("Sign Language Translator", frame)

            key = cv2.waitKey(33)
            quit_requested = self.on_key(key)
            if quit_requested:
                break

        # Cleanup
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.holistic:
            self.holistic.close()
        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()
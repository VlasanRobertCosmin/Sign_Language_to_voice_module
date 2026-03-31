"""
Sign Language Translator Application (PyQt5 Version)
=====================================================
A complete application with two modules:
1. Sign to Voice - Recognizes ASL signs and speaks them
2. Voice to Sign - Converts speech to animated sign language

Requirements:
    pip install PyQt5 numpy opencv-python torch mediapipe
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

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit, QFrame, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush

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


# ============================================================
# STYLESHEET
# ============================================================

STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
}
QLabel {
    color: #ffffff;
}
QPushButton {
    background-color: #3a3a3a;
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #4a4a4a;
}
QPushButton:pressed {
    background-color: #2a2a2a;
}
QPushButton#activeMode {
    background-color: #4a9eff;
}
QPushButton#startBtn {
    background-color: #4aff7f;
    color: #000000;
    font-weight: bold;
    font-size: 16px;
    padding: 15px 40px;
}
QPushButton#startBtn:hover {
    background-color: #3aef6f;
}
QPushButton#stopBtn {
    background-color: #ff6b6b;
    color: #ffffff;
}
QPushButton#speakBtn {
    background-color: #4a9eff;
}
QLineEdit {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    padding: 10px;
    border-radius: 5px;
    color: #ffffff;
}
QTextEdit {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 5px;
    color: #ffffff;
    padding: 5px;
}
QFrame#panel {
    background-color: #2a2a2a;
    border-radius: 10px;
}
"""


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
            return self.dropout(x + self.pe[:, :x.size(1)])

    class ConvSubsampling(nn.Module):
        def __init__(self, d_model, dropout=0.1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout))
            self.norm = nn.LayerNorm(d_model)
        
        def forward(self, x):
            return self.norm(x + self.conv(x.transpose(1, 2)).transpose(1, 2))

    class ASLModelV3(nn.Module):
        def __init__(self, input_size, num_classes, d_model=384, n_heads=8, n_layers=6, dim_ff=1536, dropout=0.4, max_frames=64):
            super().__init__()
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
                nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout * 0.7), nn.Linear(384, num_classes))
        
        def forward(self, x):
            batch_size = x.size(0)
            x = self.input_ln(self.input_proj2(self.input_proj1(self.input_norm(x))) + self.input_proj1(self.input_norm(x)))
            x = self.conv_subsample(x)
            x_trans = self.trans_ln(self.transformer(self.pos_encoder(x)))
            x_lstm, _ = self.lstm(x)
            x_lstm = self.lstm_ln(x_lstm)
            combined = torch.cat([x_trans, x_lstm], dim=-1)
            pooled, _ = self.pool_attention(self.pool_queries.expand(batch_size, -1, -1), combined, combined)
            return self.classifier(self.pool_ln(pooled).view(batch_size, -1))


# ============================================================
# HELPER CLASSES
# ============================================================

class TextToSpeech:
    def __init__(self):
        self.engine = None
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
            except:
                pass
    
    def speak(self, text):
        if self.engine:
            threading.Thread(target=lambda: (self.engine.say(text), self.engine.runAndWait()), daemon=True).start()


class SpeechRecognizer:
    def __init__(self):
        self.recognizer = None
        if SPEECH_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except:
                pass
    
    def listen(self, timeout=5):
        if not self.recognizer:
            return None, "No microphone"
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            return self.recognizer.recognize_google(audio).lower(), None
        except sr.WaitTimeoutError:
            return None, "No speech detected"
        except sr.UnknownValueError:
            return None, "Could not understand"
        except Exception as e:
            return None, str(e)


class SignDatabase:
    def __init__(self, cache_file):
        self.signs = {}
        self.classes = []
        if os.path.exists(cache_file):
            cache = np.load(cache_file, allow_pickle=True)
            X, y = cache['X'], cache['y']
            self.classes = sorted(list(set(y)))
            for i, label in enumerate(y):
                if label not in self.signs:
                    seq = X[i]
                    valid = [seq[f] for f in range(len(seq)) if np.abs(seq[f]).sum() > 0.1]
                    if valid:
                        self.signs[label] = self.interpolate(np.array(valid))
    
    def interpolate(self, frames, factor=3):
        if len(frames) < 2:
            return frames
        result = []
        for i in range(len(frames) - 1):
            result.append(frames[i])
            for j in range(1, factor):
                t = j / factor
                result.append(frames[i] * (1 - t) + frames[i + 1] * t)
        result.append(frames[-1])
        return np.array(result)
    
    def get_sign(self, word):
        return self.signs.get(word.lower().strip())
    
    def find_similar(self, word):
        word = word.lower().strip()
        if word in self.signs:
            return [word]
        return [s for s in self.classes if word in s or s in word][:3]


# ============================================================
# SIGNAL EMITTER FOR THREAD COMMUNICATION
# ============================================================

class SignalEmitter(QObject):
    update_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(str)
    update_detection = pyqtSignal(str, float)
    speech_result = pyqtSignal(str)


# ============================================================
# MAIN WINDOW
# ============================================================

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤟 Sign Language Translator")
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet(STYLESHEET)
        
        # Components
        self.tts = TextToSpeech()
        self.speech = SpeechRecognizer()
        self.sign_db = SignDatabase(CACHE_FILE)
        self.signals = SignalEmitter()
        
        # Model
        self.model = None
        self.label_encoder = None
        self.max_frames = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        # MediaPipe
        self.holistic = None
        if MEDIAPIPE_AVAILABLE:
            self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # State
        self.current_mode = "sign_to_voice"
        self.is_running = False
        self.cap = None
        self.frame_buffer = deque(maxlen=64)
        
        # Avatar state
        self.avatar_landmarks = self.create_neutral_pose()
        self.sign_queue = deque()
        self.current_sign = None
        self.sign_frame_idx = 0
        self.is_signing = False
        
        # Timers
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        self.avatar_timer = QTimer()
        self.avatar_timer.timeout.connect(self.update_avatar)
        
        # Connect signals
        self.signals.update_frame.connect(self.display_frame)
        self.signals.update_status.connect(self.set_status)
        self.signals.update_detection.connect(self.set_detection)
        self.signals.speech_result.connect(self.process_speech)
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🤟 Sign Language Translator")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        header.addWidget(title)
        header.addStretch()
        
        # Mode buttons
        self.btn_sign_to_voice = QPushButton("📹 Sign → Voice")
        self.btn_sign_to_voice.setObjectName("activeMode")
        self.btn_sign_to_voice.clicked.connect(lambda: self.switch_mode("sign_to_voice"))
        
        self.btn_voice_to_sign = QPushButton("🎤 Voice → Sign")
        self.btn_voice_to_sign.clicked.connect(lambda: self.switch_mode("voice_to_sign"))
        
        header.addWidget(self.btn_sign_to_voice)
        header.addWidget(self.btn_voice_to_sign)
        main_layout.addLayout(header)
        
        # Content
        content = QHBoxLayout()
        content.setSpacing(15)
        
        # Display area
        display_frame = QFrame()
        display_frame.setObjectName("panel")
        display_frame.setMinimumSize(700, 500)
        display_layout = QVBoxLayout(display_frame)
        
        self.display_label = QLabel("Press Start to begin")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #1a1a1a; border-radius: 5px;")
        self.display_label.setMinimumSize(680, 480)
        display_layout.addWidget(self.display_label)
        
        content.addWidget(display_frame, stretch=2)
        
        # Control panel
        control_frame = QFrame()
        control_frame.setObjectName("panel")
        control_frame.setFixedWidth(350)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(20, 20, 20, 20)
        
        # Status
        status_title = QLabel("Status")
        status_title.setFont(QFont("Arial", 14, QFont.Bold))
        control_layout.addWidget(status_title)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #4aff7f;")
        control_layout.addWidget(self.status_label)
        
        # Detection
        detect_title = QLabel("Detected Sign")
        detect_title.setFont(QFont("Arial", 14, QFont.Bold))
        control_layout.addWidget(detect_title)
        
        self.detection_label = QLabel("---")
        self.detection_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.detection_label.setStyleSheet("color: #4a9eff;")
        control_layout.addWidget(self.detection_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setStyleSheet("color: #888888;")
        control_layout.addWidget(self.confidence_label)
        
        control_layout.addSpacing(20)
        
        # Text input
        input_label = QLabel("Or type text:")
        input_label.setStyleSheet("color: #888888;")
        control_layout.addWidget(input_label)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter text and press Enter")
        self.text_input.returnPressed.connect(self.on_text_enter)
        control_layout.addWidget(self.text_input)
        
        control_layout.addSpacing(20)
        
        # Buttons
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.toggle_running)
        control_layout.addWidget(self.start_btn)
        
        self.speak_btn = QPushButton("🎤  Speak")
        self.speak_btn.setObjectName("speakBtn")
        self.speak_btn.clicked.connect(self.listen_speech)
        self.speak_btn.hide()
        control_layout.addWidget(self.speak_btn)
        
        control_layout.addSpacing(20)
        
        # History
        history_title = QLabel("History")
        history_title.setFont(QFont("Arial", 12, QFont.Bold))
        control_layout.addWidget(history_title)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(150)
        control_layout.addWidget(self.history_text)
        
        control_layout.addStretch()
        content.addWidget(control_frame)
        
        main_layout.addLayout(content)
    
    def load_model(self):
        if not TORCH_AVAILABLE or not os.path.exists(MODEL_PATH):
            self.status_label.setText("Model not found")
            self.status_label.setStyleSheet("color: #ff6b6b;")
            return
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            self.max_frames = checkpoint.get('max_frames', 64)
            self.model = ASLModelV3(checkpoint['input_size'], checkpoint['num_classes'], max_frames=self.max_frames).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            with open(ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.status_label.setText(f"Model loaded ({checkpoint['num_classes']} signs)")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:40]}")
            self.status_label.setStyleSheet("color: #ff6b6b;")
    
    def switch_mode(self, mode):
        self.stop_all()
        self.current_mode = mode
        
        if mode == "sign_to_voice":
            self.btn_sign_to_voice.setObjectName("activeMode")
            self.btn_voice_to_sign.setObjectName("")
            self.speak_btn.hide()
        else:
            self.btn_sign_to_voice.setObjectName("")
            self.btn_voice_to_sign.setObjectName("activeMode")
            self.speak_btn.show()
        
        self.btn_sign_to_voice.setStyleSheet(STYLESHEET)
        self.btn_voice_to_sign.setStyleSheet(STYLESHEET)
        self.detection_label.setText("---")
        self.display_label.setText("Press Start to begin")
    
    def toggle_running(self):
        if self.is_running:
            self.stop_all()
        else:
            self.is_running = True
            self.start_btn.setText("⬛  Stop")
            self.start_btn.setObjectName("stopBtn")
            self.start_btn.setStyleSheet(STYLESHEET)
            
            if self.current_mode == "sign_to_voice":
                self.start_camera()
            else:
                self.avatar_timer.start(33)
    
    def stop_all(self):
        self.is_running = False
        self.camera_timer.stop()
        self.avatar_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.setText("▶  Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setStyleSheet(STYLESHEET)
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.frame_buffer.clear()
            self.camera_timer.start(30)
        else:
            self.status_label.setText("Cannot open camera")
    
    def update_camera(self):
        if not self.cap or not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.holistic:
                results = self.holistic.process(frame_rgb)
                landmarks = self.extract_landmarks(results)
                self.frame_buffer.append(landmarks)
                frame = self.draw_landmarks(frame, results)
                
                if len(self.frame_buffer) >= self.max_frames:
                    self.predict_sign()
                    self.frame_buffer.clear()
            
            self.signals.update_frame.emit(frame)
    
    def extract_landmarks(self, results):
        landmarks = []
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 63)
        
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)
        
        return np.array(landmarks, dtype=np.float32)
    
    def draw_landmarks(self, frame, results):
        if MEDIAPIPE_AVAILABLE:
            mp_draw = mp.solutions.drawing_utils
            mp_holistic = mp.solutions.holistic
            for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand:
                    mp_draw.draw_landmarks(frame, hand, mp_holistic.HAND_CONNECTIONS)
        return frame
    
    def predict_sign(self):
        if not self.model:
            return
        
        sequence = np.array(list(self.frame_buffer), dtype=np.float32)
        if len(sequence) < self.max_frames:
            sequence = np.vstack([sequence, np.zeros((self.max_frames - len(sequence), 225))])
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(input_tensor), dim=1)[0]
            conf, idx = torch.max(probs, 0)
            
            if conf.item() > 0.4:
                word = self.label_encoder.inverse_transform([idx.item()])[0]
                self.signals.update_detection.emit(word, conf.item())
                self.add_history(f"[Sign→Voice] {word}")
                self.tts.speak(word)
    
    def update_avatar(self):
        if not self.is_running:
            return
        
        if self.is_signing and self.current_sign is not None:
            if self.sign_frame_idx < len(self.current_sign):
                self.avatar_landmarks = 0.3 * self.current_sign[self.sign_frame_idx] + 0.7 * self.avatar_landmarks
                self.sign_frame_idx += 1
            else:
                self.is_signing = False
                if self.sign_queue:
                    self.play_next_sign()
                else:
                    self.detection_label.setText("---")
        else:
            self.avatar_landmarks = 0.95 * self.avatar_landmarks + 0.05 * self.create_neutral_pose()
        
        self.draw_avatar()
    
    def create_neutral_pose(self):
        lm = np.zeros(225, dtype=np.float32)
        ps = 126
        for idx, (x, y) in [(11, (0.7, 0.4)), (12, (0.3, 0.4)), (13, (0.8, 0.6)), 
                             (14, (0.2, 0.6)), (15, (0.85, 0.8)), (16, (0.15, 0.8))]:
            lm[ps + idx * 3], lm[ps + idx * 3 + 1] = x, y
        return lm
    
    def draw_avatar(self):
        w, h = 680, 480
        img = np.full((h, w, 3), 26, dtype=np.uint8)
        
        cx, cy, scale = w // 2, h // 2 + 30, min(w, h) * 0.45
        pose = self.avatar_landmarks[126:225].reshape(33, 3)
        right_hand = self.avatar_landmarks[63:126].reshape(21, 3)
        
        # Body
        body_color = (150, 130, 120)
        shirt_color = (180, 100, 80)
        
        shoulder_y = int(cy - scale * 0.15)
        shoulder_w = int(scale * 0.22)
        pts = np.array([[cx-shoulder_w, shoulder_y], [cx+shoulder_w, shoulder_y],
                        [cx+shoulder_w+20, int(cy+scale*0.35)], [cx-shoulder_w-20, int(cy+scale*0.35)]], np.int32)
        cv2.fillPoly(img, [pts], shirt_color)
        cv2.rectangle(img, (cx-15, shoulder_y-25), (cx+15, shoulder_y), body_color, -1)
        
        # Head
        head_y = shoulder_y - int(scale * 0.18)
        head_r = int(scale * 0.12)
        cv2.circle(img, (cx, head_y), head_r, body_color, -1)
        cv2.ellipse(img, (cx, head_y - head_r//3), (head_r, head_r//2), 0, 180, 360, (40, 30, 25), -1)
        cv2.circle(img, (cx - head_r//3, head_y), 6, (255, 255, 255), -1)
        cv2.circle(img, (cx + head_r//3, head_y), 6, (255, 255, 255), -1)
        cv2.circle(img, (cx - head_r//3, head_y), 3, (40, 30, 20), -1)
        cv2.circle(img, (cx + head_r//3, head_y), 3, (40, 30, 20), -1)
        cv2.ellipse(img, (cx, head_y + head_r//2), (8, 4), 0, 0, 180, (100, 80, 80), 2)
        
        # Arms
        for side, s_idx, w_idx in [(1, 11, 15), (-1, 12, 16)]:
            sx = cx + side * shoulder_w
            wx = int(cx + (pose[w_idx][0] - 0.5) * scale * 1.5)
            wy = int(cy + (pose[w_idx][1] - 0.5) * scale)
            cv2.line(img, (sx, shoulder_y), (wx, wy), body_color, 16)
            cv2.circle(img, (wx, wy), 18, body_color, -1)
        
        # Right hand fingers
        if np.abs(right_hand).sum() > 0.5:
            fingers = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
            pts = [(int(cx + (right_hand[i][0] - 0.5) * scale * 1.5), 
                    int(cy + (right_hand[i][1] - 0.5) * scale)) for i in range(21)]
            for finger in fingers:
                for i in range(len(finger) - 1):
                    cv2.line(img, pts[finger[i]], pts[finger[i+1]], body_color, max(4, 12-i*2))
                for f in finger:
                    cv2.circle(img, pts[f], 5, body_color, -1)
        
        # Convert and display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qimg))
    
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = img.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(QPixmap.fromImage(scaled))
    
    def set_status(self, text):
        self.status_label.setText(text)
    
    def set_detection(self, word, conf):
        self.detection_label.setText(word.upper())
        self.confidence_label.setText(f"Confidence: {conf*100:.1f}%")
    
    def listen_speech(self):
        self.status_label.setText("Listening...")
        def _listen():
            text, err = self.speech.listen()
            if text:
                self.signals.speech_result.emit(text)
            else:
                self.signals.update_status.emit(err or "Try again")
        threading.Thread(target=_listen, daemon=True).start()
    
    def process_speech(self, text):
        self.status_label.setText(f"Heard: {text}")
        self.add_history(f"[Voice→Sign] {text}")
        
        for word in text.split():
            word = ''.join(c for c in word if c.isalpha())
            if self.sign_db.get_sign(word) is not None:
                self.sign_queue.append(word)
            else:
                similar = self.sign_db.find_similar(word)
                if similar:
                    self.sign_queue.append(similar[0])
        
        if self.sign_queue and not self.is_signing:
            self.play_next_sign()
    
    def play_next_sign(self):
        if self.sign_queue:
            word = self.sign_queue.popleft()
            self.current_sign = self.sign_db.get_sign(word)
            self.sign_frame_idx = 0
            self.is_signing = True
            self.detection_label.setText(word.upper())
    
    def on_text_enter(self):
        text = self.text_input.text().strip()
        if text:
            self.text_input.clear()
            self.process_speech(text)
    
    def add_history(self, text):
        self.history_text.append(text)
    
    def closeEvent(self, event):
        self.stop_all()
        if self.holistic:
            self.holistic.close()
        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
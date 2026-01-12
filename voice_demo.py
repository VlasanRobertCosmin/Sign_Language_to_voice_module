"""
Real-Time ASL Signs Recognition with Voice Output
===================================================
Speaks the detected sign using text-to-speech.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
from collections import deque
import math
import threading

# ============================================================
# TEXT-TO-SPEECH SETUP
# ============================================================

# Try different TTS engines
TTS_ENGINE = None

def init_tts():
    """Initialize text-to-speech engine."""
    global TTS_ENGINE
    
    # Option 1: pyttsx3 (offline, works everywhere)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed
        engine.setProperty('volume', 0.9)  # Volume
        TTS_ENGINE = ('pyttsx3', engine)
        print("TTS: Using pyttsx3 (offline)")
        return True
    except:
        pass
    
    # Option 2: gTTS + pygame (online, better quality)
    try:
        from gtts import gTTS
        import pygame
        pygame.mixer.init()
        TTS_ENGINE = ('gtts', None)
        print("TTS: Using gTTS (online)")
        return True
    except:
        pass
    
    # Option 3: espeak (Linux)
    try:
        import subprocess
        result = subprocess.run(['espeak', '--version'], capture_output=True)
        if result.returncode == 0:
            TTS_ENGINE = ('espeak', None)
            print("TTS: Using espeak (Linux)")
            return True
    except:
        pass
    
    print("WARNING: No TTS engine found!")
    print("Install one with: pip install pyttsx3")
    print("Or: pip install gtts pygame")
    return False


def speak(text):
    """Speak text in background thread."""
    if TTS_ENGINE is None:
        print(f"[SPEAK]: {text}")
        return
    
    def _speak():
        try:
            if TTS_ENGINE[0] == 'pyttsx3':
                engine = TTS_ENGINE[1]
                engine.say(text)
                engine.runAndWait()
            
            elif TTS_ENGINE[0] == 'gtts':
                from gtts import gTTS
                import pygame
                import io
                
                tts = gTTS(text=text, lang='en')
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                
                pygame.mixer.music.load(fp, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pass
            
            elif TTS_ENGINE[0] == 'espeak':
                import subprocess
                subprocess.run(['espeak', text], capture_output=True)
        
        except Exception as e:
            print(f"TTS Error: {e}")
    
    # Run in background thread so it doesn't block video
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "asl_signs_model.pth"
ENCODER_PATH = "asl_signs_encoder.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LANDMARKS = ['left_hand', 'right_hand', 'pose']


# ============================================================
# MODELS
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ASLHybridModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=256, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x_trans = self.pos_encoder(x)
        x_trans = self.transformer(x_trans)
        x_lstm, _ = self.lstm(x)
        combined = torch.cat([x_trans, x_lstm], dim=-1)
        attn_weights = torch.softmax(self.attention(combined), dim=1)
        pooled = torch.sum(combined * attn_weights, dim=1)
        return self.classifier(pooled)


class ASLSignsModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context)


# ============================================================
# LANDMARK EXTRACTOR
# ============================================================

class LandmarkExtractor:
    def __init__(self):
        import mediapipe as mp
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract(self, frame_rgb):
        results = self.holistic.process(frame_rgb)
        
        landmarks = []
        
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)
        
        return np.array(landmarks, dtype=np.float32), results
    
    def draw(self, frame, results):
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        return frame
    
    def close(self):
        self.holistic.close()


# ============================================================
# DEMO WITH VOICE
# ============================================================

def load_model():
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    input_size = checkpoint['input_size']
    num_classes = checkpoint['num_classes']
    max_frames = checkpoint.get('max_frames', 60)
    classes = checkpoint.get('classes', [])
    
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    
    if any('transformer' in k for k in state_dict_keys):
        print("Model: Hybrid Transformer + LSTM")
        model = ASLHybridModel(input_size, num_classes).to(DEVICE)
    else:
        print("Model: LSTM + Attention")
        model = ASLSignsModel(input_size, num_classes).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Classes: {num_classes} | Frames: {max_frames}")
    
    return model, label_encoder, max_frames, classes


def run_demo():
    # Initialize TTS
    init_tts()
    
    # Load model
    model, label_encoder, max_frames, classes = load_model()
    extractor = LandmarkExtractor()
    
    frame_buffer = deque(maxlen=max_frames)
    
    print("\n" + "=" * 50)
    print("ASL RECOGNITION WITH VOICE OUTPUT")
    print("=" * 50)
    print("Controls:")
    print("  r = Start recording")
    print("  c = Clear buffer")
    print("  p = Predict now")
    print("  v = Toggle voice on/off")
    print("  q = Quit")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    recording = False
    voice_enabled = True
    last_prediction = ""
    last_confidence = 0.0
    top_3 = []
    last_spoken = ""  # Avoid repeating same word
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks
        landmarks, results = extractor.extract(frame_rgb)
        
        # Draw landmarks
        frame = extractor.draw(frame, results)
        
        # Add to buffer if recording
        if recording:
            frame_buffer.append(landmarks)
        
        # Auto-predict when buffer is full
        if len(frame_buffer) >= max_frames:
            sequence = np.array(list(frame_buffer), dtype=np.float32)
            sequence = np.nan_to_num(sequence, nan=0.0)
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                
                top_probs, top_indices = torch.topk(probs, min(3, len(classes)))
                top_3 = [(label_encoder.inverse_transform([idx.item()])[0], prob.item()) 
                         for idx, prob in zip(top_indices, top_probs)]
                
                last_prediction = top_3[0][0]
                last_confidence = top_3[0][1]
            
            # Speak the prediction if confident enough
            if voice_enabled and last_confidence > 0.4 and last_prediction != last_spoken:
                speak(last_prediction)
                last_spoken = last_prediction
                print(f"🔊 {last_prediction} ({last_confidence*100:.1f}%)")
            
            frame_buffer.clear()
            recording = False
        
        # Draw UI
        h, w = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Status line
        status = "● RECORDING" if recording else "○ PAUSED"
        color = (0, 255, 0) if recording else (128, 128, 128)
        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Voice status
        voice_status = "🔊 ON" if voice_enabled else "🔇 OFF"
        voice_color = (0, 255, 0) if voice_enabled else (0, 0, 255)
        cv2.putText(frame, voice_status, (320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        # Progress bar
        progress = len(frame_buffer) / max_frames
        cv2.rectangle(frame, (20, 45), (380, 65), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 45), (20 + int(360 * progress), 65), (0, 200, 0), -1)
        cv2.putText(frame, f"{len(frame_buffer)}/{max_frames}", (170, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Top 3 predictions
        y = 95
        for i, (sign, conf) in enumerate(top_3[:3]):
            if i == 0:
                color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
                cv2.putText(frame, f"1. {sign}: {conf*100:.1f}%", (20, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(frame, f"{i+1}. {sign}: {conf*100:.1f}%", (20, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y += 35
        
        # Instructions
        cv2.putText(frame, "r=rec | c=clear | p=predict | v=voice | q=quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        cv2.imshow('ASL Recognition + Voice', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = True
            frame_buffer.clear()
            top_3 = []
            last_spoken = ""
            print("Recording...")
        elif key == ord('c'):
            frame_buffer.clear()
            recording = False
            top_3 = []
            print("Cleared")
        elif key == ord('v'):
            voice_enabled = not voice_enabled
            print(f"Voice: {'ON' if voice_enabled else 'OFF'}")
        elif key == ord('p') and len(frame_buffer) >= 10:
            # Manual predict
            sequence = np.array(list(frame_buffer), dtype=np.float32)
            if len(sequence) < max_frames:
                padding = np.zeros((max_frames - len(sequence), sequence.shape[1]), dtype=np.float32)
                sequence = np.vstack([sequence, padding])
            
            sequence = np.nan_to_num(sequence, nan=0.0)
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                top_probs, top_indices = torch.topk(probs, min(3, len(classes)))
                top_3 = [(label_encoder.inverse_transform([idx.item()])[0], prob.item()) 
                         for idx, prob in zip(top_indices, top_probs)]
            
            if voice_enabled and top_3[0][1] > 0.3:
                speak(top_3[0][0])
                last_spoken = top_3[0][0]
            
            print(f"Prediction: {top_3[0][0]} ({top_3[0][1]*100:.1f}%)")
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("\nDemo ended.")


if __name__ == "__main__":
    run_demo()
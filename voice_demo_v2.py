"""
Real-Time ASL Signs Recognition with Voice Output
===================================================
Supports all model versions (v1, v2, v3)
Auto-detects model architecture from checkpoint.
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

TTS_ENGINE = None

def init_tts():
    global TTS_ENGINE
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        TTS_ENGINE = ('pyttsx3', engine)
        print("TTS: pyttsx3 (offline)")
        return True
    except:
        pass
    
    try:
        from gtts import gTTS
        import pygame
        pygame.mixer.init()
        TTS_ENGINE = ('gtts', None)
        print("TTS: gTTS (online)")
        return True
    except:
        pass
    
    try:
        import subprocess
        result = subprocess.run(['espeak', '--version'], capture_output=True)
        if result.returncode == 0:
            TTS_ENGINE = ('espeak', None)
            print("TTS: espeak (Linux)")
            return True
    except:
        pass
    
    print("WARNING: No TTS engine found! Install: pip install pyttsx3")
    return False


def speak(text):
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
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()


# ============================================================
# CONFIGURATION - CHANGE THIS TO SWITCH MODELS
# ============================================================

# Choose your model version:
MODEL_PATH = "asl_signs_model_v3.pth"      # V3 model
ENCODER_PATH = "asl_signs_encoder_v3.pkl"  # V3 encoder

# Or use older versions:
# MODEL_PATH = "asl_signs_model_v2.pth"
# ENCODER_PATH = "asl_signs_encoder_v2.pkl"

# MODEL_PATH = "asl_signs_model.pth"
# ENCODER_PATH = "asl_signs_encoder.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LANDMARKS = ['left_hand', 'right_hand', 'pose']


# ============================================================
# ALL MODEL ARCHITECTURES
# ============================================================

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
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.norm(x + residual)


# V3 Model
class ASLModelV3(nn.Module):
    def __init__(self, input_size, num_classes, d_model=384, n_heads=8, 
                 n_layers=6, dim_ff=1536, dropout=0.4, max_frames=64):
        super().__init__()
        
        self.d_model = d_model
        
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj1 = nn.Linear(input_size, d_model)
        self.input_proj2 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),
        )
        self.input_ln = nn.LayerNorm(d_model)
        
        self.conv_subsample = ConvSubsampling(d_model, dropout * 0.5)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames, dropout=dropout * 0.5)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.trans_ln = nn.LayerNorm(d_model)
        
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm_ln = nn.LayerNorm(d_model)
        
        self.n_queries = 4
        self.pool_queries = nn.Parameter(torch.randn(1, self.n_queries, d_model * 2))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=d_model * 2, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.pool_ln = nn.LayerNorm(d_model * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2 * self.n_queries, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(384, num_classes)
        )
    
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


# V2 / Hybrid Model
class ASLHybridModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=256, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x_trans = self.transformer(self.pos_encoder(x))
        x_lstm, _ = self.lstm(x)
        combined = torch.cat([x_trans, x_lstm], dim=-1)
        attn = torch.softmax(self.attention(combined), dim=1)
        pooled = torch.sum(combined * attn, dim=1)
        return self.classifier(pooled)


# V1 / Simple LSTM Model
class ASLSignsModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, num_layers=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.ReLU(), nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn, dim=1)
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
# LOAD MODEL - AUTO DETECT VERSION
# ============================================================

def load_model():
    print(f"Loading {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    input_size = checkpoint['input_size']
    num_classes = checkpoint['num_classes']
    max_frames = checkpoint.get('max_frames', 64)
    classes = checkpoint.get('classes', [])
    version = checkpoint.get('version', 'v1')
    
    print(f"Model version: {version}")
    print(f"Classes: {num_classes}")
    print(f"Max frames: {max_frames}")
    
    # Auto-detect model architecture
    if version == 'v3' or checkpoint.get('d_model') == 384:
        print("Architecture: V3 (Large Hybrid)")
        d_model = checkpoint.get('d_model', 384)
        n_layers = checkpoint.get('n_layers', 6)
        model = ASLModelV3(
            input_size, num_classes, 
            d_model=d_model, n_layers=n_layers,
            max_frames=max_frames
        ).to(DEVICE)
    elif 'pool_queries' in str(checkpoint['model_state_dict'].keys()):
        # Check for V3 specific layers
        print("Architecture: V3 (detected from state dict)")
        model = ASLModelV3(input_size, num_classes, max_frames=max_frames).to(DEVICE)
    elif any('transformer' in k for k in checkpoint['model_state_dict'].keys()):
        print("Architecture: V2 (Hybrid)")
        model = ASLHybridModel(input_size, num_classes).to(DEVICE)
    else:
        print("Architecture: V1 (LSTM)")
        model = ASLSignsModel(input_size, num_classes).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    
    accuracy = checkpoint.get('accuracy', 0)
    if accuracy:
        print(f"Trained accuracy: {accuracy*100:.1f}%")
    
    return model, encoder, max_frames, classes


# ============================================================
# DEMO
# ============================================================

def run_demo():
    init_tts()
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
    top_3 = []
    last_spoken = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        landmarks, results = extractor.extract(frame_rgb)
        frame = extractor.draw(frame, results)
        
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
            
            if voice_enabled and top_3[0][1] > 0.4 and top_3[0][0] != last_spoken:
                speak(top_3[0][0])
                last_spoken = top_3[0][0]
                print(f"🔊 {top_3[0][0]} ({top_3[0][1]*100:.1f}%)")
            
            frame_buffer.clear()
            recording = False
        
        # UI
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        status = "● RECORDING" if recording else "○ PAUSED"
        color = (0, 255, 0) if recording else (128, 128, 128)
        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        voice_status = "🔊 ON" if voice_enabled else "🔇 OFF"
        voice_color = (0, 255, 0) if voice_enabled else (0, 0, 255)
        cv2.putText(frame, voice_status, (320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        progress = len(frame_buffer) / max_frames
        cv2.rectangle(frame, (20, 45), (380, 65), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 45), (20 + int(360 * progress), 65), (0, 200, 0), -1)
        cv2.putText(frame, f"{len(frame_buffer)}/{max_frames}", (170, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
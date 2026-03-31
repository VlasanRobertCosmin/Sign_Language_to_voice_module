"""
Sign Language Translator Application (OpenCV UI)
=================================================
Modules:
  1. Sign -> Voice  — camera + MediaPipe -> ASL model -> speech
  2. Voice -> Sign  — mic / text input   -> avatar animation

Model support: V1 (LSTM), V2 (Hybrid), V3 (Large Hybrid) — auto-detected.
TTS fallback:  pyttsx3 -> gTTS -> espeak -> print.

Requirements:
    pip install numpy opencv-python torch mediapipe Pillow
    pip install SpeechRecognition pyaudio pyttsx3
"""

import os
import threading
import time
import numpy as np
import cv2
from collections import deque
import math
import pickle

# -- Suppress ALSA noise on Linux --------------------------------------------
try:
    from ctypes import *
    _EHF = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def _noop(*a): pass
    try:
        cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_EHF(_noop))
    except Exception:
        pass
except Exception:
    pass

# -- Optional deps -----------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available -- Sign->Voice disabled")

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("SpeechRecognition not available -- mic input disabled")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available -- Sign->Voice disabled")


# ============================================================
# CONFIGURATION  (edit paths here)
# ============================================================

MODEL_PATH   = "asl_signs_model_v3.pth"
ENCODER_PATH = "asl_signs_encoder_v3.pkl"
CACHE_FILE   = "asl_signs_cache.npz"

# UI layout
WIN_W   = 1280
WIN_H   = 720
VIDEO_W = 820
PANEL_W = WIN_W - VIDEO_W   # 460

# Palette (BGR)
BG_COLOR      = (43,  43,  43)
PANEL_COLOR   = (54,  54,  54)
ACCENT_COLOR  = (255, 158,  74)
SUCCESS_COLOR = (127, 255,  74)
ERROR_COLOR   = (107, 107, 255)
TEXT_COLOR    = (240, 240, 240)
DIM_COLOR     = (140, 140, 140)
DARK_COLOR    = (30,  30,  30)
HIGHLIGHT     = (255, 200,  80)
REC_COLOR     = (60,  60,  220)


# ============================================================
# TEXT-TO-SPEECH  (pyttsx3 -> gTTS -> espeak -> print)
# ============================================================

class TextToSpeech:
    def __init__(self):
        self._kind   = None
        self._engine = None
        self._init()

    def _init(self):
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty('rate', 150)
            e.setProperty('volume', 0.9)
            self._kind, self._engine = 'pyttsx3', e
            print("TTS: pyttsx3 (offline)"); return
        except Exception:
            pass
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            self._kind = 'gtts'
            print("TTS: gTTS (online)"); return
        except Exception:
            pass
        try:
            import subprocess
            if subprocess.run(['espeak','--version'], capture_output=True).returncode == 0:
                self._kind = 'espeak'
                print("TTS: espeak (Linux)"); return
        except Exception:
            pass
        print("WARNING: No TTS engine found -- pip install pyttsx3")

    def speak(self, text):
        def _run():
            try:
                if self._kind == 'pyttsx3':
                    self._engine.say(text); self._engine.runAndWait()
                elif self._kind == 'gtts':
                    from gtts import gTTS; import pygame, io
                    fp = io.BytesIO(); gTTS(text=text, lang='en').write_to_fp(fp); fp.seek(0)
                    pygame.mixer.music.load(fp, 'mp3'); pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy(): pass
                elif self._kind == 'espeak':
                    import subprocess; subprocess.run(['espeak', text], capture_output=True)
                else:
                    print(f"[TTS] {text}")
            except Exception as exc:
                print(f"TTS error: {exc}")
        threading.Thread(target=_run, daemon=True).start()


# ============================================================
# SPEECH RECOGNISER
# ============================================================

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = self.microphone = None
        if not SPEECH_AVAILABLE:
            return
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=0.5)
        except Exception:
            self.recognizer = None

    def listen(self, timeout=5):
        if not self.recognizer:
            return None, "Microphone not available"
        try:
            with self.microphone as src:
                audio = self.recognizer.listen(src, timeout=timeout, phrase_time_limit=5)
            return self.recognizer.recognize_google(audio).lower(), None
        except sr.WaitTimeoutError:
            return None, "No speech detected"
        except sr.UnknownValueError:
            return None, "Could not understand"
        except Exception as e:
            return None, str(e)


# ============================================================
# MODEL ARCHITECTURES  (V1 / V2 / V3)
# ============================================================

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))
        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])

    class ConvSubsampling(nn.Module):
        def __init__(self, d_model, dropout=0.1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(d_model)
        def forward(self, x):
            r = x
            return self.norm(self.conv(x.transpose(1,2)).transpose(1,2) + r)

    # V3
    class ASLModelV3(nn.Module):
        def __init__(self, input_size, num_classes, d_model=384, n_heads=8,
                     n_layers=6, dim_ff=1536, dropout=0.4, max_frames=64):
            super().__init__()
            self.input_norm  = nn.LayerNorm(input_size)
            self.input_proj1 = nn.Linear(input_size, d_model)
            self.input_proj2 = nn.Sequential(nn.GELU(), nn.Dropout(dropout*.5), nn.Linear(d_model, d_model))
            self.input_ln    = nn.LayerNorm(d_model)
            self.conv_sub    = ConvSubsampling(d_model, dropout*.5)
            self.pos_enc     = PositionalEncoding(d_model, max_len=max_frames, dropout=dropout*.5)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout,
                                           activation='gelu', batch_first=True, norm_first=True),
                num_layers=n_layers)
            self.trans_ln    = nn.LayerNorm(d_model)
            self.lstm        = nn.LSTM(d_model, d_model//2, 2, batch_first=True, dropout=dropout, bidirectional=True)
            self.lstm_ln     = nn.LayerNorm(d_model)
            self.n_q         = 4
            self.pool_q      = nn.Parameter(torch.randn(1, self.n_q, d_model*2))
            self.pool_attn   = nn.MultiheadAttention(d_model*2, n_heads, dropout=dropout, batch_first=True)
            self.pool_ln     = nn.LayerNorm(d_model*2)
            self.classifier  = nn.Sequential(
                nn.Linear(d_model*2*self.n_q, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout*.7),
                nn.Linear(384, num_classes))

        def forward(self, x):
            B = x.size(0)
            xn = self.input_norm(x)
            x  = self.input_ln(self.input_proj2(self.input_proj1(xn)) + self.input_proj1(xn))
            x  = self.conv_sub(x)
            xt = self.trans_ln(self.transformer(self.pos_enc(x)))
            xl, _ = self.lstm(x); xl = self.lstm_ln(xl)
            comb  = torch.cat([xt, xl], -1)
            p, _  = self.pool_attn(self.pool_q.expand(B,-1,-1), comb, comb)
            return self.classifier(self.pool_ln(p).view(B, -1))

    # V2
    class ASLHybridModel(nn.Module):
        def __init__(self, input_size, num_classes, d_model=256, dropout=0.3):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(input_size, d_model), nn.LayerNorm(d_model),
                                      nn.GELU(), nn.Dropout(dropout))
            self.pos  = PositionalEncoding(d_model)
            self.tr   = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 8, 512, dropout, activation='gelu', batch_first=True),
                num_layers=3)
            self.lstm = nn.LSTM(d_model, d_model//2, 2, batch_first=True, dropout=dropout, bidirectional=True)
            self.attn = nn.Sequential(nn.Linear(d_model*2, d_model), nn.Tanh(), nn.Linear(d_model, 1))
            self.cls  = nn.Sequential(
                nn.Linear(d_model*2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(512, 256),       nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes))
        def forward(self, x):
            x = self.proj(x); xt = self.tr(self.pos(x)); xl, _ = self.lstm(x)
            comb = torch.cat([xt, xl], -1)
            return self.cls((comb * torch.softmax(self.attn(comb), 1)).sum(1))

    # V1
    class ASLSignsModel(nn.Module):
        def __init__(self, input_size, num_classes, hidden=256, layers=2):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(input_size, hidden), nn.LayerNorm(hidden),
                                      nn.ReLU(), nn.Dropout(0.2))
            self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=0.3, bidirectional=True)
            self.attn = nn.Sequential(nn.Linear(hidden*2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.cls  = nn.Sequential(
                nn.Linear(hidden*2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 256),      nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes))
        def forward(self, x):
            x = self.proj(x); out, _ = self.lstm(x)
            return self.cls((out * torch.softmax(self.attn(out), 1)).sum(1))

    def load_model(device):
        print(f"Loading {MODEL_PATH} ...")
        ck   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        inp  = ck['input_size']
        ncls = ck['num_classes']
        mf   = ck.get('max_frames', 64)
        ver  = ck.get('version', 'v1')
        keys = str(ck['model_state_dict'].keys())

        if ver == 'v3' or ck.get('d_model') == 384 or 'pool_q' in keys or 'pool_queries' in keys:
            print(f"Architecture: V3  ({ncls} classes, {mf} frames)")
            mdl = ASLModelV3(inp, ncls, d_model=ck.get('d_model', 384),
                             n_layers=ck.get('n_layers', 6), max_frames=mf).to(device)
        elif 'transformer' in keys:
            print(f"Architecture: V2  ({ncls} classes)")
            mdl = ASLHybridModel(inp, ncls).to(device)
        else:
            print(f"Architecture: V1  ({ncls} classes)")
            mdl = ASLSignsModel(inp, ncls).to(device)

        mdl.load_state_dict(ck['model_state_dict'])
        mdl.eval()
        with open(ENCODER_PATH, 'rb') as f:
            enc = pickle.load(f)
        acc = ck.get('accuracy', 0)
        if acc:
            print(f"Trained accuracy: {acc*100:.1f}%")
        return mdl, enc, mf


# ============================================================
# SIGN DATABASE  (for Voice -> Sign avatar)
# ============================================================

class SignDatabase:
    def __init__(self, cache_file):
        self.signs, self.classes = {}, []
        if not os.path.exists(cache_file):
            return
        cache = np.load(cache_file, allow_pickle=True)
        X, y  = cache['X'], cache['y']
        self.classes = sorted(set(y))
        for i, label in enumerate(y):
            if label not in self.signs:
                seq   = X[i]
                valid = [seq[f] for f in range(len(seq)) if np.abs(seq[f]).sum() > 0.1]
                if valid:
                    self.signs[label] = self._interp(np.array(valid))

    def _interp(self, frames, factor=3):
        if len(frames) < 2: return frames
        out = []
        for i in range(len(frames)-1):
            out.append(frames[i])
            for j in range(1, factor):
                t = j/factor
                out.append(frames[i]*(1-t) + frames[i+1]*t)
        out.append(frames[-1])
        return np.array(out)

    def get_sign(self, word):
        return self.signs.get(word.lower().strip())

    def find_similar(self, word):
        w = word.lower().strip()
        if w in self.signs: return [w]
        return [s for s in self.classes if w in s or s in w][:3]


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_rect(img, x, y, w, h, color, radius=6):
    cv2.rectangle(img, (x+radius, y),       (x+w-radius, y+h),       color, -1)
    cv2.rectangle(img, (x,        y+radius), (x+w,        y+h-radius), color, -1)
    for cx, cy in [(x+radius,y+radius),(x+w-radius,y+radius),
                   (x+radius,y+h-radius),(x+w-radius,y+h-radius)]:
        cv2.circle(img, (cx, cy), radius, color, -1)

def put(img, text, x, y, scale=0.52, color=TEXT_COLOR, thick=1, center=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    tx = x - tw//2 if center else x
    cv2.putText(img, text, (tx, y), font, scale, color, thick, cv2.LINE_AA)
    return tw

def draw_button(img, label, x, y, w, h, color, active=False):
    draw_rect(img, x, y, w, h, color)
    if active:
        cv2.rectangle(img, (x,y), (x+w,y+h), HIGHLIGHT, 2)
    put(img, label, x+w//2, y+h//2+6, scale=0.50, color=DARK_COLOR, center=True)

def draw_bar(img, x, y, w, h, ratio, fg, bg=(50,50,50)):
    draw_rect(img, x, y, w, h, bg, radius=3)
    if ratio > 0:
        draw_rect(img, x, y, max(6, int(w*ratio)), h, fg, radius=3)


# ============================================================
# AVATAR RENDERING
# ============================================================

def _neutral():
    lm = np.zeros(225, dtype=np.float32)
    ps = 126
    for idx, x, y in [(11,.7,.4),(12,.3,.4),(13,.8,.6),(14,.2,.6),(15,.85,.8),(16,.15,.8)]:
        lm[ps+idx*3]=x; lm[ps+idx*3+1]=y
    return lm

def render_avatar(landmarks, w, h):
    img  = np.full((h, w, 3), 28, dtype=np.uint8)
    cx   = w//2; cy = h//2+30; scale = min(w,h)*0.5
    lh   = landmarks[0:63].reshape(21,3)
    rh   = landmarks[63:126].reshape(21,3)
    pose = landmarks[126:225].reshape(33,3)
    skin = (120,130,150); shirt = (180,100,80)

    # body
    sw  = int(scale*.25); shy = cy - int(scale*.1)
    pts = np.array([[cx-sw,shy],[cx+sw,shy],
                    [cx+sw+20,cy+int(scale*.3)],[cx-sw-20,cy+int(scale*.3)]], np.int32)
    cv2.fillPoly(img, [pts], shirt)
    cv2.rectangle(img, (cx-15,shy-30), (cx+15,shy), skin, -1)
    for i, side in [(15, -1),(16, 1)]:
        p  = pose[i]
        wp = (int(cx+(p[0]-.5)*scale*1.5*side*(-1)), int(cy+(p[1]-.5)*scale))
        ox = -sw if i==15 else sw
        cv2.line(img, (cx+ox, shy), wp, skin, 18)
        cv2.circle(img, wp, 20, skin, -1)

    # hands
    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for hand in (lh, rh):
        if np.abs(hand).sum() < 0.5: continue
        pts2 = [(int(cx+(hand[i][0]-.5)*scale*1.5), int(cy+(hand[i][1]-.5)*scale)) for i in range(21)]
        for ch in chains:
            for a, b in zip(ch, ch[1:]):
                cv2.line(img, pts2[a], pts2[b], skin, max(4, 12-ch.index(b)*2))
            for f in ch: cv2.circle(img, pts2[f], 5, skin, -1)

    # head
    hcy = cy - int(scale*.4); hr = int(scale*.15)
    cv2.circle(img, (cx,hcy), hr, skin, -1)
    cv2.ellipse(img, (cx,hcy-hr//4), (hr,hr//2), 0, 180, 360, (40,30,25), -1)
    for ex in [cx-hr//3, cx+hr//3]:
        cv2.circle(img, (ex, hcy-hr//6), 8, (255,255,255), -1)
        cv2.circle(img, (ex, hcy-hr//6), 4, (40,30,20),    -1)
    cv2.ellipse(img, (cx, hcy+hr//3), (10,5), 0, 0, 180, (100,80,80), 2)
    return img


# ============================================================
# MAIN APPLICATION
# ============================================================

class SignLanguageApp:

    def __init__(self):
        self.tts     = TextToSpeech()
        self.speech  = SpeechRecognizer()
        self.sign_db = SignDatabase(CACHE_FILE)
        self.device  = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        if TORCH_AVAILABLE else None)

        # model
        self.model         = None
        self.label_encoder = None
        self.max_frames    = 64

        # mediapipe
        self.holistic    = None
        self.mp_holistic = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_holistic = mp.solutions.holistic
            self.holistic    = self.mp_holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # state
        self.mode         = "sign_to_voice"
        self.is_running   = False
        self.status_msg   = "Ready"
        self.status_color = SUCCESS_COLOR
        self.history      = []

        # Sign->Voice
        self.recording     = False
        self.voice_enabled = True
        self.frame_buffer  = deque(maxlen=self.max_frames)
        self.top3          = []
        self.last_spoken   = ""
        self._cam_frame    = None
        self._lock         = threading.Lock()

        # Voice->Sign
        self.avatar_lm    = _neutral()
        self.sign_queue   = deque()
        self.current_sign = None
        self.sign_frame_i = 0
        self.is_signing   = False
        self.listening    = False
        self.text_input   = ""
        self.text_focused = False

        self._load_model()

    # -- model ---------------------------------------------------------------

    def _load_model(self):
        if not TORCH_AVAILABLE:
            self._status("PyTorch not available", ERROR_COLOR); return
        if not os.path.exists(MODEL_PATH):
            self._status(f"Model not found: {MODEL_PATH}", ERROR_COLOR); return
        try:
            self.model, self.label_encoder, self.max_frames = load_model(self.device)
            self.frame_buffer = deque(maxlen=self.max_frames)
            self._status(f"Model ready  ({self.max_frames} frames)", SUCCESS_COLOR)
        except Exception as e:
            self._status(f"Load error: {str(e)[:40]}", ERROR_COLOR)

    # -- helpers -------------------------------------------------------------

    def _status(self, msg, color=SUCCESS_COLOR):
        self.status_msg, self.status_color = msg, color

    def _add_history(self, text):
        self.history.append(text)
        if len(self.history) > 20: self.history.pop(0)

    # -- camera thread -------------------------------------------------------

    def _camera_thread(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._status("Cannot open camera", ERROR_COLOR)
            self.is_running = False; return
        self.frame_buffer.clear()
        while self.is_running and self.mode == "sign_to_voice":
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.holistic:
                res = self.holistic.process(rgb)
                lm  = self._extract(res)
                if self.recording:
                    self.frame_buffer.append(lm)
                frame = self._draw_lm(frame, res)
                if self.recording and len(self.frame_buffer) >= self.max_frames:
                    self._predict()
                    self.frame_buffer.clear()
                    self.recording = False
            with self._lock:
                self._cam_frame = frame.copy()
            time.sleep(0.03)
        cap.release()

    def _extract(self, results):
        lm = []
        for attr, n in [('left_hand_landmarks',63),('right_hand_landmarks',63),('pose_landmarks',99)]:
            src = getattr(results, attr)
            if src:
                for p in src.landmark: lm.extend([p.x, p.y, p.z])
            else:
                lm.extend([0.0]*n)
        return np.array(lm, dtype=np.float32)

    def _draw_lm(self, frame, results):
        if not MEDIAPIPE_AVAILABLE: return frame
        draw = mp.solutions.drawing_utils
        for attr in ('left_hand_landmarks','right_hand_landmarks'):
            src = getattr(results, attr)
            if src: draw.draw_landmarks(frame, src, self.mp_holistic.HAND_CONNECTIONS)
        return frame

    def _run_predict(self, seq):
        seq = np.nan_to_num(seq)
        if len(seq) < self.max_frames:
            seq = np.vstack([seq, np.zeros((self.max_frames-len(seq), seq.shape[1]), np.float32)])
        t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), 1)[0]
            top_p, top_i = torch.topk(probs, min(3, probs.shape[0]))
            self.top3 = [(self.label_encoder.inverse_transform([i.item()])[0], p.item())
                         for i, p in zip(top_i, top_p)]

    def _predict(self):
        if not self.model: return
        self._run_predict(np.array(list(self.frame_buffer), dtype=np.float32))
        if self.top3 and self.top3[0][1] > 0.4:
            word = self.top3[0][0]
            self._add_history(f"[Sign->Voice] {word}  {self.top3[0][1]*100:.0f}%")
            if self.voice_enabled and word != self.last_spoken:
                self.tts.speak(word); self.last_spoken = word

    def _predict_now(self):
        if not self.model or len(self.frame_buffer) < 5: return
        self._run_predict(np.array(list(self.frame_buffer), dtype=np.float32))
        if self.top3 and self.voice_enabled and self.top3[0][1] > 0.3:
            self.tts.speak(self.top3[0][0]); self.last_spoken = self.top3[0][0]
            self._add_history(f"[Sign->Voice] {self.top3[0][0]}  {self.top3[0][1]*100:.0f}%")

    # -- avatar --------------------------------------------------------------

    def _step_avatar(self):
        if self.is_signing and self.current_sign is not None:
            if self.sign_frame_i < len(self.current_sign):
                tgt = self.current_sign[self.sign_frame_i]
                self.avatar_lm  = .3*tgt + .7*self.avatar_lm
                self.sign_frame_i += 1
            else:
                self.is_signing = False; self.sign_frame_i = 0
                if self.sign_queue: self._play_next()
        else:
            self.avatar_lm = .95*self.avatar_lm + .05*_neutral()

    def _play_next(self):
        if self.sign_queue:
            word = self.sign_queue.popleft()
            self.current_sign = self.sign_db.get_sign(word)
            self.sign_frame_i = 0; self.is_signing = True

    def _process_text(self, text):
        self._status(f"Heard: {text}", SUCCESS_COLOR)
        self._add_history(f"[Voice->Sign] {text}")
        for word in text.lower().split():
            word = ''.join(c for c in word if c.isalpha())
            if not word: continue
            if self.sign_db.get_sign(word):
                self.sign_queue.append(word)
            else:
                sim = self.sign_db.find_similar(word)
                if sim: self.sign_queue.append(sim[0])
        if self.sign_queue and not self.is_signing:
            self._play_next()

    def _listen_thread(self):
        self._status("Listening...", ACCENT_COLOR)
        text, err = self.speech.listen()
        self.listening = False
        if text: self._process_text(text)
        else:    self._status(err, ERROR_COLOR)

    # -- mode / running ------------------------------------------------------

    def _switch_mode(self, mode):
        self.is_running = False; self._cam_frame = None
        self.recording  = False; self.top3 = []
        self.mode       = mode
        self._status(f"Mode: {mode.replace('_',' ').title()}", SUCCESS_COLOR)

    def _toggle_running(self):
        if self.is_running:
            self.is_running = False; self.recording = False
            self._status("Stopped", DIM_COLOR)
        else:
            self.is_running = True
            self._status("Running...", SUCCESS_COLOR)
            if self.mode == "sign_to_voice":
                threading.Thread(target=self._camera_thread, daemon=True).start()

    # -- panel render --------------------------------------------------------

    def _render_panel(self, panel):
        h, w = panel.shape[:2]
        panel[:] = np.array(PANEL_COLOR, dtype=np.uint8)

        # title
        put(panel, "Sign Language Translator", w//2, 30, .68, TEXT_COLOR, 2, center=True)

        # mode buttons
        draw_button(panel, "Sign -> Voice", 8,   58, 215, 36,
                    ACCENT_COLOR if self.mode=="sign_to_voice" else (72,72,72),
                    active=(self.mode=="sign_to_voice"))
        draw_button(panel, "Voice -> Sign", 230, 58, 215, 36,
                    ACCENT_COLOR if self.mode=="voice_to_sign" else (72,72,72),
                    active=(self.mode=="voice_to_sign"))

        # start/stop  (top-right)
        s_col = ERROR_COLOR if self.is_running else SUCCESS_COLOR
        s_lbl = "[ STOP ]" if self.is_running else "[ START ]"
        draw_button(panel, s_lbl, w-145, 102, 138, 36, s_col)

        cv2.line(panel, (8,105), (w-8,105), (70,70,70), 1)

        # status
        put(panel, "STATUS", 8, 122, .38, DIM_COLOR)
        put(panel, self.status_msg[:52], 8, 140, .46, self.status_color)

        cv2.line(panel, (8,155), (w-8,155), (70,70,70), 1)

        # ---- Sign->Voice pane ----
        if self.mode == "sign_to_voice":

            # REC indicator + VOL toggle
            r_col = REC_COLOR if self.recording else (72,72,72)
            draw_rect(panel, 8, 162, 115, 28, r_col, radius=5)
            put(panel, "● REC" if self.recording else "○ READY", 66, 181, .5, TEXT_COLOR, center=True)

            v_col = SUCCESS_COLOR if self.voice_enabled else (72,72,72)
            draw_rect(panel, 132, 162, 105, 28, v_col, radius=5)
            put(panel, "VOL ON" if self.voice_enabled else "VOL OFF", 184, 181, .5, DARK_COLOR, center=True)

            # buffer progress
            ratio = len(self.frame_buffer) / max(self.max_frames, 1)
            put(panel, f"Buffer  {len(self.frame_buffer)}/{self.max_frames}", 8, 213, .4, DIM_COLOR)
            draw_bar(panel, 8, 218, w-16, 13, ratio, SUCCESS_COLOR)

            # top-3 predictions
            put(panel, "TOP PREDICTIONS", 8, 252, .4, DIM_COLOR)
            y0 = 258
            for i, (word, conf) in enumerate(self.top3[:3]):
                bar_y  = y0 + i*52
                c_col  = SUCCESS_COLOR if conf>.5 else (ACCENT_COLOR if conf>.3 else DIM_COLOR)
                sc     = .78 if i==0 else .54
                th     = 2   if i==0 else 1
                put(panel, f"{i+1}. {word.upper()}", 8, bar_y+20, sc, c_col, th)
                put(panel, f"{conf*100:.1f}%", w-60, bar_y+20, .5, c_col)
                draw_bar(panel, 8, bar_y+24, w-16, 10, conf, c_col)

            # keyboard hint
            put(panel, "S=start  R=record  C=clear  P=predict now", 8, h-42, .37, DIM_COLOR)
            put(panel, "V=voice toggle  1/2=mode  Q=quit",           8, h-24, .37, DIM_COLOR)

        # ---- Voice->Sign pane ----
        else:
            signing_word = ""
            if self.is_signing and self.sign_queue or self.is_signing:
                signing_word = "SIGNING..."
            put(panel, "AVATAR",    8,    180, .4, DIM_COLOR)
            put(panel, signing_word or "READY", w//2, 215, .85, HIGHLIGHT, 2, center=True)

            # text input
            cv2.line(panel, (8,235), (w-8,235), (70,70,70), 1)
            put(panel, "TYPE TEXT  (Enter to send):", 8, 252, .4, DIM_COLOR)
            tbx,tby,tbw,tbh = 8, 258, w-16, 34
            draw_rect(panel, tbx, tby, tbw, tbh, (62,72,82) if self.text_focused else (50,50,50), radius=5)
            if self.text_focused:
                cv2.rectangle(panel,(tbx,tby),(tbx+tbw,tby+tbh),ACCENT_COLOR,1)
            put(panel, self.text_input[-44:]+( "|" if self.text_focused else ""), tbx+8, tby+23, .5, TEXT_COLOR)

            # mic button
            m_col = (200,200,50) if self.listening else ACCENT_COLOR
            m_lbl = "LISTENING..." if self.listening else "MIC SPEAK"
            draw_button(panel, m_lbl, 8, 302, 210, 36, m_col)

            put(panel, "M=mic  click textbox to type  Q=quit", 8, h-24, .37, DIM_COLOR)

        # history (shared)
        hist_top = h - 170
        cv2.line(panel, (8,hist_top-14), (w-8,hist_top-14), (70,70,70), 1)
        put(panel, "HISTORY", 8, hist_top, .38, DIM_COLOR)
        for i, entry in enumerate(self.history[-8:][::-1]):
            col = ACCENT_COLOR if "Sign->" in entry else SUCCESS_COLOR
            put(panel, entry[:54], 8, hist_top+16+i*19, .38, col)

    # -- full frame ----------------------------------------------------------

    def _render(self):
        canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

        if self.mode == "sign_to_voice":
            with self._lock:
                cam = self._cam_frame.copy() if self._cam_frame is not None else None
            if cam is not None:
                canvas[:, :VIDEO_W] = cv2.resize(cam, (VIDEO_W, WIN_H))
            else:
                put(canvas, "Camera not started", VIDEO_W//2, WIN_H//2-20, .9, DIM_COLOR, center=True)
                put(canvas, "Press S to start, then R to record", VIDEO_W//2, WIN_H//2+22, .6, DIM_COLOR, center=True)
        else:
            self._step_avatar()
            canvas[:, :VIDEO_W] = render_avatar(self.avatar_lm, VIDEO_W, WIN_H)

        cv2.line(canvas, (VIDEO_W,0), (VIDEO_W,WIN_H), (70,70,70), 2)
        self._render_panel(canvas[:, VIDEO_W:])
        return canvas

    # -- mouse ---------------------------------------------------------------

    def _in(self, px, py, x, y, w, h):
        return x <= px <= x+w and y <= py <= y+h

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        px = x - VIDEO_W

        if self._in(px, y,   8, 58, 215, 36): self._switch_mode("sign_to_voice"); return
        if self._in(px, y, 230, 58, 215, 36): self._switch_mode("voice_to_sign"); return
        if self._in(px, y, PANEL_W-145, 102, 138, 36): self._toggle_running(); return

        if self.mode == "sign_to_voice":
            if self._in(px, y, 8, 162, 115, 28) and self.is_running:
                self.recording = not self.recording
                if self.recording: self.frame_buffer.clear(); self.top3 = []
            elif self._in(px, y, 132, 162, 105, 28):
                self.voice_enabled = not self.voice_enabled
        elif self.mode == "voice_to_sign":
            if self._in(px, y, 8, 258, PANEL_W-16, 34):
                self.text_focused = True
            elif self._in(px, y, 8, 302, 210, 36) and not self.listening:
                self.listening = True
                threading.Thread(target=self._listen_thread, daemon=True).start()
            else:
                self.text_focused = False

    # -- keyboard ------------------------------------------------------------

    def on_key(self, key):
        if key == -1: return False
        k = key & 0xFF

        if self.text_focused and self.mode == "voice_to_sign":
            if k == 13:
                if self.text_input.strip(): self._process_text(self.text_input.strip()); self.text_input = ""
            elif k == 8:  self.text_input = self.text_input[:-1]
            elif 32 <= k <= 126: self.text_input += chr(k)
            return False

        if   k == ord('q'): return True
        elif k == ord('s'): self._toggle_running()
        elif k == ord('1'): self._switch_mode("sign_to_voice")
        elif k == ord('2'): self._switch_mode("voice_to_sign")
        elif self.mode == "sign_to_voice" and self.is_running:
            if   k == ord('r'):
                self.recording = not self.recording
                if self.recording: self.frame_buffer.clear(); self.top3=[]; print("Recording...")
                else: print("Paused")
            elif k == ord('c'):
                self.frame_buffer.clear(); self.recording=False; self.top3=[]; print("Cleared")
            elif k == ord('p'): self._predict_now()
            elif k == ord('v'):
                self.voice_enabled = not self.voice_enabled
                print(f"Voice: {'ON' if self.voice_enabled else 'OFF'}")
        elif self.mode == "voice_to_sign":
            if k == ord('m') and not self.listening:
                self.listening = True
                threading.Thread(target=self._listen_thread, daemon=True).start()
        return False

    # -- main loop -----------------------------------------------------------

    def run(self):
        cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sign Language Translator", WIN_W, WIN_H)
        cv2.setMouseCallback("Sign Language Translator", self.on_mouse)

        print("\n" + "="*54)
        print("  Sign Language Translator  |  OpenCV UI")
        print("="*54)
        print("  1 / 2     switch mode  (Sign->Voice / Voice->Sign)")
        print("  S         start / stop camera")
        print("  R         toggle recording        [Sign->Voice]")
        print("  C         clear frame buffer      [Sign->Voice]")
        print("  P         force-predict now       [Sign->Voice]")
        print("  V         toggle TTS voice        [Sign->Voice]")
        print("  M         activate microphone     [Voice->Sign]")
        print("  Q         quit")
        print("="*54 + "\n")

        while True:
            cv2.imshow("Sign Language Translator", self._render())
            if self.on_key(cv2.waitKey(33)):
                break

        self.is_running = False
        if self.holistic: self.holistic.close()
        cv2.destroyAllWindows()


# ============================================================
if __name__ == "__main__":
    SignLanguageApp().run()
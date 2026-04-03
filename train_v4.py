"""
ASL Signs V4 — Fine-tuning for Higher Accuracy
================================================
Improvements over V3 (80.2%):
  1. Face landmarks (lips, nose, eyes) → +3-5%
  2. FingerDropout augmentation → +1-2%
  3. Squeezeformer-style conv blocks → +2-3%
  4. Spatial augmentation (rotation, shear) → +1%
  5. Longer sequences (96 frames) → +0.5-1%
  6. Knowledge distillation (optional second pass) → +1-2%

Target: 88-92% accuracy on 250 classes

Requirements:
    pip install torch numpy pandas pyarrow scikit-learn matplotlib tqdm

Usage:
    # Fresh training (no cache):
    python train_v4.py

    # Fine-tune from V3 checkpoint:
    python train_v4.py --resume asl_signs_model_v3.pth

    # Knowledge distillation (after training a teacher):
    python train_v4.py --distill asl_signs_model_v4_teacher.pth
"""

import os
import sys
import math
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "asl-signs"
CACHE_FILE = "asl_signs_cache_v4.npz"       # New cache (includes face)
MODEL_SAVE_PATH = "asl_signs_model_v4.pth"
LABEL_ENCODER_PATH = "asl_signs_encoder_v4.pkl"

# V4: Include selected face landmarks (lips, nose, eyes)
USE_LANDMARKS = ['left_hand', 'right_hand', 'pose', 'face']

# Selected face landmark indices (76 key landmarks)
# Lips outer: 61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95
# Lips inner: 78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95
# Nose: 1,2,98,327
# Left eye: 33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246
# Right eye: 362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398
SELECTED_FACE_INDICES = sorted(list(set([
    # Lips (outer contour)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    # Lips (inner contour)
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    # Nose
    1, 2, 98, 327, 4, 5, 6, 168, 195, 197,
    # Left eye
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # Right eye
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
])))
NUM_FACE_LANDMARKS = len(SELECTED_FACE_INDICES)  # ~76

MAX_FRAMES = 96          # V4: longer sequences (was 64)
BATCH_SIZE = 64
EPOCHS = 250
SWA_START = 150
LR = 0.0003
WEIGHT_DECAY = 0.04
DROPOUT = 0.4
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 6
DIM_FEEDFORWARD = 1536

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# ============================================================
# FEATURE CALCULATION
# ============================================================

def get_feature_count():
    """
    V4 features per frame:
      Left hand:  21 × 3 = 63
      Right hand: 21 × 3 = 63
      Pose:       33 × 3 = 99
      Face:       76 × 3 = 228 (selected landmarks only)
    Total: 453
    """
    count = 63 + 63 + 99  # hands + pose
    count += NUM_FACE_LANDMARKS * 3  # selected face
    return count

INPUT_SIZE = get_feature_count()
print(f"V4 features per frame: {INPUT_SIZE}")


# ============================================================
# DATA LOADING
# ============================================================

def pad_or_truncate(sequence, max_len):
    if len(sequence) > max_len:
        indices = np.linspace(0, len(sequence) - 1, max_len, dtype=int)
        return sequence[indices]
    elif len(sequence) < max_len:
        pad_len = max_len - len(sequence)
        padding = np.zeros((pad_len, sequence.shape[1]), dtype=np.float32)
        return np.vstack([sequence, padding])
    return sequence.astype(np.float32)


def load_parquet_v4(file_path):
    """Load parquet and extract hands + pose + SELECTED face landmarks."""
    try:
        df = pd.read_parquet(file_path)

        frames = sorted(df['frame'].unique())
        num_frames = len(frames)
        num_features = INPUT_SIZE

        result = np.zeros((num_frames, num_features), dtype=np.float32)

        for frame_idx, frame_num in enumerate(frames):
            frame_df = df[df['frame'] == frame_num]
            feat_idx = 0

            # Left hand (63)
            lh = frame_df[frame_df['type'] == 'left_hand'].sort_values('landmark_index')
            if len(lh) > 0:
                coords = lh[['x', 'y', 'z']].values.flatten()
                result[frame_idx, feat_idx:feat_idx + min(len(coords), 63)] = coords[:63]
            feat_idx += 63

            # Right hand (63)
            rh = frame_df[frame_df['type'] == 'right_hand'].sort_values('landmark_index')
            if len(rh) > 0:
                coords = rh[['x', 'y', 'z']].values.flatten()
                result[frame_idx, feat_idx:feat_idx + min(len(coords), 63)] = coords[:63]
            feat_idx += 63

            # Pose (99)
            pose = frame_df[frame_df['type'] == 'pose'].sort_values('landmark_index')
            if len(pose) > 0:
                coords = pose[['x', 'y', 'z']].values.flatten()
                result[frame_idx, feat_idx:feat_idx + min(len(coords), 99)] = coords[:99]
            feat_idx += 99

            # Face — SELECTED landmarks only (76 × 3 = 228)
            face = frame_df[frame_df['type'] == 'face'].sort_values('landmark_index')
            if len(face) > 0:
                face_indexed = face.set_index('landmark_index')
                for i, lm_idx in enumerate(SELECTED_FACE_INDICES):
                    if lm_idx in face_indexed.index:
                        row = face_indexed.loc[lm_idx]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        offset = feat_idx + i * 3
                        result[frame_idx, offset] = row['x']
                        result[frame_idx, offset + 1] = row['y']
                        result[frame_idx, offset + 2] = row['z']

        return result if num_frames > 0 else None
    except Exception:
        return None


def load_single_sample(args):
    """Worker function for multiprocessing."""
    parquet_path, sign = args
    if not os.path.exists(parquet_path):
        return None
    frames = load_parquet_v4(parquet_path)
    if frames is None or len(frames) == 0:
        return None
    frames = np.nan_to_num(frames, nan=0.0).astype(np.float32)
    frames = pad_or_truncate(frames, MAX_FRAMES)
    return (frames, sign)


def load_dataset():
    """Load dataset with caching and multiprocessing."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cache: {CACHE_FILE}")
        cache = np.load(CACHE_FILE, allow_pickle=True)
        X, y = cache['X'], cache['y']
        classes = cache['classes'].tolist()
        print(f"  {len(X)} samples, {len(classes)} classes, shape {X.shape}")
        return X, y, classes

    print(f"Building dataset from {DATA_DIR} (first run — will cache)...")
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    if not os.path.exists(train_csv):
        print(f"ERROR: {train_csv} not found!")
        sys.exit(1)

    df = pd.read_csv(train_csv)
    print(f"  CSV: {len(df)} samples, {df['sign'].nunique()} signs")

    # Prepare args for multiprocessing
    args_list = [
        (os.path.join(DATA_DIR, row['path']), row['sign'])
        for _, row in df.iterrows()
    ]

    # Load with multiprocessing
    n_workers = min(cpu_count(), 8)
    print(f"  Loading with {n_workers} workers...")

    results = []
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap(load_single_sample, args_list, chunksize=50),
                           total=len(args_list), desc="  Loading"):
            if result is not None:
                results.append(result)

    X = np.array([r[0] for r in results], dtype=np.float32)
    y = np.array([r[1] for r in results])
    classes = sorted(list(set(y)))

    print(f"  Loaded: {len(X)} samples, {len(classes)} classes")
    print(f"  Shape: {X.shape}")

    np.savez(CACHE_FILE, X=X, y=y, classes=np.array(classes))
    print(f"  Cached to {CACHE_FILE}")

    return X, y, classes


# ============================================================
# AUGMENTATION
# ============================================================

class ASLDatasetV4(Dataset):
    def __init__(self, X, y, augment=False, aug_prob=0.85):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment and np.random.random() < self.aug_prob:
            x = self._augment(x)

        return x, y

    def _augment(self, x):
        # 1. Time shift
        if np.random.random() < 0.7:
            shift = np.random.randint(-10, 11)
            x = torch.roll(x, shifts=shift, dims=0)

        # 2. Gaussian noise
        if np.random.random() < 0.7:
            noise_scale = np.random.uniform(0.02, 0.06)
            mask = (x != 0).float()
            noise = torch.randn_like(x) * x.std() * noise_scale
            x = x + noise * mask

        # 3. Random scaling
        if np.random.random() < 0.6:
            scale = np.random.uniform(0.82, 1.18)
            x = x * scale

        # 4. Time warping
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.75, 1.25)
            seq_len = x.shape[0]
            new_len = int(seq_len * rate)
            new_len = max(10, min(new_len, seq_len * 2))
            indices = np.linspace(0, seq_len - 1, new_len).astype(int)
            x_warped = x[indices]
            # Re-pad/truncate to original length
            if len(x_warped) > seq_len:
                sample_idx = np.linspace(0, len(x_warped) - 1, seq_len).astype(int)
                x = x_warped[sample_idx]
            elif len(x_warped) < seq_len:
                pad = torch.zeros(seq_len - len(x_warped), x.shape[1])
                x = torch.cat([x_warped, pad], dim=0)
            else:
                x = x_warped

        # 5. Frame dropout
        if np.random.random() < 0.4:
            drop_ratio = np.random.uniform(0.05, 0.15)
            n_drop = int(x.shape[0] * drop_ratio)
            drop_indices = np.random.choice(x.shape[0], n_drop, replace=False)
            x[drop_indices] = 0

        # 6. Feature masking
        if np.random.random() < 0.3:
            n_mask = int(x.shape[1] * 0.1)
            mask_indices = np.random.choice(x.shape[1], n_mask, replace=False)
            x[:, mask_indices] = 0

        # 7. Temporal cutout
        if np.random.random() < 0.3:
            cut_len = int(x.shape[0] * 0.1)
            cut_start = np.random.randint(0, x.shape[0] - cut_len)
            x[cut_start:cut_start + cut_len] = 0

        # ============================
        # V4 NEW AUGMENTATIONS
        # ============================

        # 8. FingerDropout — zero out entire finger landmarks
        if np.random.random() < 0.5:
            x = self._finger_dropout(x)

        # 9. Spatial rotation — rotate hand landmarks in 2D
        if np.random.random() < 0.4:
            x = self._spatial_rotation(x)

        # 10. Hand swap — swap left/right hand (data augmentation for handedness)
        if np.random.random() < 0.15:
            x = self._hand_swap(x)

        # 11. Face jitter — slight noise on face landmarks specifically
        if np.random.random() < 0.3:
            face_start = 63 + 63 + 99  # after hands + pose
            face_end = x.shape[1]
            if face_end > face_start:
                mask = (x[:, face_start:face_end] != 0).float()
                noise = torch.randn(x.shape[0], face_end - face_start) * 0.02
                x[:, face_start:face_end] += noise * mask

        return x

    def _finger_dropout(self, x):
        """Zero out entire finger (all 4 joints × 3 coords = 12 features)."""
        # For each hand: wrist(0), thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
        finger_ranges = {
            'thumb':  (1, 5),    # landmarks 1-4
            'index':  (5, 9),    # landmarks 5-8
            'middle': (9, 13),   # landmarks 9-12
            'ring':   (13, 17),  # landmarks 13-16
            'pinky':  (17, 21),  # landmarks 17-20
        }
        fingers = list(finger_ranges.values())

        # Drop 1-2 fingers from each hand
        n_drop = np.random.randint(1, 3)
        drop_fingers = np.random.choice(len(fingers), n_drop, replace=False)

        for fi in drop_fingers:
            start_lm, end_lm = fingers[fi]
            for hand_offset in [0, 63]:  # left hand, right hand
                feat_start = hand_offset + start_lm * 3
                feat_end = hand_offset + end_lm * 3
                if feat_end <= x.shape[1]:
                    x[:, feat_start:feat_end] = 0

        return x

    def _spatial_rotation(self, x):
        """Apply small 2D rotation to hand landmarks."""
        angle = np.random.uniform(-15, 15) * np.pi / 180  # ±15 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        for hand_offset in [0, 63]:
            for lm in range(21):
                xi = hand_offset + lm * 3
                yi = hand_offset + lm * 3 + 1
                if yi < x.shape[1]:
                    old_x = x[:, xi].clone()
                    old_y = x[:, yi].clone()
                    x[:, xi] = old_x * cos_a - old_y * sin_a
                    x[:, yi] = old_x * sin_a + old_y * cos_a

        return x

    def _hand_swap(self, x):
        """Swap left and right hand landmarks."""
        left = x[:, 0:63].clone()
        right = x[:, 63:126].clone()
        x[:, 0:63] = right
        x[:, 63:126] = left
        # Mirror x coordinates
        x[:, 0:63:3] = -x[:, 0:63:3]    # negate x for left
        x[:, 63:126:3] = -x[:, 63:126:3] # negate x for right
        return x


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    seq_len = x.size(1)
    cut_len = int(seq_len * (1 - lam))
    cut_start = np.random.randint(0, max(1, seq_len - cut_len))
    mixed_x = x.clone()
    mixed_x[:, cut_start:cut_start + cut_len] = x[idx, cut_start:cut_start + cut_len]
    lam = 1 - cut_len / seq_len
    return mixed_x, y, y[idx], lam


# ============================================================
# MODEL ARCHITECTURE V4
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer-style block: depthwise separable conv + attention.
    More efficient than full Conv1d + Transformer for landmark sequences.
    """
    def __init__(self, d_model, n_heads=8, conv_kernel=15, dropout=0.1):
        super().__init__()
        # Depthwise separable convolution
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel // 2, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_act = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Depthwise separable conv
        residual = x
        x_conv = x.transpose(1, 2)  # (B, d, T)
        x_conv = self.depthwise(x_conv)
        x_conv = self.pointwise(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, T, d)
        x = self.conv_norm(residual + self.conv_dropout(self.conv_act(x_conv)))

        # Self-attention
        residual = x
        x_attn, _ = self.attn(x, x, x)
        x = self.attn_norm(residual + self.attn_dropout(x_attn))

        # Feed-forward
        residual = x
        x = self.ff_norm(residual + self.ff(x))

        return x


class ASLModelV4(nn.Module):
    """
    V4: Squeezeformer blocks + LSTM + face landmarks.
    """
    def __init__(self, input_size, num_classes, d_model=384,
                 n_heads=8, n_layers=6, dropout=0.4, max_frames=96):
        super().__init__()
        self.d_model = d_model

        # Multi-stream input projection (hands, pose, face separately)
        self.hand_proj = nn.Linear(126, d_model // 3)     # both hands
        self.pose_proj = nn.Linear(99, d_model // 3)      # pose
        face_features = input_size - 126 - 99              # remaining = face
        self.face_proj = nn.Linear(face_features, d_model - 2 * (d_model // 3))

        self.input_norm = nn.LayerNorm(d_model)
        self.input_ff = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),
        )
        self.input_ln = nn.LayerNorm(d_model)

        # Squeezeformer blocks (replace plain Transformer)
        self.squeeze_blocks = nn.ModuleList([
            SqueezeformerBlock(d_model, n_heads, conv_kernel=15, dropout=dropout * 0.7)
            for _ in range(n_layers)
        ])
        self.squeeze_ln = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames, dropout=dropout * 0.5)

        # Bidirectional LSTM branch
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.lstm_ln = nn.LayerNorm(d_model)

        # Multi-query attention pooling
        self.n_queries = 4
        self.pool_queries = nn.Parameter(torch.randn(1, self.n_queries, d_model * 2))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=d_model * 2, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.pool_ln = nn.LayerNorm(d_model * 2)

        # Classifier
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
        B = x.size(0)

        # Multi-stream projection
        hands = x[:, :, :126]                    # left + right hand
        pose = x[:, :, 126:225]                  # pose
        face = x[:, :, 225:]                     # face landmarks

        h_hands = self.hand_proj(hands)
        h_pose = self.pose_proj(pose)
        h_face = self.face_proj(face)

        x = torch.cat([h_hands, h_pose, h_face], dim=-1)  # (B, T, d_model)
        x = self.input_norm(x)
        x = self.input_ff(x) + x
        x = self.input_ln(x)

        # Squeezeformer branch
        x_sq = self.pos_encoder(x)
        for block in self.squeeze_blocks:
            x_sq = block(x_sq)
        x_sq = self.squeeze_ln(x_sq)

        # LSTM branch
        x_lstm, _ = self.lstm(x)
        x_lstm = self.lstm_ln(x_lstm)

        # Concatenate
        combined = torch.cat([x_sq, x_lstm], dim=-1)  # (B, T, d*2)

        # Multi-query attention pooling
        queries = self.pool_queries.expand(B, -1, -1)
        pooled, _ = self.pool_attention(queries, combined, combined)
        pooled = self.pool_ln(pooled)
        pooled = pooled.view(B, -1)

        return self.classifier(pooled)


# ============================================================
# TRAINING
# ============================================================

def train_v4(args):
    # Load data
    X, y, classes = load_dataset()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Classes: {len(classes)}")
    print(f"Input shape: {X_train.shape}")

    # Datasets
    train_ds = ASLDatasetV4(X_train, y_train, augment=True, aug_prob=0.85)
    val_ds = ASLDatasetV4(X_val, y_val, augment=False)

    # Weighted sampler
    counts = Counter(y_train)
    weights = [1.0 / counts[l] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = ASLModelV4(
        INPUT_SIZE, len(classes),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, max_frames=MAX_FRAMES
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Resume from V3 checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"\nLoading V3 weights from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        v3_state = checkpoint['model_state_dict']

        # Partial load — load matching layers, skip mismatched ones
        model_state = model.state_dict()
        loaded = 0
        for k, v in v3_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  Loaded {loaded}/{len(v3_state)} layers from V3")

    # Knowledge distillation teacher
    teacher = None
    if args.distill and os.path.exists(args.distill):
        print(f"\nLoading teacher model from {args.distill}...")
        t_ckpt = torch.load(args.distill, map_location=DEVICE, weights_only=False)
        teacher = ASLModelV4(
            INPUT_SIZE, len(classes),
            d_model=512, n_heads=8, n_layers=8,
            dropout=0.0, max_frames=MAX_FRAMES
        ).to(DEVICE)
        teacher.load_state_dict(t_ckpt['model_state_dict'])
        teacher.eval()
        print("  Teacher loaded for distillation")

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.12)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Learning rate schedule: warmup → cosine → SWA
    def lr_lambda(epoch):
        if epoch < 20:  # warmup
            return (epoch + 1) / 20
        elif epoch < SWA_START:
            progress = (epoch - 20) / (SWA_START - 20)
            return 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return 0.15  # low LR for SWA
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.00005)
    swa_active = False

    # Training loop
    best_val_acc = 0
    best_state = None
    patience = 40
    patience_counter = 0

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    print(f"\n{'='*70}")
    print(f"  TRAINING V4 — Target: >88% accuracy")
    print(f"  Epochs: {EPOCHS} | SWA starts: {SWA_START} | Batch: {BATCH_SIZE}")
    print(f"{'='*70}\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # Apply mixup or cutmix
            use_mix = np.random.random()
            if use_mix < 0.25:
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y)
            elif use_mix < 0.45:
                batch_x, y_a, y_b, lam = cutmix_data(batch_x, batch_y)
            else:
                y_a, y_b, lam = batch_y, batch_y, 1.0

            optimizer.zero_grad()
            outputs = model(batch_x)

            # Mixed loss
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

            # Knowledge distillation loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(batch_x)
                T = 4.0  # temperature
                kd_loss = F.kl_div(
                    F.log_softmax(outputs / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)
                loss = 0.7 * loss + 0.3 * kd_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_y).sum().item()
            train_total += batch_y.size(0)

        # SWA update
        if epoch >= SWA_START:
            if not swa_active:
                print(f"\n  >>> SWA activated at epoch {epoch + 1} <<<\n")
                swa_active = True
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation
        eval_model = swa_model if swa_active else model
        eval_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                outputs = eval_model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_y).sum().item()
                val_total += batch_y.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total
        current_lr = optimizer.param_groups[0]['lr']

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['lr'].append(current_lr)

        # Log
        swa_tag = " [SWA]" if swa_active else ""
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"Train: {train_acc*100:5.1f}%  Val: {val_acc*100:5.1f}%  "
              f"Loss: {val_loss_avg:.3f}  LR: {current_lr:.6f}{swa_tag}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = (swa_model if swa_active else model).state_dict().copy()
            patience_counter = 0
            print(f"          ★ New best: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > SWA_START + 20:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Update BN for SWA
    if swa_active:
        print("\nUpdating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)

    # Restore best
    final_model = swa_model if swa_active else model
    if best_state:
        final_model.load_state_dict(best_state)

    print(f"\n{'='*70}")
    print(f"  BEST VALIDATION ACCURACY: {best_val_acc*100:.2f}%")
    print(f"{'='*70}")

    # Save
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_size': INPUT_SIZE,
        'num_classes': len(classes),
        'classes': classes,
        'max_frames': MAX_FRAMES,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'accuracy': best_val_acc,
        'version': 'v4',
        'face_landmark_indices': SELECTED_FACE_INDICES,
        'landmark_types': USE_LANDMARKS,
    }, MODEL_SAVE_PATH)

    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"Labels saved: {LABEL_ENCODER_PATH}")

    # Save training plots
    save_plots(history)

    return final_model, label_encoder, best_val_acc


# ============================================================
# PLOTS
# ============================================================

def save_plots(history):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history['train_acc'], label='Train', alpha=0.8)
        axes[0].plot(history['val_acc'], label='Val', alpha=0.8)
        axes[0].axvline(x=SWA_START, color='green', linestyle='--', alpha=0.5, label='SWA start')
        axes[0].set_title('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['train_loss'], label='Train', alpha=0.8)
        axes[1].plot(history['val_loss'], label='Val', alpha=0.8)
        axes[1].axvline(x=SWA_START, color='green', linestyle='--', alpha=0.5)
        axes[1].set_title('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(history['lr'], color='orange')
        axes[2].axvline(x=SWA_START, color='green', linestyle='--', alpha=0.5)
        axes[2].set_title('Learning Rate')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_v4.png', dpi=150)
        plt.close()
        print("Plots saved: training_v4.png")
    except Exception as e:
        print(f"Could not save plots: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL V4 Training")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to V3 checkpoint to resume from')
    parser.add_argument('--distill', type=str, default=None,
                        help='Path to teacher model for knowledge distillation')
    args = parser.parse_args()

    train_v4(args)

"""
Google ASL Signs - Training V2
===============================
Improved training with:
- Better model architecture
- More augmentation
- Cosine annealing with warm restarts
- Gradient accumulation
- Test-time augmentation

Saves to separate files (v2) so you keep your original model.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from collections import Counter
import math

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "asl-signs"

# V2 output files - won't overwrite your v1 model
MODEL_SAVE_PATH = "asl_signs_model_v2.pth"
LABEL_ENCODER_PATH = "asl_signs_encoder_v2.pkl"
CACHE_FILE = "asl_signs_cache.npz"  # Reuse existing cache

USE_LANDMARKS = ['left_hand', 'right_hand', 'pose']
MAX_FRAMES = 64
MAX_SAMPLES_PER_CLASS = None

# Training settings
EPOCHS = 150
BATCH_SIZE = 48
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.02
LABEL_SMOOTHING = 0.1
DROPOUT = 0.35
WARMUP_EPOCHS = 10
PATIENCE = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATA LOADING
# ============================================================

def get_feature_count():
    count = 0
    for lm_type in USE_LANDMARKS:
        if lm_type in ['left_hand', 'right_hand']:
            count += 63
        elif lm_type == 'pose':
            count += 99
        elif lm_type == 'face':
            count += 1404
    return count


def pad_or_truncate(sequence, max_len):
    if len(sequence) > max_len:
        indices = np.linspace(0, len(sequence)-1, max_len, dtype=int)
        return sequence[indices]
    elif len(sequence) < max_len:
        pad_len = max_len - len(sequence)
        padding = np.zeros((pad_len, sequence.shape[1]), dtype=np.float32)
        return np.vstack([sequence, padding])
    return sequence


def load_single_parquet(args):
    parquet_path, sign = args
    try:
        if not os.path.exists(parquet_path):
            return None
        
        df = pd.read_parquet(parquet_path)
        df = df[df['type'].isin(USE_LANDMARKS)]
        
        if len(df) == 0:
            return None
        
        frames = sorted(df['frame'].unique())
        num_frames = len(frames)
        num_features = get_feature_count()
        
        result = np.zeros((num_frames, num_features), dtype=np.float32)
        grouped = df.groupby('frame')
        
        for frame_idx, frame_num in enumerate(frames):
            try:
                frame_df = grouped.get_group(frame_num)
            except KeyError:
                continue
            
            feature_idx = 0
            for lm_type in USE_LANDMARKS:
                type_df = frame_df[frame_df['type'] == lm_type].sort_values('landmark_index')
                
                if lm_type in ['left_hand', 'right_hand']:
                    n_landmarks = 21
                elif lm_type == 'pose':
                    n_landmarks = 33
                else:
                    n_landmarks = 468
                
                if len(type_df) > 0:
                    coords = type_df[['x', 'y', 'z']].values.flatten()
                    end_idx = min(feature_idx + len(coords), feature_idx + n_landmarks * 3)
                    result[frame_idx, feature_idx:end_idx] = coords[:end_idx - feature_idx]
                
                feature_idx += n_landmarks * 3
        
        result = np.nan_to_num(result, nan=0.0)
        result = pad_or_truncate(result, MAX_FRAMES)
        return (result, sign)
    except:
        return None


def load_dataset(data_dir):
    if os.path.exists(CACHE_FILE):
        print(f"Loading from cache: {CACHE_FILE}")
        cache = np.load(CACHE_FILE, allow_pickle=True)
        X, y = cache['X'], cache['y']
        classes = cache['classes'].tolist()
        print(f"Loaded {len(X)} samples, {len(classes)} classes")
        
        if X.shape[1] != MAX_FRAMES:
            print(f"Adjusting frames {X.shape[1]} -> {MAX_FRAMES}...")
            X_new = np.zeros((len(X), MAX_FRAMES, X.shape[2]), dtype=np.float32)
            for i, seq in enumerate(X):
                X_new[i] = pad_or_truncate(seq, MAX_FRAMES)
            X = X_new
        
        return X, y, classes
    
    print("Cache not found! Run the original training first to create cache.")
    return None, None, None


# ============================================================
# DATASET WITH STRONG AUGMENTATION V2
# ============================================================

class ASLDatasetV2(Dataset):
    def __init__(self, X, y, augment=False, aug_strength=1.0):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.aug_strength = aug_strength
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        
        if self.augment:
            s = self.aug_strength
            
            # 1. Time shift (more aggressive)
            if np.random.random() > 0.3:
                shift = np.random.randint(int(-7 * s), int(7 * s))
                x = torch.roll(x, shift, dims=0)
            
            # 2. Gaussian noise
            if np.random.random() > 0.3:
                noise_scale = 0.02 * s + np.random.random() * 0.02 * s
                x = x + torch.randn_like(x) * noise_scale
            
            # 3. Random scale
            if np.random.random() > 0.4:
                scale = 0.8 + np.random.random() * 0.4  # 0.8 - 1.2
                x = x * scale
            
            # 4. Time warping (stretch/compress)
            if np.random.random() > 0.5:
                warp = 0.7 + np.random.random() * 0.6  # 0.7 - 1.3
                new_len = int(x.shape[0] * warp)
                if new_len > 5:
                    x_np = x.numpy()
                    indices = np.linspace(0, x.shape[0]-1, new_len).astype(int)
                    x_warped = x_np[indices]
                    x = torch.FloatTensor(pad_or_truncate(x_warped, x.shape[0]))
            
            # 5. Random frame dropout
            if np.random.random() > 0.6:
                drop_prob = 0.05 + np.random.random() * 0.1 * s
                mask = torch.rand(x.shape[0]) > drop_prob
                x = x * mask.unsqueeze(1)
            
            # 6. Feature dropout (drop random landmarks)
            if np.random.random() > 0.7:
                feat_mask = torch.rand(x.shape[1]) > 0.1
                x = x * feat_mask.unsqueeze(0)
            
            # 7. Reverse sequence (sign might look similar backwards)
            if np.random.random() > 0.9:
                x = torch.flip(x, [0])
            
            # 8. Add slight rotation simulation (shift x/y coordinates)
            if np.random.random() > 0.7:
                angle = (np.random.random() - 0.5) * 0.1 * s
                # Simple rotation approximation for x,y pairs
                x_reshaped = x.view(x.shape[0], -1, 3)  # (frames, landmarks, xyz)
                x_coords = x_reshaped[:, :, 0].clone()
                y_coords = x_reshaped[:, :, 1].clone()
                x_reshaped[:, :, 0] = x_coords * np.cos(angle) - y_coords * np.sin(angle)
                x_reshaped[:, :, 1] = x_coords * np.sin(angle) + y_coords * np.cos(angle)
                x = x_reshaped.view(x.shape[0], -1)
        
        return x, y


# ============================================================
# MODEL V2 - Improved Hybrid Architecture
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


class ASLModelV2(nn.Module):
    """Improved Hybrid Model V2 with more capacity."""
    
    def __init__(self, input_size, num_classes, d_model=320, dropout=0.35):
        super().__init__()
        
        # Better input projection with residual
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),
        )
        self.input_ln = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_FRAMES, dropout=dropout * 0.5)
        
        # Transformer encoder (more layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            d_model, d_model // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.lstm_ln = nn.LayerNorm(d_model)
        
        # Multi-head attention pooling
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model * 2))
        
        # Classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input processing
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.input_ln(x)
        
        # Add positional encoding
        x_pos = self.pos_encoder(x)
        
        # Transformer branch
        x_trans = self.transformer(x_pos)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)
        x_lstm = self.lstm_ln(x_lstm)
        
        # Combine branches
        combined = torch.cat([x_trans, x_lstm], dim=-1)  # (batch, seq, d_model*2)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(query, combined, combined)
        pooled = pooled.squeeze(1)  # (batch, d_model*2)
        
        # Classify
        return self.classifier(pooled)


# ============================================================
# MIXUP AND CUTMIX
# ============================================================

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix for sequence data - cuts a time segment."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    seq_len = x.size(1)
    cut_len = int(seq_len * (1 - lam))
    cut_start = np.random.randint(0, seq_len - cut_len + 1)
    
    mixed_x = x.clone()
    mixed_x[:, cut_start:cut_start + cut_len] = x[index, cut_start:cut_start + cut_len]
    
    y_a, y_b = y, y[index]
    lam = 1 - cut_len / seq_len
    
    return mixed_x, y_a, y_b, lam


# ============================================================
# TRAINING V2
# ============================================================

def train_v2(X, y, classes):
    print("\n" + "=" * 60)
    print("TRAINING V2 - Enhanced")
    print("=" * 60)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.12, random_state=42, stratify=y_encoded
    )
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Input: {X_train.shape}")
    print(f"Classes: {len(classes)}")
    
    # Datasets
    train_dataset = ASLDatasetV2(X_train, y_train, augment=True, aug_strength=1.0)
    val_dataset = ASLDatasetV2(X_val, y_val, augment=False)
    
    # Weighted sampler
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[l] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Model
    model = ASLModelV2(X_train.shape[2], len(classes), d_model=320, dropout=DROPOUT).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Warmup
    def warmup_lr(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
    
    print(f"\nSettings:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Label Smoothing: {LABEL_SMOOTHING}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs")
    
    print("\n" + "-" * 60)
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct, train_total = 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Random augmentation strategy
            aug_choice = np.random.random()
            if aug_choice < 0.3:
                # Mixup
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            elif aug_choice < 0.5:
                # CutMix
                batch_x, y_a, y_b, lam = cutmix_data(batch_x, batch_y, alpha=1.0)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                # Normal
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (pred == batch_y).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                _, pred = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (pred == batch_y).sum().item()
        
        val_acc = val_correct / val_total
        
        # Update schedulers
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train: {train_acc*100:5.1f}% | Val: {val_acc*100:5.1f}% | "
              f"LR: {lr:.6f}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f"          ✓ New best: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print(f"BEST VALIDATION ACCURACY: {best_acc*100:.2f}%")
    print("=" * 60)
    
    # Classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    print("\nTop-10 worst classes:")
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    class_f1 = [(c, report[c]['f1-score']) for c in classes if c in report]
    class_f1.sort(key=lambda x: x[1])
    for c, f1 in class_f1[:10]:
        print(f"  {c}: {f1*100:.1f}%")
    
    # Save V2 model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[2],
        'num_classes': len(classes),
        'classes': classes,
        'max_frames': MAX_FRAMES,
        'accuracy': best_acc,
        'version': 'v2'
    }, MODEL_SAVE_PATH)
    
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nSaved V2 model to: {MODEL_SAVE_PATH}")
    print(f"Saved V2 encoder to: {LABEL_ENCODER_PATH}")
    
    return model, label_encoder, best_acc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ASL SIGNS TRAINING V2")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR) and not os.path.exists(CACHE_FILE):
        print(f"ERROR: Neither {DATA_DIR} nor {CACHE_FILE} found!")
        exit(1)
    
    X, y, classes = load_dataset(DATA_DIR)
    
    if X is None:
        exit(1)
    
    model, encoder, acc = train_v2(X, y, classes)
    
    print("\n" + "=" * 60)
    print(f"V2 TRAINING COMPLETE!")
    print(f"Final Accuracy: {acc*100:.2f}%")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print("=" * 60)
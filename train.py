"""
Google ASL Signs - ENHANCED Transformer Model
==============================================
Transformer encoder for better sequence modeling.
Target: 75-80%+ accuracy
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
MODEL_SAVE_PATH = "asl_signs_model.pth"
LABEL_ENCODER_PATH = "asl_signs_encoder.pkl"
CACHE_FILE = "asl_signs_cache_v2.npz"  # New cache with face landmarks

# Include face for more features - BUT check if cache exists first
USE_LANDMARKS = ['left_hand', 'right_hand', 'pose']  # Start without face
CACHE_FILE = "asl_signs_cache.npz"  # Use existing cache if available
MAX_FRAMES = 64
MAX_SAMPLES_PER_CLASS = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATA LOADING (same fast loading)
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


def load_dataset_fast(data_dir, max_samples_per_class=None):
    if os.path.exists(CACHE_FILE):
        print(f"Loading from cache: {CACHE_FILE}")
        cache = np.load(CACHE_FILE, allow_pickle=True)
        X, y = cache['X'], cache['y']
        classes = cache['classes'].tolist()
        print(f"Loaded {len(X)} samples, {len(classes)} classes")
        
        # Adjust frames if needed
        if X.shape[1] != MAX_FRAMES:
            print(f"Adjusting frames {X.shape[1]} -> {MAX_FRAMES}...")
            X_new = np.zeros((len(X), MAX_FRAMES, X.shape[2]), dtype=np.float32)
            for i, seq in enumerate(X):
                X_new[i] = pad_or_truncate(seq, MAX_FRAMES)
            X = X_new
        
        return X, y, classes
    
    print(f"Loading dataset...")
    print("This will be cached for next time\n")
    
    train_csv = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(train_csv)
    print(f"Total samples: {len(df)} | Classes: {df['sign'].nunique()}")
    
    if max_samples_per_class:
        df = df.groupby('sign').head(max_samples_per_class)
        print(f"Limited to {len(df)} samples")
    
    # Single-threaded loading (stable)
    print(f"\nLoading {len(df)} parquet files...")
    
    X, y = [], []
    total = len(df)
    
    for idx, row in df.iterrows():
        parquet_path = os.path.join(data_dir, row['path'])
        result = load_single_parquet((parquet_path, row['sign']))
        
        if result is not None:
            X.append(result[0])
            y.append(result[1])
        
        if (idx + 1) % 2000 == 0:
            print(f"  {(idx + 1) / total * 100:.0f}% | Loaded: {len(X)}")
    
    print(f"\nTotal loaded: {len(X)}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    classes = sorted(list(set(y)))
    
    print(f"Saving cache...")
    np.savez_compressed(CACHE_FILE, X=X, y=y, classes=np.array(classes))
    
    return X, y, classes


# ============================================================
# DATASET WITH STRONG AUGMENTATION
# ============================================================

class ASLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        
        if self.augment:
            # Time shift
            if np.random.random() > 0.5:
                shift = np.random.randint(-5, 5)
                x = torch.roll(x, shift, dims=0)
            
            # Add noise
            if np.random.random() > 0.5:
                x = x + torch.randn_like(x) * 0.03
            
            # Scale
            if np.random.random() > 0.5:
                x = x * (0.85 + np.random.random() * 0.3)
            
            # Time stretch (interpolate)
            if np.random.random() > 0.7:
                scale = 0.8 + np.random.random() * 0.4
                new_len = int(x.shape[0] * scale)
                if new_len > 2:
                    x_np = x.numpy()
                    indices = np.linspace(0, x.shape[0]-1, new_len).astype(int)
                    x_stretched = x_np[indices]
                    # Pad/truncate back to original size
                    if len(x_stretched) > x.shape[0]:
                        x = torch.FloatTensor(x_stretched[:x.shape[0]])
                    else:
                        pad = np.zeros((x.shape[0] - len(x_stretched), x.shape[1]))
                        x = torch.FloatTensor(np.vstack([x_stretched, pad]))
            
            # Random frame dropout
            if np.random.random() > 0.8:
                mask = torch.rand(x.shape[0]) > 0.1
                x = x * mask.unsqueeze(1)
        
        return x, y


# ============================================================
# TRANSFORMER MODEL
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


class ASLTransformer(nn.Module):
    """Transformer encoder for sign language recognition."""
    
    def __init__(self, input_size, num_classes, d_model=256, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_FRAMES)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling + classification
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        
        return self.classifier(cls_output)


# ============================================================
# HYBRID MODEL (Transformer + LSTM)
# ============================================================

class ASLHybridModel(nn.Module):
    """Combines Transformer and LSTM for best of both."""
    
    def __init__(self, input_size, num_classes, d_model=256, dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer branch
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # LSTM branch
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention for combining
        self.attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Classifier
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
        # Project
        x = self.input_proj(x)
        
        # Transformer branch
        x_trans = self.pos_encoder(x)
        x_trans = self.transformer(x_trans)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)
        
        # Combine
        combined = torch.cat([x_trans, x_lstm], dim=-1)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(combined), dim=1)
        pooled = torch.sum(combined * attn_weights, dim=1)
        
        return self.classifier(pooled)


# ============================================================
# TRAINING WITH MIXUP
# ============================================================

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_model(X, y, classes, epochs=120, batch_size=64, lr=0.0005):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.12, random_state=42, stratify=y_encoded
    )
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Input shape: {X_train.shape}")
    
    train_dataset = ASLDataset(X_train, y_train, augment=True)
    val_dataset = ASLDataset(X_val, y_val, augment=False)
    
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[l] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Use hybrid model
    model = ASLHybridModel(X_train.shape[2], len(classes), d_model=256).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Warmup + cosine decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("\n" + "=" * 55)
    print("TRAINING - Hybrid Transformer + LSTM")
    print("=" * 55)
    
    best_acc = 0.0
    best_state = None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct, train_total = 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Mixup
            if np.random.random() > 0.5:
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (pred == batch_y).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validate
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
        scheduler.step()
        
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {train_acc*100:5.1f}% | Val: {val_acc*100:5.1f}% | LR: {lr_now:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            patience = 0
            print(f"          ✓ New best: {val_acc*100:.2f}%")
        else:
            patience += 1
            if patience >= 25:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print("\n" + "=" * 55)
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print("=" * 55)
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[2],
        'num_classes': len(classes),
        'classes': classes,
        'max_frames': MAX_FRAMES,
    }, MODEL_SAVE_PATH)
    
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Saved: {MODEL_SAVE_PATH}")
    return model, label_encoder, best_acc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("ASL SIGNS - ENHANCED TRANSFORMER TRAINING")
    print("=" * 55)
    
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: {DATA_DIR} not found!")
        exit(1)
    
    X, y, classes = load_dataset_fast(DATA_DIR, MAX_SAMPLES_PER_CLASS)
    
    if X is None:
        exit(1)
    
    model, encoder, acc = train_model(X, y, classes, epochs=120, batch_size=64, lr=0.0005)
    
    print("\n" + "=" * 55)
    print(f"FINAL ACCURACY: {acc*100:.2f}%")
    print("=" * 55)
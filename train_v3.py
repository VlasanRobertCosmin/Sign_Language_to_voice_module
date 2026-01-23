"""
Google ASL Signs - Training V3
===============================
Maximum accuracy version with:
- Larger model (d_model=384, 6 layers)
- Longer training (200 epochs)
- Better augmentation
- Stochastic Weight Averaging (SWA)
- Test-Time Augmentation ready
- Optional class filtering

Saves to v3 files.
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
from torch.optim.swa_utils import AveragedModel, SWALR
from collections import Counter
import math
import copy
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "asl-signs"

# V3 output files
MODEL_SAVE_PATH = "asl_signs_model_v3.pth"
LABEL_ENCODER_PATH = "asl_signs_encoder_v3.pkl"
CACHE_FILE = "asl_signs_cache.npz"
GRAPHS_DIR = "training_graphs_v3"  # Folder for graphs

USE_LANDMARKS = ['left_hand', 'right_hand', 'pose']
MAX_FRAMES = 64

# Class filtering (set to None to use all classes)
MIN_SAMPLES_PER_CLASS = 30  # Classes must have at least this many samples
MAX_CLASSES = None  # Set to 100, 150, etc. to limit classes (None = all)

# Training settings - V3 enhanced
EPOCHS = 200
BATCH_SIZE = 48
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.03
LABEL_SMOOTHING = 0.15
DROPOUT = 0.4
WARMUP_EPOCHS = 15
PATIENCE = 35

# Model settings
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 6
DIM_FEEDFORWARD = 1536  # 4x d_model

# SWA settings
SWA_START = 100  # Start SWA after this epoch
SWA_LR = 0.00005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


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


def load_dataset():
    if not os.path.exists(CACHE_FILE):
        print(f"Cache not found: {CACHE_FILE}")
        print("Run the original training first to create cache.")
        return None, None, None
    
    print(f"Loading from cache: {CACHE_FILE}")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    X, y = cache['X'], cache['y']
    
    print(f"Loaded {len(X)} samples")
    
    # Adjust frames if needed
    if X.shape[1] != MAX_FRAMES:
        print(f"Adjusting frames {X.shape[1]} -> {MAX_FRAMES}...")
        X_new = np.zeros((len(X), MAX_FRAMES, X.shape[2]), dtype=np.float32)
        for i, seq in enumerate(X):
            X_new[i] = pad_or_truncate(seq, MAX_FRAMES)
        X = X_new
    
    # Filter classes
    if MIN_SAMPLES_PER_CLASS or MAX_CLASSES:
        print(f"\nFiltering classes...")
        class_counts = Counter(y)
        print(f"Original classes: {len(class_counts)}")
        
        # Filter by minimum samples
        if MIN_SAMPLES_PER_CLASS:
            valid_classes = {c for c, n in class_counts.items() if n >= MIN_SAMPLES_PER_CLASS}
            print(f"Classes with >= {MIN_SAMPLES_PER_CLASS} samples: {len(valid_classes)}")
        else:
            valid_classes = set(class_counts.keys())
        
        # Limit number of classes (take most frequent)
        if MAX_CLASSES and len(valid_classes) > MAX_CLASSES:
            sorted_classes = sorted(valid_classes, key=lambda c: class_counts[c], reverse=True)
            valid_classes = set(sorted_classes[:MAX_CLASSES])
            print(f"Limited to top {MAX_CLASSES} classes")
        
        # Filter data
        mask = np.array([yi in valid_classes for yi in y])
        X = X[mask]
        y = y[mask]
        print(f"Filtered samples: {len(X)}")
    
    classes = sorted(list(set(y)))
    print(f"Final classes: {len(classes)}")
    
    return X, y, classes


# ============================================================
# DATASET WITH ADVANCED AUGMENTATION
# ============================================================

class ASLDatasetV3(Dataset):
    def __init__(self, X, y, augment=False, aug_prob=0.8):
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
            x = self.apply_augmentation(x)
        
        return x, y
    
    def apply_augmentation(self, x):
        # 1. Time shift
        if np.random.random() > 0.3:
            shift = np.random.randint(-8, 8)
            x = torch.roll(x, shift, dims=0)
        
        # 2. Gaussian noise (adaptive)
        if np.random.random() > 0.3:
            std = x.std() * (0.02 + np.random.random() * 0.03)
            x = x + torch.randn_like(x) * std
        
        # 3. Random scaling
        if np.random.random() > 0.4:
            scale = 0.85 + np.random.random() * 0.3
            x = x * scale
        
        # 4. Time warping
        if np.random.random() > 0.5:
            warp = 0.75 + np.random.random() * 0.5
            new_len = max(10, int(x.shape[0] * warp))
            x_np = x.numpy()
            indices = np.linspace(0, x.shape[0]-1, new_len).astype(int)
            x_warped = x_np[indices]
            x = torch.FloatTensor(pad_or_truncate(x_warped, x.shape[0]))
        
        # 5. Frame dropout
        if np.random.random() > 0.6:
            drop_prob = 0.05 + np.random.random() * 0.1
            mask = torch.rand(x.shape[0]) > drop_prob
            x = x * mask.unsqueeze(1)
        
        # 6. Feature masking (landmark dropout)
        if np.random.random() > 0.7:
            n_features = x.shape[1]
            n_mask = int(n_features * 0.1 * np.random.random())
            mask_indices = np.random.choice(n_features, n_mask, replace=False)
            x[:, mask_indices] = 0
        
        # 7. Temporal cutout
        if np.random.random() > 0.7:
            cut_len = int(x.shape[0] * 0.1 * np.random.random())
            cut_start = np.random.randint(0, x.shape[0] - cut_len)
            x[cut_start:cut_start + cut_len] = 0
        
        # 8. Mixup with zeros (fade in/out simulation)
        if np.random.random() > 0.8:
            fade_len = int(x.shape[0] * 0.15)
            fade_in = torch.linspace(0, 1, fade_len).unsqueeze(1)
            fade_out = torch.linspace(1, 0, fade_len).unsqueeze(1)
            x[:fade_len] = x[:fade_len] * fade_in
            x[-fade_len:] = x[-fade_len:] * fade_out
        
        return x


# ============================================================
# MODEL V3 - Maximum Capacity
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
    """Convolutional subsampling like in Conformer."""
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
        # x: (batch, seq, features)
        residual = x
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq, features)
        return self.norm(x + residual)


class ASLModelV3(nn.Module):
    """V3 Model - Larger and more powerful."""
    
    def __init__(self, input_size, num_classes, d_model=384, n_heads=8, 
                 n_layers=6, dim_ff=1536, dropout=0.4):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection with residual
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj1 = nn.Linear(input_size, d_model)
        self.input_proj2 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),
        )
        self.input_ln = nn.LayerNorm(d_model)
        
        # Convolutional feature extraction
        self.conv_subsample = ConvSubsampling(d_model, dropout * 0.5)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_FRAMES, dropout=dropout * 0.5)
        
        # Transformer encoder (pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.trans_ln = nn.LayerNorm(d_model)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            d_model, d_model // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.lstm_ln = nn.LayerNorm(d_model)
        
        # Multi-query attention pooling
        self.n_queries = 4
        self.pool_queries = nn.Parameter(torch.randn(1, self.n_queries, d_model * 2))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input processing
        x = self.input_norm(x)
        x = self.input_proj1(x)
        x = self.input_proj2(x) + x  # Residual
        x = self.input_ln(x)
        
        # Conv feature extraction
        x = self.conv_subsample(x)
        
        # Add positional encoding
        x_pos = self.pos_encoder(x)
        
        # Transformer branch
        x_trans = self.transformer(x_pos)
        x_trans = self.trans_ln(x_trans)
        
        # LSTM branch (on original projected features)
        x_lstm, _ = self.lstm(x)
        x_lstm = self.lstm_ln(x_lstm)
        
        # Combine branches
        combined = torch.cat([x_trans, x_lstm], dim=-1)  # (batch, seq, d_model*2)
        
        # Multi-query attention pooling
        queries = self.pool_queries.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(queries, combined, combined)
        pooled = self.pool_ln(pooled)
        pooled = pooled.view(batch_size, -1)  # (batch, d_model*2*n_queries)
        
        return self.classifier(pooled)


# ============================================================
# GRAPH PLOTTING
# ============================================================

def plot_training_history(history, save_dir):
    """Plot and save training graphs."""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # 1. Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    if history['swa_epoch']:
        plt.axvline(x=history['swa_epoch'], color='g', linestyle='--', label=f'SWA Start (epoch {history["swa_epoch"]})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=150)
    plt.close()
    
    # 2. Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=150)
    plt.close()
    
    # 3. Learning rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['lr'], 'g-', linewidth=2)
    if history['swa_epoch']:
        plt.axvline(x=history['swa_epoch'], color='r', linestyle='--', label=f'SWA Start')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=150)
    plt.close()
    
    # 4. Combined plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best accuracy text
    axes[1, 1].axis('off')
    best_val = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val) + 1
    final_val = history['val_acc'][-1]
    
    text = f"""
    Training Summary
    ================
    
    Best Validation Accuracy: {best_val:.2f}%
    Best Epoch: {best_epoch}
    
    Final Validation Accuracy: {final_val:.2f}%
    Total Epochs: {len(epochs)}
    
    Classes: {history.get('num_classes', 'N/A')}
    Training Samples: {history.get('train_samples', 'N/A')}
    """
    axes[1, 1].text(0.1, 0.5, text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_summary.png'), dpi=150)
    plt.close()
    
    print(f"\nGraphs saved to: {save_dir}/")
    print(f"  - accuracy.png")
    print(f"  - loss.png")
    print(f"  - learning_rate.png")
    print(f"  - training_summary.png")


def plot_confusion_matrix(y_true, y_pred, classes, save_dir, top_n=20):
    """Plot confusion matrix for worst classes."""
    from sklearn.metrics import confusion_matrix
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Get worst classes
    worst_idx = np.argsort(class_acc)[:top_n]
    worst_classes = [classes[i] for i in worst_idx]
    
    # Subset confusion matrix
    cm_subset = cm[np.ix_(worst_idx, worst_idx)]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_subset, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Worst {top_n} Classes)', fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(worst_classes))
    plt.xticks(tick_marks, worst_classes, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_marks, worst_classes, fontsize=8)
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Per-class accuracy bar chart
    plt.figure(figsize=(14, 8))
    sorted_idx = np.argsort(class_acc)
    sorted_acc = class_acc[sorted_idx]
    sorted_names = [classes[i] for i in sorted_idx]
    
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.7 else 'green' for acc in sorted_acc]
    
    plt.barh(range(len(sorted_acc)), sorted_acc * 100, color=colors)
    plt.yticks(range(len(sorted_acc)), sorted_names, fontsize=6)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=70, color='orange', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=150)
    plt.close()
    
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_accuracy.png")


# ============================================================
# MIXUP / CUTMIX
# ============================================================

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    seq_len = x.size(1)
    cut_len = int(seq_len * (1 - lam))
    cut_start = np.random.randint(0, seq_len - cut_len + 1) if cut_len < seq_len else 0
    
    mixed_x = x.clone()
    mixed_x[:, cut_start:cut_start + cut_len] = x[index, cut_start:cut_start + cut_len]
    
    y_a, y_b = y, y[index]
    lam = 1 - cut_len / seq_len
    return mixed_x, y_a, y_b, lam


# ============================================================
# TRAINING V3
# ============================================================

def train_v3(X, y, classes):
    print("\n" + "=" * 60)
    print("TRAINING V3 - Maximum Performance")
    print("=" * 60)
    
    # Local copy of SWA_START
    swa_start_epoch = SWA_START
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    
    print(f"\nData:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Input shape: {X_train.shape}")
    print(f"  Classes: {len(classes)}")
    
    # Datasets
    train_dataset = ASLDatasetV3(X_train, y_train, augment=True, aug_prob=0.85)
    val_dataset = ASLDatasetV3(X_val, y_val, augment=False)
    
    # Weighted sampler
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[l] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Model
    model = ASLModelV3(
        X_train.shape[2], len(classes),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dim_ff=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:")
    print(f"  Parameters: {total_params:,}")
    print(f"  d_model: {D_MODEL}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Heads: {N_HEADS}")
    
    # SWA model
    swa_model = AveragedModel(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Schedulers
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    
    print(f"\nTraining:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs")
    print(f"  SWA start: epoch {swa_start_epoch}")
    
    print("\n" + "-" * 60)
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    swa_started = False
    
    # History tracking for graphs
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'lr': [],
        'swa_epoch': None,
        'num_classes': len(classes),
        'train_samples': len(X_train)
    }
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct, train_total = 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Augmentation strategy
            aug_choice = np.random.random()
            if aug_choice < 0.25:
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            elif aug_choice < 0.45:
                batch_x, y_a, y_b, lam = cutmix_data(batch_x, batch_y, alpha=1.0)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
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
        
        # Track history
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)
        history['train_loss'].append(train_loss / len(train_loader))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Update schedulers
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        elif epoch >= swa_start_epoch:
            if not swa_started:
                print(f"\n>>> Starting SWA at epoch {epoch+1}")
                swa_started = True
                history['swa_epoch'] = epoch + 1
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            main_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        swa_marker = " [SWA]" if swa_started else ""
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train: {train_acc*100:5.1f}% | Val: {val_acc*100:5.1f}% | "
              f"LR: {lr:.6f}{swa_marker}")
        
        # Save best (before SWA)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f"          ✓ New best: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping (only before SWA)
        if not swa_started and patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}, switching to SWA")
            swa_started = True
            swa_start_epoch = epoch + 1
    
    # Update SWA batch normalization
    if swa_started:
        print("\nUpdating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
        
        # Evaluate SWA model
        swa_model.eval()
        swa_correct, swa_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = swa_model(batch_x)
                _, pred = torch.max(outputs, 1)
                swa_total += batch_y.size(0)
                swa_correct += (pred == batch_y).sum().item()
        
        swa_acc = swa_correct / swa_total
        print(f"SWA Accuracy: {swa_acc*100:.2f}%")
        
        # Use SWA model if better
        if swa_acc > best_acc:
            print("Using SWA model (better than best snapshot)")
            best_acc = swa_acc
            best_state = swa_model.module.state_dict().copy()
    
    # Final results
    print("\n" + "=" * 60)
    print(f"BEST VALIDATION ACCURACY: {best_acc*100:.2f}%")
    print("=" * 60)
    
    # Plot and save graphs
    print("\nGenerating training graphs...")
    plot_training_history(history, GRAPHS_DIR)
    
    # Final evaluation for confusion matrix
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    plot_confusion_matrix(all_labels, all_preds, classes, GRAPHS_DIR)
    
    # Save V3 model
    torch.save({
        'model_state_dict': best_state,
        'input_size': X_train.shape[2],
        'num_classes': len(classes),
        'classes': classes,
        'max_frames': MAX_FRAMES,
        'accuracy': best_acc,
        'd_model': D_MODEL,
        'n_layers': N_LAYERS,
        'version': 'v3'
    }, MODEL_SAVE_PATH)
    
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nSaved: {MODEL_SAVE_PATH}")
    print(f"Saved: {LABEL_ENCODER_PATH}")
    
    return best_acc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ASL SIGNS TRAINING V3")
    print("=" * 60)
    
    X, y, classes = load_dataset()
    
    if X is None:
        exit(1)
    
    acc = train_v3(X, y, classes)
    
    print("\n" + "=" * 60)
    print(f"V3 TRAINING COMPLETE!")
    print(f"Final Accuracy: {acc*100:.2f}%")
    print(f"Model: {MODEL_SAVE_PATH}")
    print("=" * 60)
    print("\nTo use in demo, update paths in demo_voice.py:")
    print('  MODEL_PATH = "asl_signs_model_v3.pth"')
    print('  ENCODER_PATH = "asl_signs_encoder_v3.pkl"')
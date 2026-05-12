"""
Per-Class Accuracy — Separate Plots
=====================================
Generates 5 individual plot files:
  1. per_class_tier_donut.png     — Donut chart showing tier distribution
  2. per_class_histogram.png      — Accuracy distribution histogram
  3. per_class_top20.png          — Top 20 best performing signs
  4. per_class_bottom20.png       — Bottom 20 worst performing signs
  5. per_class_heatmap.png        — All 250 classes in a color-coded grid

Usage:
    python plot_per_class_separate.py

    # With custom paths:
    python plot_per_class_separate.py --model asl_signs_model_v3.pth --cache asl_signs_cache.npz
"""

import os
import sys
import math
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "data/asl_signs_model_v3.pth"
CACHE_FILE = "data/asl_signs_cache.npz"
OUTPUT_DIR = "plots"

# Style
BG_COLOR = '#0a0a0a'
PANEL_BG = '#171717'
TEXT_COLOR = '#e5e5e5'
MUTED_COLOR = '#a3a3a3'
DIM_COLOR = '#525252'
BORDER_COLOR = '#262626'
ACCENT_PINK = '#f472b6'

TIERS = [
    ('Excellent (≥90%)', 90, 100, '#22c55e'),
    ('Good (75-90%)',    75, 90,  '#84cc16'),
    ('Needs Work (50-75%)', 50, 75, '#f59e0b'),
    ('Failing (<50%)',   0,  50,  '#ef4444'),
]


def get_tier_color(acc):
    for _, lo, hi, color in TIERS:
        if lo == 90 and acc >= 90:
            return color
        elif acc >= lo and acc < hi:
            return color
    return '#ef4444'


# ============================================================
# MODEL ARCHITECTURE (must match V3 training code)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
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
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + residual)


class ASLModelV3(nn.Module):
    def __init__(self, input_size=225, num_classes=250, d_model=384,
                 n_heads=8, n_layers=6, dim_ff=1536, dropout=0.0, max_frames=64):
        super().__init__()
        self.d_model = d_model
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj1 = nn.Linear(input_size, d_model)
        self.input_proj2 = nn.Sequential(nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(d_model, d_model))
        self.input_ln = nn.LayerNorm(d_model)
        self.conv_subsample = ConvSubsampling(d_model, dropout * 0.5)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames, dropout=dropout * 0.5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
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
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_norm(x)
        x = self.input_proj1(x)
        x = self.input_proj2(x) + x
        x = self.input_ln(x)
        x = self.conv_subsample(x)
        x_pos = self.pos_encoder(x)
        x_trans = self.trans_ln(self.transformer(x_pos))
        x_lstm, _ = self.lstm(x)
        x_lstm = self.lstm_ln(x_lstm)
        combined = torch.cat([x_trans, x_lstm], dim=-1)
        queries = self.pool_queries.expand(B, -1, -1)
        pooled, _ = self.pool_attention(queries, combined, combined)
        pooled = self.pool_ln(pooled).view(B, -1)
        return self.classifier(pooled)


# ============================================================
# DATA + INFERENCE
# ============================================================

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def compute_per_class_accuracy(args):
    """Load model, run on validation set, return per-class accuracies."""

    print("Loading cache...")
    if not os.path.exists(args.cache):
        print(f"ERROR: Cache not found: {args.cache}")
        sys.exit(1)

    cache = np.load(args.cache, allow_pickle=True)
    X, y = cache['X'], cache['y']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    _, X_val, _, y_val = train_test_split(
        X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    print(f"Validation: {len(X_val)} samples, {num_classes} classes")

    # Load model
    print("Loading model...")
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)

    input_size = checkpoint.get('input_size', 225)
    max_frames = checkpoint.get('max_frames', 64)
    d_model = checkpoint.get('d_model', 384)
    n_layers = checkpoint.get('n_layers', 6)
    stored_acc = checkpoint.get('accuracy', 0)
    print(f"Stored accuracy: {stored_acc*100:.2f}%")

    model = ASLModelV3(
        input_size=input_size, num_classes=num_classes,
        d_model=d_model, n_heads=8, n_layers=n_layers,
        dim_ff=1536, dropout=0.0, max_frames=max_frames
    )

    state = checkpoint['model_state_dict']
    clean = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Running inference on {device}...")

    val_ds = SimpleDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    total_correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)

            total_correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

            for i in range(batch_y.size(0)):
                label = batch_y[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    overall_acc = total_correct / total
    print(f"Computed accuracy: {overall_acc*100:.2f}%")

    # Per-class accuracy
    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = (class_correct[i] / class_total[i]) * 100
        else:
            per_class_acc[i] = 0

    return le.classes_, per_class_acc, class_total, overall_acc


# ============================================================
# PLOT 1: TIER DONUT
# ============================================================

def plot_tier_donut(names, accs, overall_acc, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    tier_counts = []
    tier_labels = []
    tier_colors = []

    for label, lo, hi, color in TIERS:
        if lo == 90:
            count = int(np.sum(accs >= 90))
        else:
            count = int(np.sum((accs >= lo) & (accs < hi)))
        tier_counts.append(count)
        tier_labels.append(f"{label}\n({count} signs)")
        tier_colors.append(color)

    wedges, texts, autotexts = ax.pie(
        tier_counts, labels=tier_labels, colors=tier_colors,
        autopct='%1.1f%%', pctdistance=0.78, startangle=90,
        textprops={'color': TEXT_COLOR, 'fontsize': 11, 'fontweight': '500'},
        wedgeprops={'width': 0.35, 'edgecolor': BG_COLOR, 'linewidth': 3}
    )
    for t in autotexts:
        t.set_color('#000000')
        t.set_fontweight('bold')
        t.set_fontsize(10)

    ax.text(0, 0.08, f'{overall_acc*100:.1f}%', ha='center', va='center',
            fontsize=36, fontweight='bold', color=ACCENT_PINK)
    ax.text(0, -0.15, 'overall accuracy', ha='center', va='center',
            fontsize=12, color=MUTED_COLOR)

    ax.set_title('Accuracy Distribution by Tier', color=TEXT_COLOR,
                 fontsize=18, fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 2: HISTOGRAM
# ============================================================

def plot_histogram(accs, overall_acc, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_BG)

    bins = np.arange(0, 105, 5)
    n, bin_edges, patches = ax.hist(accs, bins=bins, edgecolor=BG_COLOR, linewidth=1)

    for patch, left in zip(patches, bin_edges[:-1]):
        patch.set_facecolor(get_tier_color(left))

    mean_val = np.mean(accs)
    median_val = np.median(accs)

    ax.axvline(mean_val, color='#c084fc', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_val:.1f}%', zorder=5)
    ax.axvline(median_val, color='#818cf8', linestyle=':', linewidth=2.5,
               label=f'Median: {median_val:.1f}%', zorder=5)

    ax.set_xlabel('Accuracy (%)', color=MUTED_COLOR, fontsize=13, labelpad=10)
    ax.set_ylabel('Number of Classes', color=MUTED_COLOR, fontsize=13, labelpad=10)
    ax.set_title('Per-Class Accuracy Distribution', color=TEXT_COLOR,
                 fontsize=18, fontweight='bold', pad=20)

    legend = ax.legend(loc='upper left', facecolor='#262626', edgecolor='#404040',
                       labelcolor=TEXT_COLOR, fontsize=11, framealpha=0.9)

    # Stats box
    stats_text = (f'Classes: {len(accs)}\n'
                  f'Std Dev: {np.std(accs):.1f}%\n'
                  f'Min: {np.min(accs):.1f}%\n'
                  f'Max: {np.max(accs):.1f}%')
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, color=MUTED_COLOR, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#262626',
                      edgecolor='#404040', alpha=0.9))

    ax.tick_params(colors=DIM_COLOR, labelsize=10)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-2, 102)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 3: TOP 20
# ============================================================

def plot_top20(names_sorted, accs_sorted, counts_sorted, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_BG)

    n = 20
    top_names = names_sorted[:n][::-1]
    top_accs = accs_sorted[:n][::-1]
    top_counts = counts_sorted[:n][::-1]

    colors = [get_tier_color(a) for a in top_accs]
    bars = ax.barh(range(n), top_accs, color=colors, height=0.7,
                   edgecolor=BG_COLOR, linewidth=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(top_names, fontsize=11, color=TEXT_COLOR, fontweight='500')
    ax.set_xlim(0, 108)
    ax.set_xlabel('Accuracy (%)', color=MUTED_COLOR, fontsize=13, labelpad=10)
    ax.set_title('Top 20 Best Performing Signs', color='#22c55e',
                 fontsize=18, fontweight='bold', pad=20)

    for i, (acc, count) in enumerate(zip(top_accs, top_counts)):
        ax.text(acc + 1, i, f'{acc:.1f}%  ({int(count)} samples)',
                va='center', fontsize=9, color=MUTED_COLOR)

    ax.tick_params(colors=DIM_COLOR, labelsize=10)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 4: BOTTOM 20
# ============================================================

def plot_bottom20(names_sorted, accs_sorted, counts_sorted, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_BG)

    n = 20
    bot_names = names_sorted[-n:]
    bot_accs = accs_sorted[-n:]
    bot_counts = counts_sorted[-n:]

    colors = [get_tier_color(a) for a in bot_accs]
    bars = ax.barh(range(n), bot_accs, color=colors, height=0.7,
                   edgecolor=BG_COLOR, linewidth=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(bot_names, fontsize=11, color=TEXT_COLOR, fontweight='500')
    ax.set_xlim(0, 108)
    ax.set_xlabel('Accuracy (%)', color=MUTED_COLOR, fontsize=13, labelpad=10)
    ax.set_title('Bottom 20 Worst Performing Signs', color='#ef4444',
                 fontsize=18, fontweight='bold', pad=20)

    for i, (acc, count) in enumerate(zip(bot_accs, bot_counts)):
        ax.text(max(acc + 1, 3), i, f'{acc:.1f}%  ({int(count)} samples)',
                va='center', fontsize=9, color=MUTED_COLOR)

    ax.tick_params(colors=DIM_COLOR, labelsize=10)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 5: HEATMAP GRID (all 250 classes)
# ============================================================

def plot_heatmap(names_sorted, accs_sorted, save_path):
    n_cols = 10
    n_rows = int(np.ceil(len(names_sorted) / n_cols))

    fig, ax = plt.subplots(figsize=(20, n_rows * 0.5 + 3))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for i, (name, acc) in enumerate(zip(names_sorted, accs_sorted)):
        row = i // n_cols
        col = i % n_cols

        color = get_tier_color(acc)
        alpha = max(0.35, acc / 100)

        rect = plt.Rectangle((col * 1.05, n_rows - 1 - row), 1.0, 0.88,
                              facecolor=color, alpha=alpha,
                              edgecolor='#1a1a1a', linewidth=0.5)
        ax.add_patch(rect)

        # Name (truncate if long)
        label = name[:13] if len(name) > 13 else name
        ax.text(col * 1.05 + 0.5, n_rows - 1 - row + 0.58, label,
                ha='center', va='center', fontsize=5.5, color='#ffffff',
                fontweight='bold')
        ax.text(col * 1.05 + 0.5, n_rows - 1 - row + 0.25, f'{acc:.0f}%',
                ha='center', va='center', fontsize=5, color='#ffffff', alpha=0.75)

    ax.set_xlim(-0.2, n_cols * 1.05 + 0.2)
    ax.set_ylim(-1.2, n_rows + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('All Classes — Sorted by Accuracy (left → right, top → bottom)',
                 color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=25)

    # Legend at bottom
    legend_y = -0.6
    for i, (label, lo, hi, color) in enumerate(TIERS):
        x_pos = i * 2.7
        ax.add_patch(plt.Rectangle((x_pos, legend_y), 0.35, 0.35,
                                    facecolor=color, edgecolor='none'))
        ax.text(x_pos + 0.5, legend_y + 0.17, label,
                fontsize=9, color=MUTED_COLOR, va='center')

    plt.savefig(save_path, dpi=200, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# TEXT SUMMARY
# ============================================================

def print_summary(names_sorted, accs_sorted, counts_sorted, overall_acc):
    print(f"\n{'='*65}")
    print(f"  PER-CLASS ACCURACY SUMMARY")
    print(f"{'='*65}")
    print(f"  Overall accuracy:  {overall_acc*100:.2f}%")
    print(f"  Total classes:     {len(accs_sorted)}")
    print(f"  Mean per-class:    {np.mean(accs_sorted):.1f}%")
    print(f"  Median per-class:  {np.median(accs_sorted):.1f}%")
    print(f"  Std deviation:     {np.std(accs_sorted):.1f}%")
    print(f"  Best:  {accs_sorted[0]:6.1f}%  ({names_sorted[0]})")
    print(f"  Worst: {accs_sorted[-1]:6.1f}%  ({names_sorted[-1]})")
    print()

    for label, lo, hi, _ in TIERS:
        if lo == 90:
            count = int(np.sum(accs_sorted >= 90))
        else:
            count = int(np.sum((accs_sorted >= lo) & (accs_sorted < hi)))
        pct = count / len(accs_sorted) * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:25s}  {count:3d} signs ({pct:4.1f}%)  {bar}")

    print(f"\n  {'─'*60}")
    print(f"  TOP 10:")
    for i in range(min(10, len(names_sorted))):
        print(f"    {i+1:2d}. {names_sorted[i]:20s}  {accs_sorted[i]:5.1f}%  ({int(counts_sorted[i])} samples)")

    print(f"\n  BOTTOM 10:")
    for i in range(min(10, len(names_sorted))):
        idx = -(i + 1)
        print(f"    {len(names_sorted)+idx+1:3d}. {names_sorted[idx]:20s}  {accs_sorted[idx]:5.1f}%  ({int(counts_sorted[idx])} samples)")

    print(f"{'='*65}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-class accuracy plots (separate files)")
    parser.add_argument('--model', default=MODEL_PATH, help='Path to .pth checkpoint')
    parser.add_argument('--cache', default=CACHE_FILE, help='Path to .npz cache')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory for plots')
    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    # Compute per-class accuracy
    class_names, per_class_acc, class_total, overall_acc = compute_per_class_accuracy(args)

    # Sort by accuracy (descending)
    sorted_idx = np.argsort(per_class_acc)[::-1]
    names_sorted = [class_names[i] for i in sorted_idx]
    accs_sorted = per_class_acc[sorted_idx]
    counts_sorted = class_total[sorted_idx]

    # Generate all plots
    print("\nGenerating plots...")

    plot_tier_donut(
        names_sorted, accs_sorted, overall_acc,
        os.path.join(args.output, 'per_class_tier_donut.png')
    )

    plot_histogram(
        accs_sorted, overall_acc,
        os.path.join(args.output, 'per_class_histogram.png')
    )

    plot_top20(
        names_sorted, accs_sorted, counts_sorted,
        os.path.join(args.output, 'per_class_top20.png')
    )

    plot_bottom20(
        names_sorted, accs_sorted, counts_sorted,
        os.path.join(args.output, 'per_class_bottom20.png')
    )

    plot_heatmap(
        names_sorted, accs_sorted,
        os.path.join(args.output, 'per_class_heatmap.png')
    )

    # Print text summary
    print_summary(names_sorted, accs_sorted, counts_sorted, overall_acc)

    print(f"All plots saved to: {args.output}/")
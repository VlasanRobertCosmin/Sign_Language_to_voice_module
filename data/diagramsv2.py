"""
Per-Class Accuracy — Best, Middle, and Worst Classes
====================================================

Generates one clean horizontal bar chart containing:
    - 10 best-performing classes
    - 10 middle-performing classes
    - 10 worst-performing classes

Output:
    plots/per_class_accuracy_best_middle_worst.png

Usage:
    python plot_per_class_best_middle_worst.py

With custom paths:
    python plot_per_class_best_middle_worst.py --model data/asl_signs_model_v3.pth --cache data/asl_signs_cache.npz
"""

import os
import sys
import math
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

TIERS = [
    ("Excellent (≥90%)", 90, 100, "#22c55e"),
    ("Good (75–90%)", 75, 90, "#84cc16"),
    ("Needs Work (50–75%)", 50, 75, "#f59e0b"),
    ("Failing (<50%)", 0, 50, "#ef4444"),
]


def get_tier_color(acc):
    """Return color based on accuracy tier."""
    for _, lo, hi, color in TIERS:
        if lo == 90 and acc >= 90:
            return color
        elif lo <= acc < hi:
            return color
    return "#ef4444"


# ============================================================
# MODEL ARCHITECTURE
# Must match the architecture used during V3 training.
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ConvSubsampling(nn.Module):
    def __init__(self, d_model, dropout=0.0):
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
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + residual)


class ASLModelV3(nn.Module):
    def __init__(
        self,
        input_size=225,
        num_classes=250,
        d_model=384,
        n_heads=8,
        n_layers=6,
        dim_ff=1536,
        dropout=0.0,
        max_frames=64
    ):
        super().__init__()

        self.d_model = d_model

        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj1 = nn.Linear(input_size, d_model)

        self.input_proj2 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model)
        )

        self.input_ln = nn.LayerNorm(d_model)
        self.conv_subsample = ConvSubsampling(d_model, dropout * 0.5)

        self.pos_encoder = PositionalEncoding(
            d_model,
            max_len=max_frames,
            dropout=dropout * 0.5
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.trans_ln = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(
            d_model,
            d_model // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.lstm_ln = nn.LayerNorm(d_model)

        self.n_queries = 4

        self.pool_queries = nn.Parameter(
            torch.randn(1, self.n_queries, d_model * 2)
        )

        self.pool_attention = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
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
        x_trans = self.trans_ln(self.transformer(x_pos))

        x_lstm, _ = self.lstm(x)
        x_lstm = self.lstm_ln(x_lstm)

        combined = torch.cat([x_trans, x_lstm], dim=-1)

        queries = self.pool_queries.expand(batch_size, -1, -1)

        pooled, _ = self.pool_attention(
            queries,
            combined,
            combined
        )

        pooled = self.pool_ln(pooled).view(batch_size, -1)

        return self.classifier(pooled)


# ============================================================
# DATASET
# ============================================================

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# DATA + INFERENCE
# ============================================================

def compute_per_class_accuracy(args):
    """
    Load the cached dataset and trained model.
    Run inference on the validation split.
    Return class names, per-class accuracy, class counts, and overall accuracy.
    """

    print("Loading cache...")

    if not os.path.exists(args.cache):
        print(f"ERROR: Cache not found: {args.cache}")
        sys.exit(1)

    cache = np.load(args.cache, allow_pickle=True)

    X = cache["X"]
    y = cache["y"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    num_classes = len(le.classes_)

    _, X_val, _, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.1,
        random_state=42,
        stratify=y_encoded
    )

    print(f"Validation samples: {len(X_val)}")
    print(f"Number of classes: {num_classes}")

    print("Loading model...")

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    checkpoint = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False
    )

    input_size = checkpoint.get("input_size", 225)
    max_frames = checkpoint.get("max_frames", 64)
    d_model = checkpoint.get("d_model", 384)
    n_layers = checkpoint.get("n_layers", 6)
    stored_acc = checkpoint.get("accuracy", 0)

    print(f"Stored checkpoint accuracy: {stored_acc * 100:.2f}%")

    model = ASLModelV3(
        input_size=input_size,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=8,
        n_layers=n_layers,
        dim_ff=1536,
        dropout=0.0,
        max_frames=max_frames
    )

    state = checkpoint["model_state_dict"]

    clean_state = {
        k.replace("module.", ""): v
        for k, v in state.items()
    }

    model.load_state_dict(clean_state, strict=True)
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print(f"Running inference on: {device}")

    val_ds = SimpleDataset(X_val, y_val)

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

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

    print(f"Computed validation accuracy: {overall_acc * 100:.2f}%")

    per_class_acc = np.zeros(num_classes)

    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = (
                class_correct[i] / class_total[i]
            ) * 100
        else:
            per_class_acc[i] = 0

    return le.classes_, per_class_acc, class_total, overall_acc


# ============================================================
# SELECT BEST, MIDDLE, AND WORST CLASSES
# ============================================================

def select_best_middle_worst(class_names, accs, counts, n=10):
    """
    Select n best, n middle, and n worst classes based on accuracy.
    """

    sorted_idx = np.argsort(accs)[::-1]

    best_idx = sorted_idx[:n]
    worst_idx = sorted_idx[-n:]

    middle_start = max(0, (len(sorted_idx) // 2) - (n // 2))
    middle_idx = sorted_idx[middle_start:middle_start + n]

    selected = []

    for idx in best_idx:
        selected.append(("Best 10", class_names[idx], accs[idx], counts[idx]))

    for idx in middle_idx:
        selected.append(("Middle 10", class_names[idx], accs[idx], counts[idx]))

    for idx in worst_idx:
        selected.append(("Worst 10", class_names[idx], accs[idx], counts[idx]))

    return selected


# ============================================================
# SINGLE PAPER-FRIENDLY PLOT
# ============================================================

def plot_best_middle_worst(selected, overall_acc, save_path):
    """
    Plot one clean chart with best, middle, and worst classes.
    """

    groups = [item[0] for item in selected]
    names = [item[1] for item in selected]
    accs = np.array([item[2] for item in selected])
    counts = [item[3] for item in selected]

    labels = [
        f"{name} ({int(count)} samples)"
        for name, count in zip(names, counts)
    ]

    colors = [
        get_tier_color(acc)
        for acc in accs
    ]

    y_positions = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(
        y_positions,
        accs,
        color=colors,
        height=0.72,
        edgecolor="white",
        linewidth=0.5
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)

    ax.invert_yaxis()

    ax.set_xlim(0, 105)

    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Per-Class Accuracy Summary: Best, Middle, and Worst Classes\n"
        f"Overall Accuracy: {overall_acc * 100:.2f}%",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    # Group separators
    ax.axhline(9.5, color="#999999", linewidth=1)
    ax.axhline(19.5, color="#999999", linewidth=1)

    # Group labels
    ax.text(
        103,
        4.5,
        "Best 10",
        va="center",
        ha="right",
        fontsize=11,
        fontweight="bold",
        color="#15803d"
    )

    ax.text(
        103,
        14.5,
        "Middle 10",
        va="center",
        ha="right",
        fontsize=11,
        fontweight="bold",
        color="#b45309"
    )

    ax.text(
        103,
        24.5,
        "Worst 10",
        va="center",
        ha="right",
        fontsize=11,
        fontweight="bold",
        color="#b91c1c"
    )

    # Threshold lines
    ax.axvline(
        50,
        color="#ef4444",
        linestyle="--",
        linewidth=1.3,
        label="50% threshold"
    )

    ax.axvline(
        75,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.3,
        label="75% threshold"
    )

    ax.axvline(
        90,
        color="#22c55e",
        linestyle="--",
        linewidth=1.3,
        label="90% threshold"
    )

    # Accuracy labels at bar ends
    for i, acc in enumerate(accs):
        ax.text(
            min(acc + 1.2, 101),
            i,
            f"{acc:.1f}%",
            va="center",
            fontsize=9
        )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        fontsize=9,
        frameon=True
    )

    caption = (
        "Green indicates high accuracy, orange indicates moderate accuracy, "
        "and red indicates low accuracy."
    )

    fig.text(
        0.5,
        0.01,
        caption,
        ha="center",
        fontsize=10,
        color="#444444"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved paper-friendly plot: {save_path}")


# ============================================================
# TEXT SUMMARY
# ============================================================

def print_summary(class_names, accs, counts, overall_acc):
    sorted_idx = np.argsort(accs)[::-1]

    print(f"\n{'=' * 65}")
    print("  PER-CLASS ACCURACY SUMMARY")
    print(f"{'=' * 65}")

    print(f"  Overall accuracy:  {overall_acc * 100:.2f}%")
    print(f"  Total classes:     {len(accs)}")
    print(f"  Mean per-class:    {np.mean(accs):.1f}%")
    print(f"  Median per-class:  {np.median(accs):.1f}%")
    print(f"  Std deviation:     {np.std(accs):.1f}%")

    best = sorted_idx[0]
    worst = sorted_idx[-1]

    print(f"  Best class:        {class_names[best]} ({accs[best]:.1f}%)")
    print(f"  Worst class:       {class_names[worst]} ({accs[worst]:.1f}%)")

    print(f"\n  {'─' * 60}")
    print("  PERFORMANCE TIERS:")

    for label, lo, hi, _ in TIERS:
        if lo == 90:
            count = int(np.sum(accs >= 90))
        else:
            count = int(
                np.sum(
                    (accs >= lo)
                    & (accs < hi)
                )
            )

        pct = count / len(accs) * 100

        print(
            f"  {label:25s}  "
            f"{count:3d} classes "
            f"({pct:4.1f}%)"
        )

    print(f"{'=' * 65}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate one paper-friendly per-class accuracy plot"
    )

    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to .pth checkpoint"
    )

    parser.add_argument(
        "--cache",
        default=CACHE_FILE,
        help="Path to .npz cache"
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help="Output directory for plot"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    class_names, per_class_acc, class_total, overall_acc = (
        compute_per_class_accuracy(args)
    )

    selected = select_best_middle_worst(
        class_names,
        per_class_acc,
        class_total,
        n=10
    )

    output_path = os.path.join(
        args.output,
        "per_class_accuracy_best_middle_worst.png"
    )

    print("\nGenerating paper-friendly plot...")

    plot_best_middle_worst(
        selected,
        overall_acc,
        output_path
    )

    print_summary(
        class_names,
        per_class_acc,
        class_total,
        overall_acc
    )

    print(f"Done. Plot saved to: {output_path}")
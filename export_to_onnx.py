"""
Step 1: Export ASL V3 Model to ONNX for Meta Quest / Unity Inference Engine
============================================================================

This script:
1. Loads your trained PyTorch ASLModelV3 checkpoint
2. Exports it to ONNX format
3. Validates the ONNX model produces identical outputs
4. Provides landmark remapping info for Quest hand tracking

Usage:
    python step1_export_to_onnx.py

Requirements:
    pip install torch onnx onnxruntime numpy

After export:
    - Import the .onnx file into Unity
    - Use Meta Tools > Unity Inference Engine > ONNX > Sentis Converter
    - Quantize to Float16 for Quest performance
"""

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# CONFIG — Update these paths to match your setup
# ============================================================
MODEL_PATH = "asl_signs_model_v3.pth"          # Your trained V3 checkpoint
ENCODER_PATH = "asl_signs_encoder_v3.pkl"       # Your label encoder
ONNX_OUTPUT_PATH = "asl_model_v3.onnx"          # Output ONNX file
ONNX_OPTIMIZED_PATH = "asl_model_v3_opt.onnx"   # Optimized ONNX (optional)

# Model architecture params (must match training)
INPUT_SIZE = 225        # 75 landmarks × 3 coordinates (x, y, z)
MAX_FRAMES = 64         # Sequence length
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 6
DIM_FF = 1536
DROPOUT = 0.0           # Set to 0 for export (inference mode)


# ============================================================
# MODEL ARCHITECTURE (must match your training code exactly)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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
        x = x.transpose(1, 2)   # (B, d_model, seq)
        x = self.conv(x)
        x = x.transpose(1, 2)   # (B, seq, d_model)
        return self.norm(x + residual)


class ASLModelV3(nn.Module):
    """V3 Enhanced Hybrid Transformer-LSTM — must match training code."""

    def __init__(self, input_size=225, num_classes=250, d_model=384,
                 n_heads=8, n_layers=6, dim_ff=1536, dropout=0.0,
                 max_frames=64):
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

        # Convolutional subsampling
        self.conv_subsample = ConvSubsampling(d_model, dropout * 0.5)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_frames, dropout=dropout * 0.5
        )

        # Transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.trans_ln = nn.LayerNorm(d_model)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.lstm_ln = nn.LayerNorm(d_model)

        # Multi-query attention pooling
        self.n_queries = 4
        self.pool_queries = nn.Parameter(
            torch.randn(1, self.n_queries, d_model * 2)
        )
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
        batch_size = x.size(0)

        # Input projection
        x = self.input_norm(x)
        x = self.input_proj1(x)
        x = self.input_proj2(x) + x  # residual
        x = self.input_ln(x)

        # Conv subsampling
        x = self.conv_subsample(x)
        x_pos = self.pos_encoder(x)

        # Transformer branch
        x_trans = self.transformer(x_pos)
        x_trans = self.trans_ln(x_trans)

        # LSTM branch (uses pre-positional-encoding features)
        x_lstm, _ = self.lstm(x)
        x_lstm = self.lstm_ln(x_lstm)

        # Concatenate branches
        combined = torch.cat([x_trans, x_lstm], dim=-1)  # (B, seq, d*2)

        # Multi-query attention pooling
        queries = self.pool_queries.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(queries, combined, combined)
        pooled = self.pool_ln(pooled)
        pooled = pooled.view(batch_size, -1)  # (B, d*2*n_queries)

        return self.classifier(pooled)


# ============================================================
# STEP 1: Load trained checkpoint
# ============================================================

def load_checkpoint():
    """Load the trained V3 model and label encoder."""
    print("=" * 60)
    print("STEP 1: Loading trained model checkpoint")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model file not found: {MODEL_PATH}")
        print("Make sure this script is in the same directory as your model.")
        print("Update MODEL_PATH at the top of this file if needed.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    # Extract model info
    num_classes = checkpoint.get('num_classes', 250)
    input_size = checkpoint.get('input_size', INPUT_SIZE)
    max_frames = checkpoint.get('max_frames', MAX_FRAMES)
    d_model = checkpoint.get('d_model', D_MODEL)
    n_layers = checkpoint.get('n_layers', N_LAYERS)
    accuracy = checkpoint.get('accuracy', 0)

    print(f"  Classes:      {num_classes}")
    print(f"  Input size:   {input_size}")
    print(f"  Max frames:   {max_frames}")
    print(f"  d_model:      {d_model}")
    print(f"  Layers:       {n_layers}")
    if accuracy:
        print(f"  Accuracy:     {accuracy*100:.1f}%")

    # Build model with dropout=0 for inference
    model = ASLModelV3(
        input_size=input_size,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=N_HEADS,
        n_layers=n_layers,
        dim_ff=DIM_FF,
        dropout=0.0,        # No dropout for inference
        max_frames=max_frames,
    )

    # Load weights (handle SWA state dicts)
    state_dict = checkpoint['model_state_dict']

    # If SWA was used, weights might be wrapped
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix from DataParallel if present
        key = k.replace('module.', '')
        new_state_dict[key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    print(f"  Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print("  Model loaded successfully!")

    # Load label encoder
    classes = None
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        classes = encoder.classes_.tolist()
        print(f"  Labels loaded: {len(classes)} classes")
    else:
        print(f"  WARNING: Label encoder not found at {ENCODER_PATH}")
        print("  You'll need the class labels later for Unity inference.")

    return model, num_classes, input_size, max_frames, classes


# ============================================================
# STEP 2: Export to ONNX
# ============================================================

def export_to_onnx(model, input_size, max_frames):
    """Export PyTorch model to ONNX format."""
    print("\n" + "=" * 60)
    print("STEP 2: Exporting to ONNX")
    print("=" * 60)

    # Create dummy input matching your data shape
    # Shape: (batch_size=1, sequence_length=64, features=225)
    dummy_input = torch.randn(1, max_frames, input_size)

    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output path:  {ONNX_OUTPUT_PATH}")

    # Test forward pass first
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")

    # Export to ONNX using the legacy TorchScript exporter
    # (avoids onnxscript/InlinePass errors in newer PyTorch versions)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=14,           # opset 14 — stable, widely supported
        do_constant_folding=True,    # Optimize constant folding
        input_names=['landmarks'],   # Name the input
        output_names=['logits'],     # Name the output
        dynamic_axes={
            'landmarks': {0: 'batch_size'},  # Allow variable batch size
            'logits': {0: 'batch_size'},
        },
        dynamo=False,                # Force legacy exporter (no onnxscript)
    )

    # Check file size
    file_size_mb = os.path.getsize(ONNX_OUTPUT_PATH) / (1024 * 1024)
    print(f"\n  ONNX file size: {file_size_mb:.1f} MB")
    print("  Export complete!")

    return dummy_input, test_output


# ============================================================
# STEP 3: Validate ONNX model
# ============================================================

def validate_onnx(dummy_input, pytorch_output):
    """Verify ONNX model produces identical outputs to PyTorch."""
    print("\n" + "=" * 60)
    print("STEP 3: Validating ONNX model")
    print("=" * 60)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  Install onnx and onnxruntime for validation:")
        print("  pip install onnx onnxruntime")
        print("  Skipping validation (export was still successful).")
        return

    # Load and check ONNX model
    onnx_model = onnx.load(ONNX_OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure: VALID")

    # Print model info
    print(f"\n  Graph inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")

    print(f"\n  Graph outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    # Run inference with ONNX Runtime
    session = ort.InferenceSession(ONNX_OUTPUT_PATH)
    ort_inputs = {'landmarks': dummy_input.numpy()}
    ort_output = session.run(None, ort_inputs)[0]

    # Compare outputs
    pytorch_np = pytorch_output.detach().numpy()
    max_diff = np.max(np.abs(pytorch_np - ort_output))
    mean_diff = np.mean(np.abs(pytorch_np - ort_output))

    print(f"\n  PyTorch vs ONNX comparison:")
    print(f"    Max absolute difference:  {max_diff:.8f}")
    print(f"    Mean absolute difference: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("    Status: PERFECT MATCH")
    elif max_diff < 1e-2:
        print("    Status: ACCEPTABLE (small numerical differences)")
    else:
        print("    WARNING: Large differences detected!")
        print("    Check model architecture matches exactly.")

    # Test with multiple random inputs
    print("\n  Running 10 random input tests...")
    all_pass = True
    for i in range(10):
        rand_input = torch.randn(1, MAX_FRAMES, INPUT_SIZE)
        with torch.no_grad():
            pt_out = model(rand_input).numpy()
        ort_out = session.run(None, {'landmarks': rand_input.numpy()})[0]
        diff = np.max(np.abs(pt_out - ort_out))
        if diff > 1e-2:
            print(f"    Test {i+1}: FAIL (diff={diff:.6f})")
            all_pass = False

    if all_pass:
        print("    All 10 tests: PASSED")


# ============================================================
# STEP 4: Save class labels for Unity
# ============================================================

def save_labels_for_unity(classes):
    """Save class labels as a simple text file for Unity to load."""
    print("\n" + "=" * 60)
    print("STEP 4: Saving class labels for Unity")
    print("=" * 60)

    if classes is None:
        print("  No class labels available. Skipping.")
        return

    labels_path = "asl_class_labels.txt"
    with open(labels_path, 'w') as f:
        for label in classes:
            f.write(f"{label}\n")

    print(f"  Saved {len(classes)} labels to {labels_path}")
    print(f"  First 10: {classes[:10]}")
    print(f"  Last 5:   {classes[-5:]}")

    # Also save as JSON for easier Unity parsing
    import json
    json_path = "asl_class_labels.json"
    with open(json_path, 'w') as f:
        json.dump({"classes": classes, "num_classes": len(classes)}, f, indent=2)
    print(f"  Also saved as JSON: {json_path}")


# ============================================================
# STEP 5: Print Quest landmark mapping guide
# ============================================================

def print_landmark_mapping():
    """Print the mapping between Quest hand bones and MediaPipe landmarks."""
    print("\n" + "=" * 60)
    print("STEP 5: Quest Hand Tracking → MediaPipe Landmark Mapping")
    print("=" * 60)

    print("""
Your model was trained on MediaPipe landmarks with this layout:
  - Left hand:  21 landmarks × 3 coords = features [0:63]
  - Right hand: 21 landmarks × 3 coords = features [63:126]
  - Pose:       33 landmarks × 3 coords = features [126:225]

Meta Quest provides 24 hand bones per hand (OVR skeleton).
You need to MAP Quest bones → MediaPipe landmark indices.

LANDMARK MAPPING TABLE:
=======================

MediaPipe Index | MediaPipe Name       | Quest OVR Bone ID
----------------|----------------------|----------------------------------
 0              | WRIST                | Hand_WristRoot (0)
 1              | THUMB_CMC            | Hand_Thumb0 (2)
 2              | THUMB_MCP            | Hand_Thumb1 (3)
 3              | THUMB_IP             | Hand_Thumb2 (4)
 4              | THUMB_TIP            | Hand_Thumb3 (5) — tip position
 5              | INDEX_MCP            | Hand_Index1 (6)
 6              | INDEX_PIP            | Hand_Index2 (7)
 7              | INDEX_DIP            | Hand_Index3 (8)
 8              | INDEX_TIP            | Compute: extend from Index3
 9              | MIDDLE_MCP           | Hand_Middle1 (9)
10              | MIDDLE_PIP           | Hand_Middle2 (10)
11              | MIDDLE_DIP           | Hand_Middle3 (11)
12              | MIDDLE_TIP           | Compute: extend from Middle3
13              | RING_MCP             | Hand_Ring1 (12)
14              | RING_PIP             | Hand_Ring2 (13)
15              | RING_DIP             | Hand_Ring3 (14)
16              | RING_TIP             | Compute: extend from Ring3
17              | PINKY_MCP            | Hand_Pinky0 (15)
18              | PINKY_PIP            | Hand_Pinky1 (16)
19              | PINKY_DIP            | Hand_Pinky2 (17)
20              | PINKY_TIP            | Compute: extend from Pinky2

NOTE: Quest does NOT provide fingertip positions directly.
      Fingertips (4, 8, 12, 16, 20) must be ESTIMATED by
      extending the direction from DIP to the last bone by
      ~1.5cm along the bone direction vector.

POSE LANDMARKS (features 126-225):
==================================
Quest provides head tracking (HMD position/rotation) but NOT
full body pose. For the pose landmarks:
  - Landmarks 0 (nose) → approximate from HMD position
  - Landmarks 11-12 (shoulders) → approximate from HMD + offset
  - Landmarks 13-16 (elbows, wrists) → from controller/hand positions
  - All other pose landmarks → set to 0 (same as training zero-padding)

This is acceptable because the model was trained on data where
many pose landmarks were often NaN/zero-padded already.
""")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ASL V3 MODEL → ONNX EXPORT FOR META QUEST VR")
    print("=" * 60)

    # Step 1: Load model
    model, num_classes, input_size, max_frames, classes = load_checkpoint()

    # Step 2: Export to ONNX
    dummy_input, pytorch_output = export_to_onnx(model, input_size, max_frames)

    # Step 3: Validate
    validate_onnx(dummy_input, pytorch_output)

    # Step 4: Save labels
    save_labels_for_unity(classes)

    # Step 5: Print mapping guide
    print_landmark_mapping()

    # Summary
    print("\n" + "=" * 60)
    print("  EXPORT COMPLETE — FILES TO BRING INTO UNITY:")
    print("=" * 60)
    print(f"""
    1. {ONNX_OUTPUT_PATH}     — The ONNX model file
    2. asl_class_labels.txt    — Class labels (one per line)
    3. asl_class_labels.json   — Class labels (JSON format)

    NEXT STEPS:
    -----------
    1. Open Unity with Meta XR SDK installed
    2. Drag {ONNX_OUTPUT_PATH} into Assets/Models/
    3. Go to Meta > Tools > Unity Inference Engine > ONNX > Sentis Converter
    4. Select the model, set Quantization to Float16
    5. Click Convert — this creates a .sentis file

    Then proceed to Step 2: Unity project setup and C# scripts.
    """)
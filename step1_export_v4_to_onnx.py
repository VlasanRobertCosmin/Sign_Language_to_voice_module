"""
Step 1 (V4): Export ASL V4 Model to ONNX
=========================================
Handles the V4 architecture with face landmarks and Squeezeformer blocks.

Usage:
    python step1_export_v4_to_onnx.py
"""

import sys
import os

# Import everything from the training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v4 import (
    ASLModelV4, INPUT_SIZE, MAX_FRAMES, D_MODEL, N_HEADS, N_LAYERS,
    SELECTED_FACE_INDICES, NUM_FACE_LANDMARKS
)

import torch
import numpy as np
import pickle
import json

MODEL_PATH = "asl_signs_model_v4.pth"
ENCODER_PATH = "asl_signs_encoder_v4.pkl"
ONNX_PATH = "asl_model_v4.onnx"

def main():
    print("=" * 60)
    print("  V4 MODEL → ONNX EXPORT")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    num_classes = ckpt['num_classes']
    input_size = ckpt['input_size']
    max_frames = ckpt['max_frames']
    d_model = ckpt.get('d_model', D_MODEL)
    n_layers = ckpt.get('n_layers', N_LAYERS)
    accuracy = ckpt.get('accuracy', 0)

    print(f"  Classes:    {num_classes}")
    print(f"  Input size: {input_size} (hands=126 + pose=99 + face={input_size-225})")
    print(f"  Frames:     {max_frames}")
    print(f"  Accuracy:   {accuracy*100:.2f}%")

    # Build model with dropout=0
    model = ASLModelV4(
        input_size, num_classes,
        d_model=d_model, n_heads=N_HEADS, n_layers=n_layers,
        dropout=0.0, max_frames=max_frames
    )

    # Load weights
    state = ckpt['model_state_dict']
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Export
    dummy = torch.randn(1, max_frames, input_size)
    with torch.no_grad():
        test_out = model(dummy)
    print(f"  Output:     {test_out.shape}")

    print(f"\n  Exporting to {ONNX_PATH}...")
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['landmarks'],
        output_names=['logits'],
        dynamic_axes={
            'landmarks': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
        dynamo=False,
    )
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  File size:  {size_mb:.1f} MB")

    # Validate
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ONNX_PATH)
        ort_out = session.run(None, {'landmarks': dummy.numpy()})[0]
        diff = np.max(np.abs(test_out.numpy() - ort_out))
        print(f"  Validation: max diff = {diff:.8f} {'✓ PASS' if diff < 0.01 else '✗ FAIL'}")
    except ImportError:
        print("  Skipping validation (install onnxruntime)")

    # Save labels
    classes = ckpt.get('classes', [])
    if classes:
        with open('asl_class_labels.txt', 'w') as f:
            for c in classes:
                f.write(f"{c}\n")
        with open('asl_class_labels.json', 'w') as f:
            json.dump({
                'classes': classes,
                'num_classes': len(classes),
                'input_size': input_size,
                'max_frames': max_frames,
                'face_landmark_indices': SELECTED_FACE_INDICES,
            }, f, indent=2)
        print(f"  Labels:     {len(classes)} classes saved")

    print(f"\n  DONE — bring {ONNX_PATH} into Unity")
    print(f"  NOTE: V4 uses {input_size} features (was 225 in V3)")
    print(f"  Update QuestToMediaPipeLandmarks.cs to include face landmarks!")

if __name__ == "__main__":
    main()

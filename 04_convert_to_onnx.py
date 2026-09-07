"""
Model conversion to ONNX format for browser deployment.

Exports the Keras model (PyTorch backend) to ONNX format,
which can be loaded by ONNX Runtime Web for client-side inference.

Also copies labels.json to the web directory.
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'

import shutil
import numpy as np
import torch
import keras

from losses import FocalLoss
import schedules


def main():
    os.makedirs('web', exist_ok=True)

    # ─── Load model ──────────────────────────────────────────────
    print("Loading best model...")
    model = keras.models.load_model(
        'checkpoints/best_model.keras',
        custom_objects={
            'FocalLoss': FocalLoss,
            'CosineAnnealingWarmup': schedules.CosineAnnealingWarmup
        }
    )

    # ─── Export to ONNX ──────────────────────────────────────────
    onnx_path = 'web/model.onnx'
    print("Converting to ONNX format...")

    # Create dummy input matching the model's expected shape
    dummy_input = torch.randn(1, 64, 64, 1, dtype=torch.float32)

    # Export using torch.onnx.export
    # Keras 3 with torch backend: model acts as a torch.nn.Module
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=17
        )
        print(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        print(f"Direct export failed: {e}")
        print("Trying alternative export method...")
        
        # Alternative: trace the model manually
        model.eval() if hasattr(model, 'eval') else None
        
        # Try using torch.jit.trace
        traced = torch.jit.trace(model, dummy_input)
        torch.onnx.export(
            traced,
            (dummy_input,),
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=17
        )
        print(f"ONNX model saved to {onnx_path} (via JIT trace)")

    # ─── Verify ONNX model ──────────────────────────────────────
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except Exception as e:
        print(f"ONNX verification warning: {e}")

    # ─── Verify with ONNX Runtime ────────────────────────────────
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        dummy_np = np.random.randn(1, 64, 64, 1).astype(np.float32)
        result = session.run(None, {input_name: dummy_np})
        print(f"ONNX Runtime test: output shape = {result[0].shape}")
        print(f"Sum of probabilities: {result[0].sum():.4f} (should be ~1.0)")
    except Exception as e:
        print(f"ONNX Runtime test warning: {e}")

    # ─── Copy labels ─────────────────────────────────────────────
    if os.path.exists('data/labels.json'):
        shutil.copy('data/labels.json', 'web/labels.json')
        print("Copied labels.json to web/labels.json")

    # ─── Report file sizes ───────────────────────────────────────
    total_size = 0
    print("\nFile sizes:")
    for f in os.listdir('web'):
        fp = os.path.join('web', f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp)
            total_size += size
            print(f"  {f}: {size / 1024:.1f} KB")

    print(f"\nTotal web assets: {total_size / (1024 * 1024):.2f} MB")


if __name__ == '__main__':
    main()

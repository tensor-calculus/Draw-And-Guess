"""
Grad-CAM visualization from scratch.

Computes gradient-weighted class activation maps to show what
the model focuses on when classifying sketches.

Uses PyTorch autograd for gradient computation (Keras 3 + torch backend).
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'

import json
import random
import numpy as np
import torch
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from losses import FocalLoss
import schedules


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Compute Grad-CAM heatmap from scratch using PyTorch autograd.
    
    Args:
        img_array: preprocessed image, shape (1, 64, 64, 1), float32 [0,1]
        model: trained Keras model
        last_conv_layer_name: name of last conv layer for Grad-CAM
        pred_index: class index (None = use predicted class)
    
    Returns:
        heatmap: numpy array (H, W), values [0, 1]
    """
    # Build sub-model: inputs -> [last_conv_output, predictions]
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Convert to torch tensor with gradient tracking
    input_tensor = torch.tensor(img_array, dtype=torch.float32, requires_grad=True)

    # Forward pass
    conv_output, preds = grad_model(input_tensor)

    if pred_index is None:
        pred_index = int(torch.argmax(preds[0]).item())

    # Get class score and compute gradients
    class_score = preds[0, pred_index]
    
    # Use torch.autograd.grad for intermediate tensors
    grads = torch.autograd.grad(
        class_score, conv_output, 
        retain_graph=True, create_graph=False
    )[0]

    # Global average pool the gradients → channel weights
    pooled_grads = grads.mean(dim=(0, 1, 2))  # (channels,)

    # Weighted sum of feature maps
    conv_out = conv_output[0].detach()  # (H, W, channels)
    heatmap = (conv_out * pooled_grads).sum(dim=-1)  # (H, W)

    # ReLU and normalize
    heatmap = torch.relu(heatmap)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.detach().cpu().numpy()


def resize_heatmap(heatmap, target_size=(64, 64)):
    """Resize heatmap using PIL."""
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode='L')
    heatmap_img = heatmap_img.resize(target_size, Image.Resampling.BILINEAR)
    return np.array(heatmap_img).astype(np.float32) / 255.0


def main():
    os.makedirs('results/gradcam', exist_ok=True)

    # ─── Load model ──────────────────────────────────────────────
    print("Loading model...")
    model = keras.models.load_model(
        'checkpoints/best_model.keras',
        custom_objects={
            'FocalLoss': FocalLoss,
            'CosineAnnealingWarmup': schedules.CosineAnnealingWarmup
        }
    )
    last_conv_name = 'last_conv'

    # ─── Load test data ──────────────────────────────────────────
    print("Loading test data...")
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    with open('data/labels.json', 'r') as f:
        labels_list = json.load(f)

    num_classes = len(labels_list)

    # ─── Select samples: 2 from each of 10 random categories present in the test set ────
    random.seed(42)
    unique_test_classes = np.unique(y_test).tolist()
    num_to_sample = min(10, len(unique_test_classes))
    chosen_categories = random.sample(unique_test_classes, num_to_sample) if num_to_sample > 0 else []

    samples = []
    for cat in chosen_categories:
        cat_indices = np.where(y_test == cat)[0]
        if len(cat_indices) >= 2:
            chosen = random.sample(list(cat_indices), 2)
            samples.extend(chosen)
        elif len(cat_indices) > 0:
            samples.extend(cat_indices.tolist())

    if len(samples) > 20:
        samples = random.sample(samples, 20)

    print(f"Generating Grad-CAM for {len(samples)} samples...")

    # ─── Generate visualizations ─────────────────────────────────
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    jet = plt.colormaps.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]

    for i, idx in enumerate(samples[:20]):
        img = X_test[idx]  # (64, 64), uint8
        true_label = labels_list[y_test[idx]]

        # Preprocess: normalize, add batch + channel dims
        img_float = img.astype(np.float32) / 255.0
        img_array = img_float[np.newaxis, :, :, np.newaxis]  # (1, 64, 64, 1)

        # Get prediction
        preds = model.predict(img_array, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        pred_label = labels_list[pred_idx]

        # Compute Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_name)
        heatmap = resize_heatmap(heatmap, (64, 64))

        # Create colored heatmap overlay
        heatmap_uint8 = np.uint8(255 * heatmap)
        jet_heatmap = jet_colors[heatmap_uint8]  # (64, 64, 3)

        # Convert grayscale to RGB
        img_rgb = np.stack([img_float] * 3, axis=-1)

        # Blend: 40% heatmap + 60% sketch
        superimposed = jet_heatmap * 0.4 + img_rgb * 0.6
        superimposed = np.clip(superimposed, 0, 1)

        axes[i].imshow(superimposed)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
        axes[i].axis('off')

        # Save individual
        plt.imsave(
            f'results/gradcam/sample_{i}_{true_label}_{pred_label}.png',
            superimposed
        )

    # Hide unused subplots
    for j in range(len(samples), 20):
        axes[j].axis('off')

    plt.suptitle('Grad-CAM Visualizations — What the Model Sees', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/gradcam_examples.png', dpi=150)
    plt.close()

    print("Saved Grad-CAM grid to results/gradcam_examples.png")
    print(f"Saved {len(samples)} individual images to results/gradcam/")


if __name__ == '__main__':
    main()

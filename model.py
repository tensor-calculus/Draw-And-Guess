"""
Custom ResNet with Squeeze-and-Excitation (SE) attention blocks.
Built from scratch using Keras 3 Functional API - no pretrained model imports.

Architecture:
    Stem -> 4 stages of residual blocks with SE attention -> GlobalAvgPool -> Dense
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import keras
from keras import layers, Model


def se_block(x, reduction_ratio=16):
    """Squeeze-and-Excitation block.
    
    Squeeze: GlobalAvgPool to get per-channel descriptor.
    Excitation: Two FC layers (bottleneck) to learn channel-wise attention.
    Scale: Multiply attention weights back to feature maps.
    """
    channels = x.shape[-1]
    # Squeeze
    squeeze = layers.GlobalAveragePooling2D(keepdims=False)(x)
    # Excitation
    excitation = layers.Dense(
        channels // reduction_ratio, activation='relu',
        kernel_initializer='he_normal'
    )(squeeze)
    excitation = layers.Dense(
        channels, activation='sigmoid',
        kernel_initializer='he_normal'
    )(excitation)
    # Reshape for broadcasting: (batch, channels) -> (batch, 1, 1, channels)
    excitation = layers.Reshape((1, 1, channels))(excitation)
    # Scale
    scaled_x = layers.Multiply()([x, excitation])
    return scaled_x


def residual_block(x, filters, stride=1, use_projection=False, is_last_conv=False):
    """Residual block with SE attention.
    
    Args:
        x: Input tensor.
        filters: Number of conv filters.
        stride: Stride for first conv (>1 for downsampling).
        use_projection: If True, use 1x1 conv on skip connection for dimension matching.
        is_last_conv: If True, name the second conv 'last_conv' for Grad-CAM.
    """
    shortcut = x
    if use_projection:
        shortcut = layers.Conv2D(
            filters, (1, 1), strides=stride, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )(x)
        shortcut = layers.BatchNormalization(momentum=0.9)(shortcut)

    # First conv
    x1 = layers.Conv2D(
        filters, (3, 3), strides=stride, padding='same',
        use_bias=False, kernel_initializer='he_normal'
    )(x)
    x1 = layers.BatchNormalization(momentum=0.9)(x1)
    x1 = layers.ReLU()(x1)

    # Second conv (named 'last_conv' if this is the final residual block)
    conv2_name = 'last_conv' if is_last_conv else None
    x1 = layers.Conv2D(
        filters, (3, 3), strides=1, padding='same',
        use_bias=False, kernel_initializer='he_normal', name=conv2_name
    )(x1)
    x1 = layers.BatchNormalization(momentum=0.9)(x1)

    # Squeeze-and-Excitation attention
    x1 = se_block(x1)

    # Add skip connection and activate
    out = layers.Add()([shortcut, x1])
    out = layers.ReLU()(out)
    return out


def build_resnet_se(num_classes=345, input_shape=(64, 64, 1)):
    """Build the full ResNet-SE model.
    
    Args:
        num_classes: Number of output classes.
        input_shape: Input image shape (H, W, C).
    
    Returns:
        Uncompiled Keras Model.
    """
    inputs = layers.Input(shape=input_shape)

    # Stem: Conv 7x7, stride 2 → BN → ReLU → MaxPool
    x = layers.Conv2D(
        64, (7, 7), strides=2, padding='same',
        use_bias=False, kernel_initializer='he_normal'
    )(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((3, 3), strides=2, padding='same')(x)

    # Stage 1: 2 residual blocks, 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2: 2 residual blocks, 128 filters (downsample)
    x = residual_block(x, 128, stride=2, use_projection=True)
    x = residual_block(x, 128)

    # Stage 3: 2 residual blocks, 256 filters (downsample)
    x = residual_block(x, 256, stride=2, use_projection=True)
    x = residual_block(x, 256)

    # Stage 4: 2 residual blocks, 512 filters (downsample, last one named for Grad-CAM)
    x = residual_block(x, 512, stride=2, use_projection=True)
    x = residual_block(x, 512, is_last_conv=True)

    # Head: GlobalAvgPool → Dense → Dropout → Softmax
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs, outputs, name='resnet_se')
    return model


def get_model_summary(model):
    """Print model summary and return total parameter count."""
    model.summary()
    return model.count_params()

"""
Custom sketch-specific data augmentation for the Draw & Guess project.

Uses numpy/scipy for augmentation transforms, and keras.utils.PyDataset
for backend-agnostic data loading compatible with model.fit().

Augmentations:
    - Random rotation (±15°)
    - Random translation (±3px)
    - Elastic deformation (simulates hand tremor)
    - Stroke thickness variation (morphological dilate/erode)
    - Partial stroke dropout (random rectangular patch removal)
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, grey_dilation, grey_erosion
from PIL import Image
import keras

NUM_CLASSES = 345


def random_rotation(image, max_angle=15.0):
    """Random rotation by up to ±max_angle degrees."""
    angle = np.random.uniform(-max_angle, max_angle)
    img_pil = Image.fromarray((image[:, :, 0] * 255).astype(np.uint8), mode='L')
    img_pil = img_pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    return np.array(img_pil, dtype=np.float32)[:, :, np.newaxis] / 255.0


def random_translation(image, max_shift=3):
    """Random translation by up to ±max_shift pixels."""
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    result = np.roll(image, shift=dx, axis=1)
    result = np.roll(result, shift=dy, axis=0)
    return result


def elastic_deformation(image, alpha=2.0, sigma=0.5):
    """Elastic deformation to simulate hand tremor.
    
    Generate random displacement fields, smooth with gaussian, apply.
    """
    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = [np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,))]
    distorted = map_coordinates(image[:, :, 0], indices, order=1, mode='reflect')
    return distorted.reshape(shape)[:, :, np.newaxis].astype(np.float32)


def stroke_thickness_variation(image):
    """Randomly dilate or erode to simulate stroke thickness variation."""
    img_2d = image[:, :, 0]
    if np.random.rand() < 0.5:
        # Dilate (thicken strokes)
        result = grey_dilation(img_2d, size=(3, 3))
    else:
        # Erode (thin strokes)
        result = grey_erosion(img_2d, size=(3, 3))
    return np.clip(result, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)


def random_stroke_dropout(image, box_size=8):
    """Randomly black out rectangular patches to simulate missing strokes."""
    h, w = image.shape[:2]
    x = np.random.randint(0, max(w - box_size, 1))
    y = np.random.randint(0, max(h - box_size, 1))
    result = image.copy()
    result[y:y + box_size, x:x + box_size, :] = 0.0
    return result


def augment_image(image):
    """Apply all augmentations with random probability.
    
    Each augmentation is applied with ~50% probability.
    Input: float32 image (64, 64, 1), values [0, 1]
    Output: float32 image (64, 64, 1), values [0, 1]
    """
    if np.random.rand() < 0.5:
        image = random_rotation(image)
    if np.random.rand() < 0.5:
        image = random_translation(image)
    if np.random.rand() < 0.5:
        image = elastic_deformation(image)
    if np.random.rand() < 0.5:
        image = stroke_thickness_variation(image)
    if np.random.rand() < 0.5:
        image = random_stroke_dropout(image)
    return np.clip(image, 0.0, 1.0)


class SketchDataset(keras.utils.PyDataset):
    """Keras-compatible dataset for sketch data with on-the-fly augmentation.
    
    Works with any Keras 3 backend (PyTorch, TensorFlow, JAX).
    """
    
    def __init__(self, X, y, batch_size=128, augment=True, shuffle=True, **kwargs):
        """
        Args:
            X: numpy array (N, 64, 64), uint8 images
            y: numpy array (N,), int32 class labels
            batch_size: Batch size
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle at epoch end
        """
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Load and normalize batch
        batch_X = self.X[batch_indices].astype(np.float32) / 255.0
        batch_X = batch_X[..., np.newaxis]  # (B, 64, 64, 1)
        
        # Apply augmentation per-image
        if self.augment:
            batch_X = np.array([augment_image(img) for img in batch_X])
        
        # One-hot encode labels
        batch_y = keras.utils.to_categorical(self.y[batch_indices], NUM_CLASSES)
        
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_train_dataset(X, y, batch_size=128):
    """Create a training dataset with augmentation.
    
    Args:
        X: numpy array (N, 64, 64), uint8
        y: numpy array (N,), int32
    
    Returns:
        SketchDataset compatible with model.fit()
    """
    return SketchDataset(X, y, batch_size=batch_size, augment=True, shuffle=True)


def create_eval_dataset(X, y, batch_size=128):
    """Create an evaluation dataset without augmentation.
    
    Args:
        X: numpy array (N, 64, 64), uint8
        y: numpy array (N,), int32
    
    Returns:
        SketchDataset compatible with model.evaluate()
    """
    return SketchDataset(X, y, batch_size=batch_size, augment=False, shuffle=False)

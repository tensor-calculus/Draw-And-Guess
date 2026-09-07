"""
Focal Loss for multi-class classification.

Focal loss down-weights well-classified examples and focuses on hard ones,
especially useful when some sketch categories look visually similar.

FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

Uses Keras 3 ops for backend-agnostic computation (works with PyTorch, TF, JAX).
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import keras
from keras import ops


@keras.saving.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """Focal Loss for multi-class classification.
    
    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Balancing factor. Controls relative weight of focal term.
        label_smoothing: Amount of label smoothing to apply. 0.0 = no smoothing.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, 
                 name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.convert_to_tensor(y_true)
        y_true = ops.cast(y_true, y_pred.dtype)
        
        # Handle sparse labels: convert to one-hot
        if len(ops.shape(y_true)) == 1 or ops.shape(y_true)[-1] == 1:
            y_true = ops.squeeze(ops.cast(y_true, 'int32'))
            num_classes = ops.shape(y_pred)[-1]
            y_true = ops.one_hot(y_true, num_classes)
            y_true = ops.cast(y_true, y_pred.dtype)

        # Apply label smoothing
        if self.label_smoothing > 0.0:
            num_classes = ops.cast(ops.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        # Clip predictions for numerical stability
        epsilon = keras.backend.epsilon()
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # Compute p_t: probability of correct class
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = ops.power(1.0 - p_t, self.gamma)

        # Cross-entropy term: -log(p_t)
        ce_loss = -ops.log(p_t)

        # Combined focal loss
        loss = self.alpha * focal_weight * ce_loss

        # Sum over classes, mean over batch
        return ops.mean(ops.sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
        })
        return config

import keras
import numpy as np

@keras.saving.register_keras_serializable()
class CosineAnnealingWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine Annealing with Linear Warmup LR schedule.
    
    - Warmup phase: linearly increase from 0 to peak_lr
    - Cosine phase: decay from peak_lr to min_lr following cosine curve
    """
    
    def __init__(self, warmup_steps, total_steps, peak_lr=1e-3, min_lr=1e-6):
        super().__init__()
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = float(step)
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (step / max(self.warmup_steps, 1))
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(1.0, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay

    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'peak_lr': self.peak_lr,
            'min_lr': self.min_lr,
        }

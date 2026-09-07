"""
Training script for the Draw & Guess ResNet-SE model.
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'

import json
import numpy as np
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from model import build_resnet_se
from losses import FocalLoss
from augmentation import create_train_dataset, create_eval_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ])

from schedules import CosineAnnealingWarmup

def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    logging.info("Loading data ...")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')

    with open('data/labels.json', 'r') as f:
        labels = json.load(f)

    num_classes = len(labels)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"Classes: {num_classes}")

    # Create datasets
    batch_size = 128
    epochs = 50

    train_dataset = create_train_dataset(X_train, y_train, batch_size=batch_size)
    val_dataset = create_eval_dataset(X_val, y_val, batch_size=batch_size)

    steps_per_epoch = len(train_dataset)
    warmup_steps = 3 * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    # Build model
    model = build_resnet_se(num_classes=num_classes)
    model.summary(show_trainable=True)

    # LR schedule + optimizer
    lr_schedule = CosineAnnealingWarmup(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        peak_lr=1e-3,
        min_lr=1e-6,
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    backup_dir = 'checkpoints/backup'
    if os.path.exists(backup_dir) and os.path.isdir(backup_dir) and os.listdir(backup_dir):
        logging.info("Found latest model backup. Training will resume from the interrupted epoch...")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'checkpoints/best_model.keras',
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'checkpoints/latest_model.keras',
            save_best_only=False, verbose=1
        ),
        keras.callbacks.BackupAndRestore(
            backup_dir=backup_dir
        ),
    ]

    # Train
    logging.info(f"Starting training for {epochs} epochs ...")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Warmup steps: {warmup_steps}")
    logging.info(f"Total steps: {total_steps}")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save final model
    model.save('checkpoints/best_model.keras')
    logging.info("Model saved to checkpoints/best_model.keras")

    # Evaluate
    val_results = model.evaluate(val_dataset)
    logging.info(f"Final Validation Loss: {val_results[0]:.4f}")
    logging.info(f"Final Validation Accuracy: {val_results[1]:.4f}")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Loss History', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_title('Accuracy History', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    plt.close()
    logging.info("Training history saved to results/training_history.png")


if __name__ == '__main__':
    main()

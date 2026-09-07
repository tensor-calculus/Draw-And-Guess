"""
Model evaluation script for Draw & Guess.

Computes top-1 accuracy, top-3 accuracy, per-class stats,
and generates confusion analysis visualization.
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'

import json
import numpy as np
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging
from losses import FocalLoss
import schedules
from augmentation import create_eval_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    os.makedirs('results', exist_ok=True)

    # Load model
    logging.info("Loading model ...")
    model = keras.models.load_model(
        'checkpoints/best_model.keras',
        custom_objects={
            'FocalLoss': FocalLoss,
            'CosineAnnealingWarmup': schedules.CosineAnnealingWarmup
        }
    )

    # Load test data 
    logging.info("Loading test data ...")
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    with open('data/labels.json', 'r') as f:
        labels_list = json.load(f)

    num_classes = len(labels_list)
    test_dataset = create_eval_dataset(X_test, y_test, batch_size=128)

    # Predict
    logging.info("Running predictions ...")
    predictions = model.predict(test_dataset)
    pred_classes = np.argmax(predictions, axis=1)

    # Top-1 accuracy
    top1_correct = np.sum(pred_classes == y_test)
    top1_acc = top1_correct / len(y_test)

    # Top-3 accuracy 
    top3_preds = np.argsort(predictions, axis=1)[:, -3:]
    top3_correct = sum(y_test[i] in top3_preds[i] for i in range(len(y_test)))
    top3_acc = top3_correct / len(y_test)

    baseline_acc = 1.0 / num_classes

    print(f"\n{'='*50}")
    print(f"  Top-1 Accuracy:  {top1_acc:.4f}")
    print(f"  Top-3 Accuracy:  {top3_acc:.4f}")
    print(f"  Naive Baseline:  {baseline_acc:.4f} (1/{num_classes})")
    print(f"  Lift over base:  {top1_acc / baseline_acc:.1f}x")
    print(f"{'='*50}")

    # Per-class accuracy
    cm = confusion_matrix(y_test, pred_classes, labels=range(num_classes))

    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        total_class = np.sum(cm[i, :])
        if total_class > 0:
            per_class_acc[i] = cm[i, i] / total_class

    class_acc_sorted = np.argsort(per_class_acc)

    best_5 = [(labels_list[i], per_class_acc[i]) for i in class_acc_sorted[-5:][::-1]]
    worst_5 = [(labels_list[i], per_class_acc[i]) for i in class_acc_sorted[:5]]

    print("\nTop 5 Best Performing Categories:")
    for name, acc in best_5:
        print(f"  {name}: {acc:.4f}")

    print("\nTop 5 Worst Performing Categories:")
    for name, acc in worst_5:
        print(f"  {name}: {acc:.4f}")

    # Confusion analysis
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    confused_flat = np.argsort(cm_offdiag, axis=None)[-20:]
    confused_indices = np.unravel_index(confused_flat, cm_offdiag.shape)
    
    top_20_confused = []
    for i in range(19, -1, -1):
        true_c = confused_indices[0][i]
        pred_c = confused_indices[1][i]
        count = int(cm_offdiag[true_c, pred_c])
        top_20_confused.append({
            'true': labels_list[true_c],
            'pred': labels_list[pred_c],
            'count': count
        })

    print("\nTop 20 Most Confused Pairs:")
    for pair in top_20_confused:
        print(f"  {pair['true']} -> {pair['pred']} (Count: {pair['count']})")

    # Save metrics
    with open('results/metrics.txt', 'w') as f:
        f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n")
        f.write(f"Naive Baseline: {baseline_acc:.4f}\n")
        f.write(f"Lift over baseline: {top1_acc / baseline_acc:.1f}x\n\n")
        f.write("Top 5 Best Performing Categories:\n")
        for name, acc in best_5:
            f.write(f"  {name}: {acc:.4f}\n")
        f.write("\nTop 5 Worst Performing Categories:\n")
        for name, acc in worst_5:
            f.write(f"  {name}: {acc:.4f}\n")
        f.write("\nTop 20 Most Confused Pairs:\n")
        for pair in top_20_confused:
            f.write(f"  {pair['true']} -> {pair['pred']} (Count: {pair['count']})\n")

    # Plot confusion analysis
    plot_labels = [f"{p['true']} -> {p['pred']}" for p in top_20_confused]
    counts = [p['count'] for p in top_20_confused]

    plt.figure(figsize=(12, 8))
    plt.barh(plot_labels[::-1], counts[::-1], color='#4f8cff', edgecolor='#3a6fd8')
    plt.xlabel('Confusion Count', fontsize=12)
    plt.title('Top 20 Most Confused Category Pairs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/confusion_analysis.png', dpi=150)
    plt.close()

    logging.info("Results saved to results/metrics.txt and results/confusion_analysis.png")


if __name__ == '__main__':
    main()

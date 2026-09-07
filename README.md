# Draw & Guess

A machine learning project that trains a Convolutional Neural Network (CNN) to recognize hand-drawn sketches using the Google Quick, Draw! dataset. 

This repository contains the backend and training pipeline: downloading data, preprocessing it, training a Keras model, evaluating its performance, and exporting it to ONNX for efficient deployment (e.g., in-browser inference).

## Features

- **Data Pipeline:** Scripts to download and rasterize the Quick, Draw! dataset into bitmaps.
- **Model Training:** Train a CNN using Keras with data augmentation, learning rate schedules, and checkpointing.
- **Evaluation:** Evaluate the model and visualize predictions (includes Grad-CAM implementation).
- **Exporting:** Convert the trained model to ONNX format for cross-platform, client-side deployment.

## Installation

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch may require specific installation instructions depending on your CUDA version).*

## Usage

Run the scripts in the following order to build and export the model:

1. **Prepare Data:** Download and preprocess the Quick, Draw! data.
   ```bash
   python 01_prepare_data.py
   ```

2. **Train Model:** Train the Keras CNN on the processed data.
   ```bash
   python 02_train_model.py
   ```

3. **Evaluate Model:** Evaluate model accuracy and visualize results.
   ```bash
   python 03_evaluate_model.py
   ```

4. **Convert Model:** Export the trained model to ONNX format.
   ```bash
   python 04_convert_to_onnx.py
   ```

## Project Structure

- `01_prepare_data.py` / `02_train_model.py` / `03_evaluate_model.py` / `04_convert_to_onnx.py`: Core pipeline scripts.
- `model.py`, `losses.py`, `schedules.py`, `augmentation.py`: Model architecture, training utilities, and data augmentation.
- `gradcam.py`: Utilities for Grad-CAM visualization.
- `categories.txt`: List of drawing categories used.

# 100 Interview Questions and Answers for Draw & Guess Project (MLE / Data Science)

This document contains 100 interview-ready questions and answers based on the "Draw & Guess" machine learning project, specifically tailored for Machine Learning Engineer (MLE) and Data Science roles during on-campus placements at IITs. 

## Section 1: Data & Preprocessing

**Q1: Why represent the Quick, Draw! dataset as bitmaps instead of stroke sequences?**
A1: Bitmaps allow the use of standard, highly optimized Convolutional Neural Networks (CNNs) for spatial pattern recognition, avoiding the sequential complexity and variable-length sequence handling required for stroke sequences.

**Q2: What is the significance of the image resolution (e.g., 64x64) chosen for this project?**
A2: It balances spatial detail required to distinguish sketch classes with computational efficiency, minimizing memory footprint and accelerating both training and inference.

**Q3: How do you handle the inherent sparsity (mostly empty background) in sketch bitmaps?**
A3: We normalize the pixel values and use convolutional layers which naturally learn to detect spatial features like edges and curves while being invariant to mostly zero-valued background regions.

**Q4: What data augmentation techniques are most effective for hand-drawn sketches?**
A4: Random rotations, translations (shifts), and zooming are highly effective as they simulate the natural variations in how different users scale and position their drawings.

**Q5: Why might random flipping (horizontal/vertical) be detrimental for certain classes in this dataset?**
A5: Flipping can destroy orientation-dependent semantic meaning, for example, making a "b" look like a "d", or altering directional objects like a "right-pointing arrow."

**Q6: How does data augmentation prevent overfitting in this CNN?**
A6: It synthetically increases the diversity of the training set, forcing the model to learn generalized spatial features rather than memorizing the exact pixel configurations of the training samples.

**Q7: How do you split the dataset to ensure robust evaluation?**
A7: We use a stratified train-validation-test split to maintain the class distribution across all splits, ensuring evaluation metrics reflect performance on all classes equally.

**Q8: Why is standardization (zero mean, unit variance) or normalization (0-1 range) necessary for the image inputs?**
A8: It ensures that all input features have a consistent scale, leading to better-behaved gradients, faster convergence, and preventing any single feature from dominating the learning process.

**Q9: What format is most efficient for storing and loading the preprocessed bitmap dataset during training?**
A9: `.npy` or binary formats, as they allow for fast, sequential, and batched memory-mapped reading from disk directly into GPU memory without image decoding overhead.

**Q10: How would you handle noisy or mislabeled sketches present in the crowd-sourced Quick, Draw! dataset?**
A10: I rely on the robustness of the Focal Loss to down-weight outliers and use the inherent generalization of the CNN, combined with aggressive early stopping to avoid memorizing noisy labels.

## Section 2: Architecture (ResNet & SE Blocks)

**Q11: Why choose a ResNet architecture over a VGG-style sequential CNN for this task?**
A11: ResNet uses skip connections to mitigate the vanishing gradient problem, allowing for a deeper network that can learn more complex hierarchical features without degradation in training accuracy.

**Q12: What is the mathematical function of the skip connection in a Residual Block?**
A12: It performs an element-wise addition of the block's input to its output: $F(x) + x$, forcing the convolutional layers to learn the residual mapping $F(x)$ rather than the direct mapping.

**Q13: Why is a 1x1 convolution (projection shortcut) sometimes used in the skip connection?**
A13: It is used to match dimensions when the number of filters increases or the spatial dimensions decrease (via strided convolutions) between the input and output of the residual block.

**Q14: What is the purpose of the 7x7 convolution in the stem of the ResNet?**
A14: It rapidly reduces the spatial resolution of the input image while increasing the channel depth, acting as an aggressive feature extractor that reduces the computational load for subsequent layers.

**Q15: Why use Batch Normalization after convolutions in the residual blocks?**
A15: It normalizes the intermediate feature maps, stabilizing the learning process, reducing internal covariate shift, and allowing for higher learning rates.

**Q16: Where exactly is the activation function applied in relation to the skip connection?**
A16: The final ReLU activation is applied *after* the element-wise addition of the residual mapping and the shortcut connection.

**Q17: What does the Squeeze-and-Excitation (SE) block achieve?**
A17: It explicitly models channel-wise interdependencies, adaptively recalibrating channel feature responses by suppressing less useful features and emphasizing informative ones.

**Q18: Explain the "Squeeze" operation in the SE block.**
A18: It uses Global Average Pooling to compress the spatial dimensions (H x W) of the feature maps into a single 1D vector (1 x 1 x C), capturing a global receptive field for each channel.

**Q19: Explain the "Excitation" operation in the SE block.**
A19: It passes the squeezed vector through a two-layer dense bottleneck (down-projection then up-projection) with a sigmoid activation to output channel-specific scaling weights between 0 and 1.

**Q20: Why does the Excitation step use a bottleneck (reduction ratio)?**
A20: To limit the computational overhead and model complexity while forcing the network to learn a compact, generalized representation of channel dependencies.

**Q21: How are the SE block's output weights applied to the feature maps?**
A21: They are applied via element-wise channel broadcasting multiplication, scaling each feature map by its corresponding learned attention weight.

**Q22: Why place the SE block inside the residual block rather than outside it?**
A22: Placing it inside allows the block to refine the residual mapping $F(x)$ before it is added to the identity shortcut, optimizing the new features being learned.

**Q23: How does replacing Flatten with Global Average Pooling (GAP) before the final dense layer benefit the model?**
A23: GAP significantly reduces the total number of parameters, making the model less prone to overfitting, and inherently provides localization ability useful for interpretability.

**Q24: What is the role of Dropout before the final classification layer?**
A24: It randomly zeroes out a fraction of activations during training to prevent co-adaptation of neurons, acting as a strong regularizer.

**Q25: Why use the He Normal initialization for convolutional layers in this architecture?**
A25: It is specifically designed to keep the variance of activations stable across deep networks that use ReLU activation functions, preventing gradients from vanishing or exploding early in training.

**Q26: What is the effect of setting `use_bias=False` in convolutions followed by Batch Normalization?**
A26: The bias term is redundant because Batch Normalization centers the data, effectively cancelling out any learned bias from the preceding convolution.

**Q27: How does a stride of 2 in a convolutional layer differ from Max Pooling for downsampling?**
A27: A strided convolution learns the downsampling operation, whereas Max Pooling is a fixed, non-parametric spatial downsampling based on the maximum activation.

**Q28: How does the network's receptive field change as you go deeper into the ResNet stages?**
A28: The receptive field exponentially increases, allowing deeper layers to capture large-scale, global semantic features of the sketch, while early layers capture local features like strokes and edges.

**Q29: What happens if the reduction ratio in the SE block is set to 1?**
A29: The bottleneck is removed, drastically increasing parameter count and computational cost, and potentially leading to overfitting without significant performance gains.

**Q30: Can SE blocks be trained end-to-end with the rest of the network?**
A30: Yes, they consist purely of differentiable operations (Global Average Pooling, Dense layers, Sigmoid), allowing standard backpropagation.

**Q31: What is the advantage of using the Keras Functional API over the Sequential API for this model?**
A31: The Functional API is required to implement non-linear topologies like the skip connections in ResNet and the parallel branching in SE blocks.

**Q32: How does the number of filters scale across the 4 stages of this ResNet?**
A32: It typically doubles at each spatial downsampling stage (e.g., 64 -> 128 -> 256 -> 512) to compensate for the loss of spatial resolution and maintain information capacity.

**Q33: Why is the last convolutional layer explicitly named ('last_conv')?**
A33: To provide a deterministic string reference needed to extract feature maps and gradients for the Grad-CAM visualization algorithm.

**Q34: How do you handle multi-class classification at the output layer?**
A34: I use a Dense layer with a number of units equal to the number of classes, followed by a Softmax activation function to output a valid probability distribution.

**Q35: In terms of hardware deployment, what is the drawback of SE blocks?**
A35: The GAP and dense operations break the pure spatial parallelizability of standard convolutions, occasionally causing memory access bottlenecks on some specialized edge AI accelerators.

**Q36: If the model is underfitting, which architectural hyperparameters would you change?**
A36: I would increase the network capacity by adding more residual blocks per stage or increasing the base number of convolutional filters.

**Q37: If the model is overfitting despite Dropout, what architectural change might help?**
A37: I would decrease model depth/width, increase the dropout rate, or add higher weight decay (L2 regularization) to the convolutional kernels.

**Q38: Why use ReLU over Sigmoid or Tanh in the hidden layers?**
A38: ReLU is computationally cheaper and avoids the vanishing gradient problem in the positive domain, leading to faster and more stable convergence in deep networks.

**Q39: How does Batch Normalization behave differently during inference compared to training?**
A39: During training, it normalizes using the batch mean and variance; during inference, it uses the moving average of mean and variance accumulated during training.

**Q40: What defines a "stage" in this ResNet architecture?**
A40: A stage is a sequence of residual blocks operating at the same spatial resolution, usually ending with a block that downsamples the resolution and increases channel depth.

## Section 3: Training & Loss Functions

**Q41: Why use Focal Loss instead of standard Categorical Cross-Entropy for this dataset?**
A41: Focal Loss dynamically scales down the loss assigned to easily classified examples, forcing the model to focus on hard, misclassified examples, which is crucial if hard-to-distinguish sketches exist.

**Q42: What is the role of the $\gamma$ (gamma) parameter in Focal Loss?**
A42: It is the focusing parameter that smoothly adjusts the rate at which easy examples are down-weighted; a higher $\gamma$ forces more focus on hard examples.

**Q43: What is the role of the $\alpha$ (alpha) parameter in Focal Loss?**
A43: It is the balancing parameter used to weight the loss contribution of different classes, explicitly addressing class frequency imbalance in the training data.

**Q44: If $\gamma = 0$, what does Focal Loss become?**
A44: It reduces mathematically to standard Categorical Cross-Entropy loss.

**Q45: Why use Keras 3 with a PyTorch backend (`KERAS_BACKEND='torch'`)?**
A45: It combines the high-level, user-friendly API and training loop of Keras with the eager execution, debugging ease, and optimized tensor operations of PyTorch.

**Q46: Explain Cosine Annealing Learning Rate scheduling.**
A46: It smoothly decreases the learning rate following a cosine curve from a maximum value to a minimum value over a set number of epochs.

**Q47: Why include a "Warmup" phase in the learning rate schedule?**
A47: It linearly increases the learning rate from near-zero to the peak value during the initial epochs to prevent premature convergence and stabilize gradients when weights are randomly initialized.

**Q48: How does the Adam optimizer update weights compared to standard SGD?**
A48: Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients, whereas SGD uses a single, global learning rate.

**Q49: What is the purpose of Early Stopping?**
A49: It halts training when a monitored metric (like validation loss) stops improving for a specified number of epochs (patience), preventing overfitting to the training data.

**Q50: Why use `restore_best_weights=True` in Early Stopping?**
A50: Because the model's weights at the epoch where training is stopped might be overfitted (due to the patience delay); this ensures we keep the weights from the epoch with the actual best validation metric.

**Q51: What is the difference between `val_loss` and `val_accuracy` for monitoring model performance?**
A51: `val_loss` measures the continuous probability error of predictions, while `val_accuracy` measures the discrete percentage of correct top-1 predictions; loss is smoother for optimization.

**Q52: Why save checkpoints using `save_best_only=True`?**
A52: To ensure disk space is not wasted and that we only retain the model file corresponding to the highest validation performance achieved during the training run.

**Q53: How does the `BackupAndRestore` callback provide fault tolerance?**
A53: It saves temporary training state at the end of every epoch so that if the process crashes (e.g., due to a GPU timeout or OOM), training can resume exactly from the interrupted epoch.

**Q54: Why set `matplotlib.use('Agg')` in the training script?**
A54: It sets the Matplotlib backend to a non-interactive, file-writing mode, which is necessary when running scripts on headless servers without a GUI display.

**Q55: What does `batch_size=128` signify in the training loop?**
A55: The model processes 128 images simultaneously, aggregating their gradients before making a single weight update, balancing memory usage and gradient estimation stability.

**Q56: How do you determine the `steps_per_epoch`?**
A56: It is the total number of training samples divided by the batch size, representing the number of gradient updates required to pass through the entire dataset once.

**Q57: If training loss decreases but validation loss increases, what is happening?**
A57: The model is overfitting; it is memorizing the training data and losing its ability to generalize to unseen validation data.

**Q58: What is a mixed-precision training strategy, and how would it benefit this model?**
A58: It uses float16 for activations and gradients but float32 for weight updates, halving VRAM usage and doubling throughput on Tensor Cores without losing accuracy.

**Q59: How do you calculate the peak learning rate for the Cosine Annealing schedule?**
A59: It is typically found using a Learning Rate Finder technique (systematically increasing LR and observing loss) or empirically set based on standard practices for Adam (e.g., 1e-3).

**Q60: Why evaluate on a test set if we already use a validation set?**
A60: The validation set is used iteratively to tune hyperparameters and early stopping; the test set is kept completely unseen to provide an unbiased estimate of final model performance.

**Q61: What does `model.summary(show_trainable=True)` provide?**
A61: It outputs the layer topology, output shapes, and explicitly breaks down the count of trainable vs. non-trainable parameters (like Batch Norm moving averages).

**Q62: How do you handle the trade-off between `patience` in Early Stopping and total epochs?**
A62: A higher patience allows the model to ride out temporary validation bumps but risks slight overfitting, while lower patience might terminate training prematurely during a noisy gradient phase.

**Q63: What metric would you use instead of Accuracy if the dataset was highly imbalanced?**
A63: Macro F1-score, Precision-Recall AUC, or Balanced Accuracy, as they do not artificially inflate scores based on the majority class.

**Q64: Why use Python's `logging` module instead of standard `print` statements?**
A64: `logging` allows for severity levels, timestamps, concurrent writing to a file and console, and better integration with production monitoring systems.

**Q65: If the model predicts uniform probabilities for all classes at epoch 1, what does this indicate?**
A65: It indicates the weights are properly randomized and the model has not yet learned any discriminative features, functioning as a random guesser.

## Section 4: Interpretability (Grad-CAM)

**Q66: What is the primary purpose of Grad-CAM in this project?**
A66: To provide visual explanations for the CNN's predictions by highlighting the specific spatial regions of the sketch that most heavily influenced the final class decision.

**Q67: How does Grad-CAM mathematically compute the importance of feature maps?**
A67: It computes the gradient of the target class score with respect to the feature maps of the last convolutional layer, globally averaging these gradients to obtain channel importance weights.

**Q68: Why do we use the *last* convolutional layer for Grad-CAM?**
A68: It contains the best compromise between high-level semantic information (class-specific features) and sufficient spatial resolution to map back to the original image.

**Q69: What happens when you apply a ReLU function to the weighted combination of feature maps in Grad-CAM?**
A69: It zeros out negative values, ensuring we only highlight features that have a *positive* influence on the target class, ignoring features that suppress it.

**Q70: How is the low-resolution Grad-CAM heatmap mapped back to the input image?**
A70: It is upsampled (usually via bilinear interpolation) to match the dimensions of the original input image and overlaid as a heatmap.

**Q71: Can Grad-CAM be used to explain misclassifications?**
A71: Yes, by passing the *incorrectly predicted* class index to the Grad-CAM algorithm, we can visualize what parts of the image confused the model.

**Q72: How does the Global Average Pooling (GAP) layer interact with Grad-CAM?**
A72: Since GAP directly connects the last conv layer to the dense layer, the Grad-CAM weights are theoretically equivalent to the dense layer weights for a specific class (like CAM).

**Q73: What is the difference between standard CAM and Grad-CAM?**
A73: Standard CAM requires a specific architecture (GAP followed directly by a linear classifier), whereas Grad-CAM uses gradients and can be applied to any CNN architecture.

**Q74: Why is model interpretability important for a "Draw and Guess" application?**
A74: It builds user trust, allows developers to debug biases (e.g., the model only looks at wheels to predict a car), and improves the educational aspect of the game.

**Q75: What would a completely diffuse, random Grad-CAM heatmap indicate?**
A75: It indicates the model is guessing, has not learned localized features, or the object is entirely unidentifiable, often seen with random noise inputs or severely untrained models.

**Q76: How do you extract gradients using the Keras/Torch backend for Grad-CAM?**
A76: By creating a sub-model that outputs both the target conv layer activations and the final predictions, and using a GradientTape (or Torch `autograd`) to differentiate the predicted class score with respect to the conv outputs.

**Q77: What happens to the Grad-CAM heatmap if the SE block aggressively suppresses a channel?**
A77: That channel's corresponding spatial activation map will contribute very little to the final heatmap, reflecting the SE block's attention mechanism.

**Q78: Is Grad-CAM an architecture-specific or architecture-agnostic technique?**
A78: It is architecture-agnostic for CNNs, provided there is a spatial convolutional layer from which to extract gradients.

**Q79: How can Grad-CAM highlight potential dataset biases?**
A79: If the heatmap consistently highlights a watermark, background artifact, or a specific drawing corner instead of the main object, it indicates the model learned a dataset bias.

**Q80: What is the computational overhead of running Grad-CAM during inference?**
A80: It requires an additional backward pass (backpropagation) to compute gradients, making it strictly slower than a standard forward pass.

## Section 5: ML Engineering, ONNX & Deployment

**Q81: What is ONNX and why export the Keras model to it?**
A81: Open Neural Network Exchange (ONNX) is an open format built to represent machine learning models; exporting allows the model to be run on various frameworks and hardware, specifically enabling client-side inference in the browser.

**Q82: How does ONNX enable in-browser inference for this project?**
A82: The `.onnx` file can be loaded by ONNX Runtime Web, which uses WebGL or WebGPU to execute the model directly in the user's browser without needing a Python backend.

**Q83: What is the primary advantage of client-side (in-browser) inference for a drawing game?**
A83: Drastically reduced latency, zero server compute costs for inference, and the ability to work offline, as predictions happen instantly on the user's device.

**Q84: What happens to the Keras-specific operations (like SE blocks) during ONNX conversion?**
A84: They are translated into a standardized computational graph of fundamental mathematical operators (MatMul, Add, Sigmoid) supported by the ONNX specification.

**Q85: Why must you define `input_shape` and batch size dynamically or statically when converting to ONNX?**
A85: ONNX creates a static computation graph; defining the input signature ensures the runtime allocates the exact memory tensors needed for execution.

**Q86: What is Post-Training Quantization in the context of ONNX?**
A86: It is the process of converting the model's 32-bit floating-point weights (FP32) to 8-bit integers (INT8) after training, reducing model size and speeding up CPU inference.

**Q87: What is the trade-off of quantizing the model to INT8?**
A87: You gain a ~4x reduction in file size and faster inference, but you typically lose a small margin of classification accuracy due to the loss of precision in the weights.

**Q88: Why is model file size a critical metric for this project?**
A88: Because the model must be downloaded by the client's web browser over the network before the game can start; a smaller size means faster load times.

**Q89: Can the ONNX Runtime utilize the user's GPU?**
A89: Yes, through execution providers like WebGL, WebGPU (in browsers), or CUDA/TensorRT (on desktop environments), ONNX Runtime can offload operations to the GPU.

**Q90: What is the difference between ONNX tracing and scripting (in a PyTorch context)?**
A90: Tracing runs a sample input through the model and records the exact operations executed, while scripting parses the Python code to understand control flow (if/else loops) inside the model.

**Q91: If the converted ONNX model produces different outputs than the Keras model, what is the likely cause?**
A91: It is usually due to operator numerical disparities, missing opset implementations, or differences in how Batch Normalization or padding is handled between the frameworks.

**Q92: How do you verify the ONNX model is correct after conversion?**
A92: By running a consistent set of test images through both the original Keras model and the ONNX model using ONNX Runtime, and asserting that the output probability arrays match within a tiny tolerance (e.g., 1e-5).

**Q93: What is an ONNX "opset" version?**
A93: It is the version of the ONNX operator specifications; using a newer opset supports newer neural network layers, but requires a newer ONNX Runtime on the client side.

**Q94: How does the model handle the continuous stream of user drawing in a real-time web app?**
A94: The web frontend continuously rasterizes the current canvas state to a 64x64 bitmap at fixed time intervals (e.g., 200ms) and passes it to the ONNX model for real-time probability updates.

**Q95: Why decouple the training pipeline (Python) from the deployment pipeline (ONNX/JavaScript)?**
A95: Python is optimal for data processing, backpropagation, and utilizing multi-GPU setups, while JavaScript/WASM is optimal for universal, zero-install delivery to end-users.

**Q96: What preprocessing steps must be replicated in JavaScript before calling the ONNX model?**
A96: The exact same steps: resizing the canvas to 64x64, converting to grayscale, normalizing pixel values (0-1 or standardizing), and expanding dims to match the `[1, 1, 64, 64]` or `[1, 64, 64, 1]` input shape.

**Q97: If you wanted to support 1000 categories instead of 345, how would deployment be affected?**
A97: The final Dense layer's parameter count would increase significantly, enlarging the `.onnx` file size and slightly increasing inference time.

**Q98: What is Graph Optimization in ONNX Runtime?**
A98: It is an automatic process where ONNX Runtime fuses multiple operations (e.g., Conv + BatchNorm + ReLU) into a single optimized kernel to reduce memory bandwidth and increase speed.

**Q99: How does ONNX handle the channel ordering (NCHW vs NHWC)?**
A99: PyTorch typically uses NCHW, while Keras/TensorFlow use NHWC. During ONNX conversion and execution, one must explicitly ensure the input data channel format matches the exported graph's expectation.

**Q100: Describe a CI/CD pipeline for this model.**
A100: A GitHub Action triggers on new data/code, runs `01_prepare_data.py`, trains via `02_train_model.py`, evaluates via `03_evaluate_model.py`, exports via `04_convert_to_onnx.py`, tests the ONNX outputs, and deploys the `.onnx` artifact to a CDN.

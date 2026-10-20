# VIVA PREP GUIDE — Project 6: MNIST Handwritten Digit Classification
### Algorithm: Multi-Layer Perceptron (ANN) | Task: Multiclass Computer Vision Classification

This document is prepared specifically for your practical viva examination in Machine Learning Lab (CAI504C). It provides complete mathematical derivations, architecture walkthroughs, graphical interpretations, and code explanations.

---

## SECTION A — Core ML & Neural Network Foundations

### 1. Task Definition & Problem Category
- **Type:** Supervised Learning — Multiclass Computer Vision Image Classification.
- **Objective:** Map a 2D grayscale image matrix $I \in \mathbb{R}^{28 \times 28}$ to one of 10 mutually exclusive digit classes $y \in \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$.
- **Input Representation:** Flattened row vector $x \in \mathbb{R}^{784}$ where each coordinate $x_i$ represents pixel luminance ($0 = \text{black background}, 255 = \text{white stroke}$).

### 2. Multi-Layer Perceptron (MLP) Mathematical Mechanics
Our network has an input layer ($784$ units), two dense hidden layers ($128$ and $64$ units), and a softmax output layer ($10$ units):
1. **Hidden Layer 1:**
   $$z^{(1)} = W^{(1)} x + b^{(1)}, \quad a^{(1)} = \max(0, z^{(1)}) \quad (\text{ReLU Activation})$$
   *Why ReLU?* Solves the vanishing gradient problem inherent in Sigmoid/Tanh during backpropagation, accelerating gradient flow across hidden layers.
2. **Hidden Layer 2:**
   $$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}, \quad a^{(2)} = \max(0, z^{(2)})$$
3. **Softmax Output Layer:**
   $$z^{(3)} = W^{(3)} a^{(2)} + b^{(3)}$$
   $$\hat{y}_k = P(y = k \mid x) = \frac{e^{z_k^{(3)}}}{\sum_{j=0}^9 e^{z_j^{(3)}}}$$
   Softmax guarantees all output probabilities are strictly positive and sum to exactly $1.0$ ($\sum_{k=0}^9 \hat{y}_k = 1$).

### 3. Loss Function: Categorical Cross-Entropy
For multiclass classification with one-hot encoded ground truth $y$, the loss is:
$$\mathcal{L}(W, b) = -\sum_{k=0}^9 y_k \log \hat{y}_k = -\log \hat{y}_{\text{true\_class}}$$
The derivative with respect to pre-activation logits $z^{(3)}$ simplifies to:
$$\frac{\partial \mathcal{L}}{\partial z_k^{(3)}} = \hat{y}_k - y_k$$

### 4. Adam Optimizer (Adaptive Moment Estimation)
Rather than a constant learning rate $\eta$, Adam maintains running exponential averages of both the gradients ($m_t$) and squared gradients ($v_t$):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
With bias correction:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## SECTION B — Dataset Architecture & Splits

- **Raw Data Source:** MNIST Handwritten Digit Database (Yann LeCun et al.).
- **Local Disk Path:** `data/raw/mnist.csv` (15,000 balanced images, 1,500 per digit).
- **3-Way Stratified Splitting:**
  - Training Set: 70% (10,500 images)
  - Validation Set: 15% (2,250 images)
  - Test Set: 15% (2,250 images)
- **Data Leakage Prevention:** `StandardScaler` is fitted *strictly* on `X_train`. Validation and test images are scaled using the training mean and standard deviation.

---

## SECTION C — Detailed Explanation of Every Graph

| Graph File | Visual Dimensions | What It Demonstrates / Viva Explanation |
|---|---|---|
| `sample_digits_grid.png` | 4x4 Grid of 28x28 Images | Visual proof of raw digit shapes. Demonstrates thickness, tilt, and cursive variability across handwritten digits. |
| `class_distribution.png` | X: Digits 0–9<br>Y: Image Count | Verifies balanced distribution across classes (1,500 images per digit). Proves that standard multi-class accuracy is a reliable metric. |
| `confusion_matrix.png` | 10x10 Matrix | Heatmap of actual vs predicted classes. Shows that the primary classification confusions occur between visually ambiguous pairs: 4 vs 9, 3 vs 8, and 7 vs 1. |
| `roc_curves_multiclass.png` | X: FPR<br>Y: TPR | 10 One-vs-Rest ROC curves. Shows that all 10 digits achieve AUC $\ge 0.98$, with Digit 1 and Digit 0 achieving the highest discrimination. |
| `precision_recall_curves.png` | X: Recall<br>Y: Precision | One-vs-Rest Precision-Recall curves illustrating that high precision is maintained even as recall is swept to 100%. |
| `learning_curve.png` | X: Training sample count<br>Y: Accuracy | Depicts the training score starting high and cross-validation accuracy ascending steadily toward ~94%, showing smooth neural network convergence. |
| `cross_validation.png` | X: Fold (1 to 5)<br>Y: Accuracy | 5-Fold cross-validation bar chart confirming performance consistency across distinct training subsets. |
| `predictions_gallery.png` | 4x4 Grid of Test Samples | Compares actual digit against predicted digit on unseen test examples, with green titles for correct classifications and red for errors. |

---

## SECTION D — Line-by-Line Code Walkthrough

### 1. `src/data/make_dataset.py`
- Fetches `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` directly from the official repository.
- Extracts binary buffers with `np.frombuffer(...)`, skipping the 16-byte magic number header for images and 8-byte header for labels.
- Samples a balanced set of 15,000 images and saves to `data/raw/mnist.csv`.

### 2. `src/features/build_features.py`
- `train_test_split(..., test_size=0.30, stratify=y)`: Preserves class proportions across 10 classes.
- `StandardScaler()`: Normalizes pixel features to zero mean and unit variance.
- `joblib.dump(scaler, 'models/scaler.joblib')`: Stores scaler parameters for production inference.

### 3. `src/models/train_model.py`
- `MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=30)`: Defines neural architecture.
- `model.fit(X_train, y_train)`: Executes forward propagation, cross-entropy loss computation, and backpropagation weight updates.
- Computes validation and test metrics and writes `accuracy.txt`.

### 4. `src/visualization/visualize.py`
- Renders images by reshaping flattened 784-length arrays into $28 \times 28$ matrices using `np.reshape(28, 28)` and `plt.imshow(..., cmap='gray')`.
- Plots confusion matrix, multi-class ROC curves, and test prediction galleries.

### 5. `web/app.py` & `web/static/js/script.js`
- Serves an interactive HTML5 canvas.
- Captures mouse/touch draw coordinates, downsamples canvas to $28 \times 28$, extracts pixel intensities, and posts to `/predict`.
- Evaluates with `model.predict_proba()` and dynamically renders probability bars.

---

## SECTION E — Likely Viva Voce Questions & Answers

**Q1: How does a Multi-Layer Perceptron classify an image?**
> **Answer:** The $28 \times 28$ 2D image matrix is flattened into a 1D vector of 784 numerical pixel values. The input vector is multiplied by weight matrix $W^{(1)}$, biased by $b^{(1)}$, and activated by the non-linear ReLU function to produce 128 hidden feature representations. These are similarly transformed by a second hidden layer of 64 units. Finally, the 64 hidden representations are projected through a 10-unit output layer with Softmax activation, which produces a normalized probability distribution over digits 0 through 9.

**Q2: Why use the ReLU activation function instead of Sigmoid in the hidden layers?**
> **Answer:** The Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ saturates for large positive or negative inputs, causing its derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ to approach 0. During backpropagation, multiplying many small derivatives causes the gradient to vanish exponentially (the **vanishing gradient problem**), preventing deep layers from learning. In contrast, ReLU's derivative is $\frac{d}{dz}\max(0, z) = 1$ for all $z > 0$, allowing gradients to flow back without attenuation.

**Q3: How does Softmax differ from Sigmoid?**
> **Answer:** Sigmoid is used for binary classification and treats each output independently: $\sigma(z_i) \in [0, 1]$. Softmax is a multiclass generalization that couples all output units by dividing each exponentiated logit by the sum of all exponentiated logits: $\hat{y}_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$. This enforces a competitive probability distribution where increasing the score of one digit directly lowers the probabilities of all competing digits, and the sum across all classes is guaranteed to be 1.

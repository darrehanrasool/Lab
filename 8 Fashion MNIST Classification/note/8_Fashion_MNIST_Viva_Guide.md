# VIVA PREP GUIDE — Project 8: Fashion-MNIST Classification
### Algorithm: Multi-Layer Perceptron (ANN) | Task: Multiclass Computer Vision Classification

This document is prepared specifically for your practical viva examination in Machine Learning Lab (CAI504C). It provides complete mathematical derivations, architecture walkthroughs, graphical interpretations, and code walkthroughs.

---

## SECTION A — Core ML & Neural Network Foundations

### 1. Task Definition & Why Fashion-MNIST Exists
- **Type:** Supervised Learning — Multiclass Computer Vision Image Classification.
- **Objective:** Map a $28 \times 28$ grayscale pixel image to one of 10 clothing articles: `['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`.
- **Why Fashion-MNIST?** The original MNIST handwritten digit dataset is considered "too easy" for modern machine learning (even simple linear models achieve >92% and CNNs reach >99.7%). Zalando Research created Fashion-MNIST with identical dimensions ($28 \times 28$ grayscale, 10 classes) to provide a more realistic benchmark exhibiting high intra-class variance, complex fabric textures, silhouette ambiguities, and 3D folding variations.

### 2. Multi-Layer Perceptron (MLP) Mathematical Mechanics
1. **Forward Propagation:**
   - Input $x \in \mathbb{R}^{784}$ (flattened $28 \times 28$ pixel vector).
   - Dense Layer 1: $z^{(1)} = W^{(1)} x + b^{(1)}, \quad a^{(1)} = \max(0, z^{(1)})$ (128 units).
   - Dense Layer 2: $z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}, \quad a^{(2)} = \max(0, z^{(2)})$ (64 units).
   - Softmax Layer: $z^{(3)} = W^{(3)} a^{(2)} + b^{(3)}, \quad \hat{y}_k = \frac{e^{z_k^{(3)}}}{\sum_{j=0}^9 e^{z_j^{(3)}}}$ (10 class probabilities).
2. **Backpropagation with Adam Optimizer:**
   $$\nabla_{W^{(l)}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot (a^{(l-1)})^T$$
   The Adam optimizer calculates exponentially decaying averages of past gradients ($m_t$) and squared gradients ($v_t$), adapting individual parameter learning rates to scale down frequent gradient directions and accelerate infrequent ones.

### 3. Classification Challenges in Fashion-MNIST
Unlike handwritten digits where strokes are sparse against black backgrounds, clothing items occupy dense, solid regions with subtle boundary contours.
- **High-Ambiguity Pairs:**
  - `Shirt` vs `T-shirt/top` vs `Coat`: All share upper-torso silhouettes; classification depends on subtle collar boundaries and button textures.
  - `Sneaker` vs `Ankle boot`: Low-top vs high-top shoe height variations.
- **Low-Ambiguity Pairs:**
  - `Trouser`: Distinct dual-leg vertical geometry (F1-score ~98.3%).
  - `Bag`: Distinct rectangular shape and handle geometry (F1-score ~96.2%).

---

## SECTION B — Dataset Architecture & Splits

- **Raw Data Source:** Fashion-MNIST (Zalando Research).
- **Physical Disk Path:** `data/raw/fashion_mnist.csv` (15,000 balanced images, 1,500 per category).
- **3-Way Stratified Splitting:**
  - Training Set: 70% (10,500 images)
  - Validation Set: 15% (2,250 images)
  - Test Set: 15% (2,250 images)
- **Data Leakage Prevention:** `StandardScaler` is fitted *strictly* on `X_train`. Validation and test images are scaled using the training mean and standard deviation.

---

## SECTION C — Detailed Explanation of Every Graph

| Graph File | Visual Dimensions | What It Demonstrates / Viva Explanation |
|---|---|---|
| `fashion_samples_gallery.png` | 4x4 Grid of 28x28 Images | Visual proof of clothing silhouettes. Demonstrates structural differences between footwear, bags, trousers, and upper-body garments. |
| `class_distribution.png` | X: 10 Fashion Classes<br>Y: Count | Confirms equal class representation (1,500 samples per class). Proves that standard multiclass accuracy is an uncompromised metric. |
| `confusion_matrix.png` | 10x10 Matrix | Heatmap of actual vs predicted categories. Clearly visualizes that the lowest precision occurs in the `Shirt` class (misclassified as `T-shirt` or `Coat`), while `Trouser` and `Bag` have near-zero confusion. |
| `roc_curves_multiclass.png` | X: FPR<br>Y: TPR | 10 One-vs-Rest ROC curves. Shows that distinct items like Trousers, Bags, and Ankle Boots achieve AUC $> 0.98$, while subtle tops achieve AUC $\sim 0.94$. |
| `precision_recall_curves.png` | X: Recall<br>Y: Precision | One-vs-Rest Precision-Recall curves illustrating trade-offs across all 10 garment classes. |
| `learning_curve.png` | X: Training sample count<br>Y: Accuracy | Shows training score and cross-validation accuracy converging to ~86%, proving that the network learns stable, generalizable features. |
| `cross_validation.png` | X: K-Fold (1 to 5)<br>Y: Accuracy | 5-Fold cross-validation bar chart confirming performance consistency across different data subsets. |
| `predictions_gallery.png` | 4x4 Grid of Test Samples | Compares actual clothing item against predicted item on unseen test examples, with green titles for correct classifications and red for errors. |

---

## SECTION D — Line-by-Line Code Walkthrough

### 1. `src/data/make_dataset.py`
- Downloads `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` directly from the official Zalando S3 repository.
- Extracts binary buffers with `np.frombuffer(...)`, skipping the 16-byte magic number header for images and 8-byte header for labels.
- Samples a balanced set of 15,000 images and saves to `data/raw/fashion_mnist.csv`.

### 2. `src/features/build_features.py`
- `train_test_split(..., test_size=0.30, stratify=y)`: Preserves class balance across all 10 garment categories.
- `StandardScaler()`: Normalizes pixel features to zero mean and unit variance.
- `joblib.dump(scaler, 'models/scaler.joblib')`: Stores scaler parameters for production inference.

### 3. `src/models/train_model.py`
- `MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=30)`: Defines neural architecture.
- `model.fit(X_train, y_train)`: Executes forward propagation, cross-entropy loss computation, and backpropagation weight updates.
- Evaluates test accuracy and writes `accuracy.txt`.

### 4. `src/visualization/visualize.py`
- Plots $4 \times 4$ sample galleries, confusion matrix, multi-class ROC and PR curves, and prediction grids.

### 5. `web/app.py` & `web/static/js/script.js`
- Serves an interactive HTML5 canvas.
- Captures mouse/touch draw coordinates, downsamples canvas to $28 \times 28$, extracts pixel intensities, and posts to `/predict`.
- Evaluates with `model.predict_proba()` and dynamically renders probability bars.

---

## SECTION E — Likely Viva Voce Questions & Answers

**Q1: Why is accuracy on Fashion-MNIST (~86%) lower than on digit MNIST (~94-98%) using the same architecture?**
> **Answer:** Digit MNIST consists of simple, sparse strokes against a pure black background. In contrast, Fashion-MNIST images represent real physical objects with complex textures, solid fabric fills, multiple neckline variations, and folds. Furthermore, clothing classes have severe visual overlap (for instance, a short-sleeve button-down `Shirt` and a `T-shirt/top` share almost identical pixel bounding boxes and silhouettes), creating fundamental Bayes error limits that make Fashion-MNIST significantly more challenging.

**Q2: Which clothing class is most easily identified by the network, and which is the hardest?**
> **Answer:** As shown in our confusion matrix and classification report:
> - **Easiest:** `Trouser` (Class 1) with an F1-score of **98.25%** and `Bag` (Class 8) with an F1-score of **96.15%**. Both have unique geometric silhouettes (two long legs or a distinct rectangular shape with handles) that rarely overlap with other garments.
> - **Hardest:** `Shirt` (Class 6) with an F1-score of **63.48%**. Shirts are frequently confused with `T-shirt/top`, `Coat`, and `Pullover` because all four are upper-body garments with similar overall silhouettes.

**Q3: What role does the activation function play in the hidden layers of this network?**
> **Answer:** Without non-linear activation functions, stacking multiple dense layers would simply result in a single linear transformation ($W^{(2)}(W^{(1)}x + b^{(1)}) + b^{(2)} = W' x + b'$), which cannot learn complex non-linear decision boundaries. The ReLU activation function ($\max(0, z)$) introduces non-linearity, allowing the network to act as a universal function approximator capable of partitioning the complex, non-linear 784-dimensional feature space.

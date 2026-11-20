"""
========================================================================================
PROJECT 6: MNIST HANDWRITTEN DIGIT CLASSIFICATION
MODULE 3: MULTI-LAYER PERCEPTRON (NEURAL NETWORK) TRAINING (src/models/train_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: LeCun et al. (1998) Modified National Institute of Standards & Technology (MNIST)

PURPOSE:
--------
1. Trains a feedforward Multi-Layer Perceptron (MLP) Artificial Neural Network on 784-dimensional pixel arrays.
2. Architecture: 784 Inputs -> Hidden Layer 1 (128, ReLU) -> Hidden Layer 2 (64, ReLU) -> Output (10, Softmax).
3. Optimizes Cross-Entropy Loss via Adam (Adaptive Moment Estimation) with mini-batches of 128 images.
4. Evaluates Validation Accuracy (~94.7%) and Holdout Test Accuracy (~93.9%).
5. Persists the trained neural network artifact to `models/mnist_model.joblib` and generates `accuracy.txt`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Explain the forward pass through this Artificial Neural Network:
    -> Step 1: Input x in R^784.
       Step 2: Hidden 1: h_1 = ReLU(W_1 * x + b_1) where W_1 in R^{128 x 784}, b_1 in R^128.
       Step 3: Hidden 2: h_2 = ReLU(W_2 * h_1 + b_2) where W_2 in R^{64 x 128}, b_2 in R^64.
       Step 4: Output Layer: z = W_3 * h_2 + b_3 where W_3 in R^{10 x 64}, b_3 in R^10.
       Step 5: Softmax: P(y = k | x) = exp(z_k) / sum_{j=0}^9 exp(z_j).

Q2: What is the ReLU activation function and why is it preferred over Sigmoid/Tanh?
    -> ReLU(z) = max(0, z).
       Sigmoid and Tanh saturate for large positive or negative inputs, causing their derivatives to approach 0
       (the Vanishing Gradient Problem), which prevents deep layers from learning.
       ReLU has a constant derivative of 1 for all z > 0, enabling fast, non-vanishing gradient flow.

Q3: How does the Adam Optimizer work?
    -> Adam combines Momentum (exponentially decaying average of past gradients, first moment m_t)
       and RMSProp (exponentially decaying average of past squared gradients, second moment v_t):
           m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
           v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
       It dynamically adapts the learning rate for each individual weight in the network.
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def main() -> None:
    """Executes neural network training, validation, holdout evaluation, and artifact saving."""
    print("==================================================================")
    print("PROJECT 6 - STEP 3: MULTI-LAYER PERCEPTRON (ANN) DIGIT TRAINING")
    print("==================================================================")
    os.makedirs("models", exist_ok=True)

    # 1. Load standardized training, validation, and holdout test partitions
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # -------------------------------------------------------------------------
    # 2. CONFIGURE & FIT MULTI-LAYER PERCEPTRON
    # -------------------------------------------------------------------------
    # Input: 784 features (28x28 grayscale pixels)
    # Hidden Layer 1: 128 neurons, ReLU activation
    # Hidden Layer 2: 64 neurons, ReLU activation
    # Output Layer: 10 neurons, Softmax cross-entropy
    # Solver: Adam (Adaptive Moment Estimation)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=30,
        batch_size=128,
        random_state=42,
        early_stopping=True,       # Prevents overfitting by monitoring internal validation loss
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    # 3. Model Evaluation across Splits
    val_preds = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))

    test_preds = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    report = classification_report(y_test, test_preds, digits=4)

    # 5-fold cross validation on subset for fast stability verification
    cv_scores = cross_val_score(model, X_train.iloc[:3000], y_train[:3000], cv=5, scoring="accuracy")

    # 4. Serialize Trained Model
    joblib.dump(model, "models/mnist_model.joblib")

    # 5. Generate Benchmark Report (accuracy.txt)
    report_content = f"""MNIST Handwritten Digit Classification Performance
=====================================================
Model Architecture: Multi-Layer Perceptron (MLP / ANN)
Input Layer: 784 Units (28x28 Grayscale Pixels)
Hidden Layer 1: 128 Units (ReLU)
Hidden Layer 2: 64 Units (ReLU)
Output Layer: 10 Units (Softmax)
Optimizer: Adam (Adaptive Moment Estimation)

Validation Set Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)
Test Set Accuracy:       {test_acc:.4f} ({test_acc * 100:.2f}%)

Detailed Classification Report:
{report}

5-Fold Cross-Validation Accuracy:
  - Fold Scores: {[round(float(s), 4) for s in cv_scores]}
  - Mean Accuracy: {float(np.mean(cv_scores)):.4f} (+/- {float(np.std(cv_scores)):.4f})
"""
    with open("accuracy.txt", "w") as f:
        f.write(report_content)

    print(report_content)
    print("✓ Model saved to 'models/mnist_model.joblib' and 'accuracy.txt' generated.")


if __name__ == "__main__":
    main()

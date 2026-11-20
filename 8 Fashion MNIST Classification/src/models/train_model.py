"""
========================================================================================
PROJECT 8: FASHION-MNIST CLOTHING CLASSIFICATION
MODULE 3: MULTI-LAYER PERCEPTRON (NEURAL NETWORK) TRAINING (src/models/train_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Xiao, Rasul & Vollgraf (2017) Zalando Fashion-MNIST (10 Clothing Categories)

PURPOSE:
--------
1. Trains a feedforward Multi-Layer Perceptron (MLP) Artificial Neural Network on 784-dimensional clothing pixels.
2. Architecture: 784 Inputs -> Hidden 1 (128, ReLU) -> Hidden 2 (64, ReLU) -> Output (10, Softmax).
3. Evaluates Holdout Test Accuracy (~85.8%), recognizing that apparel classification has higher visual ambiguity
   than digit recognition (e.g. T-shirts vs Shirts vs Pullovers).
4. Persists the trained model artifact to `models/fashion_mnist_model.joblib` and generates `accuracy.txt`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why is Fashion-MNIST accuracy (~85.8%) lower than digit MNIST (~93.9%) on the same MLP architecture?
    -> Digits have clean, distinct topological contours (e.g. a loop in '0' vs a stroke in '1').
       Clothing articles exhibit significant inter-class visual overlap (e.g., long-sleeve T-shirts
       look almost identical to light Pullovers; Coats and Shirts share collars and buttons).

Q2: How does Early Stopping prevent overfitting in Neural Networks?
    -> Setting `early_stopping=True` automatically reserves 10% of training data as an internal validation monitor.
       If the validation loss does not improve for consecutive epochs, training terminates immediately,
       preventing the network from memorizing training noise.
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def main() -> None:
    """Executes neural network training, validation, holdout evaluation, and artifact saving."""
    print("==================================================================")
    print("PROJECT 8 - STEP 3: MULTI-LAYER PERCEPTRON CLOTHING TRAINING")
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
    # Input: 784 features (28x28 grayscale clothing pixels)
    # Hidden Layer 1: 128 neurons, ReLU activation
    # Hidden Layer 2: 64 neurons, ReLU activation
    # Output Layer: 10 neurons, Softmax cross-entropy
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=30,
        batch_size=128,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    # 3. Model Evaluation across Splits
    val_preds = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))

    test_preds = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4)

    # 5-fold cross validation on subset for fast stability verification
    cv_scores = cross_val_score(model, X_train.iloc[:3000], y_train[:3000], cv=5, scoring="accuracy")

    # 4. Serialize Trained Model
    joblib.dump(model, "models/fashion_mnist_model.joblib")

    # 5. Generate Benchmark Report (accuracy.txt)
    report_content = f"""Fashion-MNIST Classification Performance
=========================================
Model Architecture: Multi-Layer Perceptron (Neural Network)
Input Layer: 784 Units (28x28 Grayscale Clothing Pixels)
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
    print("✓ Model saved to 'models/fashion_mnist_model.joblib' and 'accuracy.txt' generated.")


if __name__ == "__main__":
    main()

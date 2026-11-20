"""
========================================================================================
PROJECT 6: MNIST HANDWRITTEN DIGIT CLASSIFICATION
MODULE 2: STRATIFIED SPLIT & PIXEL STANDARDIZATION (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: LeCun et al. (1998) Modified National Institute of Standards & Technology (MNIST)

PURPOSE:
--------
1. Separates 784 grayscale pixel features from the discrete digit label (y).
2. Partitions into 70% Train (10,500), 15% Validation (2,250), and 15% Test (2,250) with stratification.
3. Enforces Zero Data Leakage: StandardScaler is fitted strictly on `X_train`.
4. Serializes processed matrices and the scaler artifact for neural network training and web serving.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why scale pixel values for an Artificial Neural Network (MLP)?
    -> Raw pixel values range from 0 to 255. Passing large inputs through neural network layers
       pushes activations into saturated regions of activation functions (causing vanishing/exploding gradients)
       and destabilizes weight updates. Standardizing (mu=0, sigma=1) ensures smooth gradient propagation.

Q2: Why use stratified sampling?
    -> Ensures every single digit (0 through 9) has an identical 10% representation in Train,
       Validation, and Holdout Test subsets, preventing class bias during evaluation.

Q3: Why fit the scaler ONLY on X_train?
    -> Calculating mean and standard deviation across the entire dataset leaks test-set image statistics
       into the training pipeline. Fitting on `X_train` ensures valid out-of-sample evaluation.
========================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Orchestrates stratified partitioning and zero-leakage pixel standardization."""
    print("==================================================================")
    print("PROJECT 6 - STEP 2: STRATIFIED PARTITIONING & PIXEL SCALING")
    print("==================================================================")

    # 1. Load clean raw MNIST dataset from Step 1
    raw_path = "data/raw/mnist.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run 'python3 src/data/make_dataset.py' first!")

    df = pd.read_csv(raw_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    pixel_cols = list(X.columns)

    # 2. Stratified 3-Way Partitioning (70% Train, 15% Val, 15% Test)
    # Stage 1: 70% Train (10,500) vs 30% Temp (4,500)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Stage 2: 30% Temp (4,500) -> 15% Val (2,250) and 15% Test (2,250)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"✓ Stratified Split Complete:")
    print(f"  - Training Set:   {X_train.shape[0]} images (70%, 1,050 per digit)")
    print(f"  - Validation Set: {X_val.shape[0]} images (15%, 225 per digit)")
    print(f"  - Test Set:       {X_test.shape[0]} images (15%, 225 per digit)")

    # 3. Zero-Data-Leakage Feature Standardization (Z-score: mu=0, sigma=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Compute mu and sigma on train only
    X_val_scaled = scaler.transform(X_val)          # Transform validation using train parameters
    X_test_scaled = scaler.transform(X_test)        # Transform test with zero leakage

    # 4. Serialize artifacts to disk
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "data/processed/scaler.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    pd.DataFrame(X_train_scaled, columns=pixel_cols).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_val_scaled, columns=pixel_cols).to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=pixel_cols).to_csv("data/processed/X_test.csv", index=False)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✓ Successfully saved processed CSVs and StandardScaler to 'data/processed/' and 'models/'.")


if __name__ == "__main__":
    main()

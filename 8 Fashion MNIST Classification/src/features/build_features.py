"""
========================================================================================
PROJECT 8: FASHION-MNIST CLOTHING CLASSIFICATION
MODULE 2: STRATIFIED SPLIT & PIXEL STANDARDIZATION (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Xiao, Rasul & Vollgraf (2017) Zalando Fashion-MNIST (10 Clothing Categories)

PURPOSE:
--------
1. Separates 784 flattened clothing pixel features from the categorical label (y).
2. Partitions into 70% Train (10,500), 15% Validation (2,250), and 15% Test (2,250) with stratification.
3. Enforces Zero Data Leakage: StandardScaler is fitted strictly on `X_train`.
4. Serializes processed matrices and the scaler artifact for neural network training and web serving.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why must pixel intensities be scaled?
    -> Unscaled inputs (0-255) cause severe non-convex gradients and gradient explosions during
       backpropagation in multi-layer neural networks. Standardizing (z = (x - mu) / sigma) centers
       the image distribution around 0, stabilizing Adam optimizer updates.

Q2: Why use stratified sampling?
    -> Guarantees that every one of the 10 fashion categories (from T-shirts to Ankle boots)
       maintains an exact 10% proportion in the training, validation, and holdout test partitions.
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
    print("PROJECT 8 - STEP 2: STRATIFIED SPLIT & PIXEL STANDARDIZATION")
    print("==================================================================")

    # 1. Load clean raw Fashion-MNIST data from Step 1
    raw_path = "data/raw/fashion_mnist.csv"
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
    print(f"  - Training Set:   {X_train.shape[0]} images (70%, 1,050 per category)")
    print(f"  - Validation Set: {X_val.shape[0]} images (15%, 225 per category)")
    print(f"  - Test Set:       {X_test.shape[0]} images (15%, 225 per category)")

    # 3. Zero-Data-Leakage Standardization
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

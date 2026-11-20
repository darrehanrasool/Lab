"""
========================================================================================
PROJECT 1: IRIS SPECIES CLASSIFICATION
MODULE 2: FEATURE PREPROCESSING, STRATIFICATION & SCALING (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Fisher's Famous 1936 Iris Flower Dataset (N = 150 Samples, 3 Classes)

PURPOSE:
--------
1. Splits data into 70% Train (105), 15% Validation (22), and 15% Test (23).
2. Applies Stratified Sampling (`stratify=y`) to maintain equal class proportions in each split.
3. Enforces Zero Data Leakage: StandardScaler is fitted strictly on `X_train`.
4. Serializes scaled matrices and the scaler artifact for downstream model training and web serving.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why use stratified splitting (`stratify=y`)?
    -> The Iris dataset has exactly 50 samples per class. If we performed a naive random split,
       one split might receive too many Setosa and too few Virginica samples by pure chance.
       Stratification ensures every subset contains an exact ~33.3% representation of each species.

Q2: Why scale features for Logistic Regression?
    -> Petal Length (~1.0 to 6.9 cm) has a much larger variance and absolute magnitude than
       Petal Width (~0.1 to 2.5 cm). Without standardization (z = (x - mu) / sigma), gradient-based
       solvers (like L-BFGS) would oscillate, and the L2 regularization penalty would unfairly penalize
       features with larger scales.

Q3: Why fit the scaler ONLY on X_train?
    -> Standardizing test data using the global mean causes Data Leakage (information from the test
       set leaks into the training pipeline). Fitting exclusively on `X_train` ensures realistic evaluation.
========================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Orchestrates feature extraction, stratified partitioning, and zero-leakage scaling."""
    print("==================================================================")
    print("PROJECT 1 - STEP 2: STRATIFIED SPLIT & ZERO-LEAKAGE STANDARDIZATION")
    print("==================================================================")

    # 1. Load clean raw Iris data from Step 1
    raw_path = "data/raw/Iris.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run 'python3 src/data/make_dataset.py' first!")

    df = pd.read_csv(raw_path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    # 2. Stratified 3-Way Partitioning (70% Train, 15% Val, 15% Test)
    # Stage 1: 70% Train (105) vs 30% Temp (45)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Stage 2: 30% Temp (45) -> 15% Val (22) and 15% Test (23)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"✓ Stratified Split Complete:")
    print(f"  - Training Set:   {X_train.shape[0]} samples (70%)")
    print(f"  - Validation Set: {X_val.shape[0]} samples (15%)")
    print(f"  - Test Set:       {X_test.shape[0]} samples (15%)")

    # 3. Zero-Data-Leakage Standardization
    # Formula: z = (x - mu_train) / sigma_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Compute mu and sigma on training data ONLY
    X_val_scaled = scaler.transform(X_val)          # Transform using frozen training statistics
    X_test_scaled = scaler.transform(X_test)        # Transform without data leakage

    # 4. Serialize artifacts to disk
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save scaler artifact in both locations for web app compatibility
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(scaler, "data/processed/scaler.joblib")

    # Save processed datasets
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv("data/processed/X_test.csv", index=False)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✓ Saved all processed CSVs to 'data/processed/' and scaler to 'models/scaler.pkl'.")


if __name__ == "__main__":
    main()

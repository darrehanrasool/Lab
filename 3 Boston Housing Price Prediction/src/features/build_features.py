"""
========================================================================================
PROJECT 3: BOSTON HOUSING PRICE PREDICTION
MODULE 2: FEATURE PREPROCESSING, 70/20/10 SPLIT & SCALING (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Harrison & Rubinfeld (1978) Boston Housing Cohort (N = 506 Census Tracts)

PURPOSE:
--------
1. Separates feature predictors (X) from continuous target MEDV (y).
2. Partitions into 70% Train (354), 20% Validation (101), and 10% Holdout Test (51)
   to fulfill the professor's strict 70/20/10 split requirement.
3. Enforces Zero Data Leakage: StandardScaler is fitted strictly on `X_train`.
4. Serializes all scaled datasets and the scaler artifact for downstream training and web inference.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How was the 70/20/10 split mathematically executed?
    -> Stage 1: Split full 506 samples into 70% Train (354) and 30% Temp (152). (test_size = 0.30).
       Stage 2: Split 152 Temp into 2/3 Val (~101 tracts = 20% of total) and 1/3 Test (~51 tracts = 10% of total).
       (test_size = 1.0 / 3.0).

Q2: Why must features like TAX (range 187-711) and NOX (range 0.38-0.87) be scaled?
    -> In regression and SVR, unscaled features with large numerical magnitudes (like TAX)
       dominate gradient calculations and penalization terms compared to features on tiny decimal scales (NOX).
       Z-score standardization (z = (x - mu) / sigma) centers all features to mean 0 and standard deviation 1.

Q3: Why fit the scaler ONLY on X_train?
    -> Standardizing test data using test-set statistics constitutes Data Leakage. Frozen training parameters
       ensure that the model evaluates on genuinely unseen test distributions.
========================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Orchestrates 70/20/10 partitioning and zero-leakage feature standardization."""
    print("==================================================================")
    print("PROJECT 3 - STEP 2: 70/20/10 DATASET PARTITIONING & STANDARDIZATION")
    print("==================================================================")

    # 1. Load clean raw Boston Housing data from Step 1
    raw_path = "data/raw/boston.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run 'python3 src/data/make_dataset.py' first!")

    df = pd.read_csv(raw_path)
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    feature_names = list(X.columns)

    # 2. Strict 70% Train, 20% Validation, 10% Test Partitioning
    # Stage 1: 70% Train (354) vs 30% Temp (152)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Stage 2: 30% Temp (152) -> 20% Val (101) and 10% Test (51)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 / 3.0), random_state=42
    )

    print(f"✓ 70/20/10 Split Partition Complete:")
    print(f"  - Training Set:   {X_train.shape[0]} tracts (70%) | Mean Price: ${y_train.mean() * 1000:,.0f}")
    print(f"  - Validation Set: {X_val.shape[0]} tracts (20%) | Mean Price: ${y_val.mean() * 1000:,.0f}")
    print(f"  - Holdout Test:   {X_test.shape[0]} tracts (10%) | Mean Price: ${y_test.mean() * 1000:,.0f}")

    # 3. Zero-Data-Leakage Feature Standardization (Z-score: mu=0, sigma=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Compute mu and sigma SOLELY on X_train
    X_val_scaled = scaler.transform(X_val)          # Transform validation using training parameters
    X_test_scaled = scaler.transform(X_test)        # Transform holdout test with zero data leakage

    # 4. Serialize artifacts to disk
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "data/processed/scaler.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    pd.DataFrame(X_train_scaled, columns=feature_names).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_val_scaled, columns=feature_names).to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=feature_names).to_csv("data/processed/X_test.csv", index=False)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✓ Successfully saved processed CSVs and StandardScaler to 'data/processed/' and 'models/'.")


if __name__ == "__main__":
    main()

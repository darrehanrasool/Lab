"""
========================================================================================
PROJECT 2: TITANIC SURVIVAL PREDICTION
MODULE 2: FEATURE ENGINEERING, IMPUTATION & SCALING (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: RMS Titanic Passenger Manifest (N = 891 Samples, Binary Classification)

PURPOSE:
--------
1. Selects the 6 key demographic and socio-economic predictive features.
2. Imputes missing passenger values (Age and Fare) using training set medians to avoid data leakage.
3. Maps categorical gender to binary numerical indicator: {'male': 0, 'female': 1}.
4. Partitions dataset into 70% Train (623), 15% Validation (134), and 15% Test (134) with stratification.
5. Standardizes numerical features using StandardScaler fitted exclusively on training data.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why use median imputation instead of mean for Age and Fare?
    -> Ticket fares and passenger ages are right-skewed and contain extreme outliers (e.g., luxury suites > £500).
       The median is robust against extreme outliers, whereas the mean would artificially inflate imputed values.

Q2: Why is 'sex' mapped to {'male': 0, 'female': 1}?
    -> Historical 'women and children first' protocol gave female passengers a dramatic survival advantage
       (~74% survival for females vs ~19% for males). Coding female as 1 allows positive weights/splits to directly
       capture this survival boost.

Q3: Why use stratified sampling (`stratify=y`)?
    -> Survival is imbalanced (~61.6% perished, 38.4% survived). Stratified splitting ensures that
       Train, Validation, and Test subsets maintain the exact same 38.4% survival proportion.
========================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Orchestrates feature selection, median imputation, stratified splitting, and standardization."""
    print("==================================================================")
    print("PROJECT 2 - STEP 2: PREPROCESSING, IMPUTATION & STRATIFIED SPLIT")
    print("==================================================================")

    # 1. Load raw manifest data from Step 1
    raw_path = "data/raw/titanic.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run 'python3 src/data/make_dataset.py' first!")

    df = pd.read_csv(raw_path)

    # 2. Select predictive feature subset
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    df_subset = df[["survived"] + feature_cols].copy()

    # 3. Categorical encoding for gender
    df_subset["sex"] = df_subset["sex"].map({"male": 0, "female": 1})

    # 4. Stratified 3-Way Partitioning (70% Train, 15% Val, 15% Test)
    X = df_subset[feature_cols]
    y = df_subset["survived"]

    # Stage 1: 70% Train (623) vs 30% Temp (268)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Stage 2: 30% Temp (268) -> 15% Val (134) and 15% Test (134)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # 5. Zero-Data-Leakage Missing Value Imputation
    # Golden ML Rule: Compute median values SOLELY on X_train to prevent leakage
    train_age_median = X_train["age"].median()
    train_fare_median = X_train["fare"].median()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train["age"] = X_train["age"].fillna(train_age_median)
    X_train["fare"] = X_train["fare"].fillna(train_fare_median)

    X_val["age"] = X_val["age"].fillna(train_age_median)
    X_val["fare"] = X_val["fare"].fillna(train_fare_median)

    X_test["age"] = X_test["age"].fillna(train_age_median)
    X_test["fare"] = X_test["fare"].fillna(train_fare_median)

    print(f"✓ Imputed missing values using training medians (Age: {train_age_median:.1f} yrs, Fare: £{train_fare_median:.2f})")
    print(f"✓ Stratified Split Complete: Train={X_train.shape[0]} (70%), Val={X_val.shape[0]} (15%), Test={X_test.shape[0]} (15%)")

    # 6. Feature Standardization (Z-Score: mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn mu and sigma on train only
    X_val_scaled = scaler.transform(X_val)          # Transform validation using training parameters
    X_test_scaled = scaler.transform(X_test)        # Transform test with zero data leakage

    # 7. Serialize artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "data/processed/scaler.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv("data/processed/X_test.csv", index=False)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✓ Successfully saved processed CSVs and StandardScaler to 'data/processed/' and 'models/'.")


if __name__ == "__main__":
    main()

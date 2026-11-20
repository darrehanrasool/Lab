"""
========================================================================================
MODULE 2: FEATURE PREPROCESSING, IMBALANCE STRATIFICATION & SCALING (src/features/build_features.py)
========================================================================================
Project: Diabetes Progression Prediction via Multiple Linear Regression
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Efron et al. (2004) Authentic Clinical Diabetes Cohort (N = 442 Patients)

PURPOSE:
--------
Transforms raw clinical records into machine-learning-ready matrices:
1. Maps categorical demographic features to binary dummy variables with an interpretable baseline.
2. Manages target distribution imbalance (continuous right-skewness) via Quantile Stratification.
3. Partitions data into a 3-way split: 70% Training (N=309), 20% Validation (N=88), 10% Holdout Test (N=45).
4. Enforces STRICT ZERO DATA LEAKAGE: StandardScaler is fitted exclusively on the training split.
5. Serializes all scaled datasets, unscaled reference files, and the fitted scaler artifact.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why map 'sex' from {1.0, 2.0} to {0.0, 1.0}?
    -> If left as 1.0 and 2.0, an ordinary least squares regression treats 'sex' as a continuous
       numerical quantity where Male (2.0) has twice the magnitude of Female (1.0).
       Mapping to {0.0 = Female, 1.0 = Male} creates a proper binary dummy indicator.
       The resulting beta weight beta_sex directly represents the difference in expected
       1-year disease progression between male patients and the female reference group.

Q2: What is continuous target imbalance and why use quantile stratification?
    -> Continuous regression targets can suffer from severe tail sparsity. In this cohort,
       target progression is right-skewed (+0.44 skewness); severe cases (progression >= 200)
       comprise only ~28.7% of all patients. A naive random split risks placing too few severe
       patients in the small 10% test set (N=45). Using pd.qcut() partitions the continuous
       target into 5 equal-frequency quantile bins, enabling stratified splitting so that
       Train, Validation, and Test sets maintain identical distributions of mild, moderate,
       and severe progression.

Q3: What is Data Leakage and how is it prevented here?
    -> Data leakage occurs if statistical properties of the test/validation set (e.g. mean,
       variance, min, max) leak into the model training pipeline. If we fitted StandardScaler
       on the entire dataset before splitting, the test set's mean would influence training.
       Here, `scaler.fit_transform(X_train)` computes mu and sigma SOLELY on X_train.
       `X_val` and `X_test` are transformed using those frozen training parameters:
       z = (x - mu_train) / sigma_train.
========================================================================================
"""

import os
from typing import List, Tuple

import joblib  # Artifact serialization for scikit-learn objects
import pandas as pd
from sklearn.model_selection import train_test_split  # Deterministic dataset splitting
from sklearn.preprocessing import StandardScaler      # Z-score standardization (mu=0, sigma=1)


def process_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and encodes input features according to clinical and statistical best practices:
    
    1. Categorical / Demographic:
       - 'sex' is originally coded as 1.0 (Female) and 2.0 (Male).
       - We convert this to a binary dummy indicator:
           0.0 = Female (reference category)
           1.0 = Male
    2. Continuous Anthropometric & Physiological:
       - 'age' (years), 'bmi' (kg/m²), 'bp' (mean arterial pressure in mmHg).
    3. Blood Serum Lipids & Biomarkers:
       - 's1' (total cholesterol), 's2' (LDL), 's3' (HDL), 's4' (TCH ratio),
       - 's5' (log serum triglycerides - LTG), 's6' (fasting blood glucose - GLU).

    Args:
        df (pd.DataFrame): Raw clinical DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with binary sex encoding.
    """
    # Create an explicit deep copy to avoid modifying original in-memory data
    df_clean: pd.DataFrame = df.copy()

    # Map biological sex: Female (1.0) -> 0.0, Male (2.0) -> 1.0
    df_clean["sex"] = df_clean["sex"].map({1.0: 0.0, 2.0: 1.0})

    return df_clean


def perform_stratified_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits data into 70% Train, 20% Validation, and 10% Test using quantile stratification:
    
    Total N = 442:
      - Train (70%): 309 patients (for OLS regression weight fitting)
      - Val   (20%):  88 patients (for independent parameter validation)
      - Test  (10%):  45 patients (for final holdout generalization evaluation)
    
    Why two stages?
      Stage 1: 442 patients -> 309 train (70%) and 133 temp (30%). Test ratio = 133 / 442.
      Stage 2: 133 temp patients -> 88 val (20% of 442) and 45 test (10% of 442).
               Test ratio = 45 / 133 = 0.33834586...

    Args:
        X (pd.DataFrame): Feature matrix (10 clinical biomarkers).
        y (pd.Series): Continuous target progression values.

    Returns:
        Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Step 1: Discretize continuous progression into 5 quantile bins for stratification
    progression_quantiles = pd.qcut(y, q=5, labels=False)

    # Step 2: First split: 70% Train (309) vs 30% Temporary (133)
    X_train, X_temp, y_train, y_temp, _, q_temp = train_test_split(
        X,
        y,
        progression_quantiles,
        test_size=133 / 442,
        random_state=42,          # Fixed seed ensures 100% reproducible splits
        stratify=progression_quantiles,  # Enforces identical target distribution in both subsets
    )

    # Step 3: Second split: 133 Temporary -> 88 Validation (20%) and 45 Test (10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=45 / 133,
        random_state=42,
        stratify=q_temp,          # Preserves quantile balance across validation and test
    )

    # Sanity checks on exact split sizes
    assert len(X_train) == 309, f"Partition Error: Expected 309 train samples, got {len(X_train)}"
    assert len(X_val) == 88, f"Partition Error: Expected 88 val samples, got {len(X_val)}"
    assert len(X_test) == 45, f"Partition Error: Expected 45 test samples, got {len(X_test)}"

    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    """
    Orchestrates Step 2 feature engineering, stratified splitting, and standardization.
    """
    print("==================================================================")
    print("STEP 2: FEATURE PREPROCESSING, IMBALANCE STRATIFICATION & SPLIT")
    print("==================================================================")

    # 1. Load raw clinical data from Step 1
    raw_path: str = "data/raw/diabetes.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Missing '{raw_path}'. Please run 'python3 src/data/make_dataset.py' first!"
        )

    df_raw: pd.DataFrame = pd.read_csv(raw_path)
    print(f"✓ Loaded raw dataset: {df_raw.shape[0]} patients, {df_raw.shape[1]} columns.")

    # 2. Process features (encode categorical sex as binary indicator [0=Female, 1=Male])
    df_processed: pd.DataFrame = process_input_features(df_raw)
    print("✓ Encoded 'sex' as binary dummy variable [0.0 = Female, 1.0 = Male].")
    print("✓ Maintained authentic physical units for all 9 remaining clinical biomarkers.")

    # 3. Separate feature matrix (X) and continuous target vector (y)
    X: pd.DataFrame = df_processed.drop(columns=["progression"])
    y: pd.Series = df_processed["progression"]
    feature_names: List[str] = list(X.columns)

    # 4. Perform quantile-stratified 3-way split (70% Train, 20% Val, 10% Test)
    X_train, X_val, X_test, y_train, y_val, y_test = perform_stratified_split(X, y)

    print("\n✓ Stratified Dataset Partitioning Completed Successfully:")
    print(f"  - Training Cohort   (70%): {len(X_train)} patients | Mean y = {y_train.mean():.2f} (Std = {y_train.std():.2f})")
    print(f"  - Validation Cohort (20%):  {len(X_val)} patients | Mean y = {y_val.mean():.2f} (Std = {y_val.std():.2f})")
    print(f"  - Holdout Test Set  (10%):  {len(X_test)} patients | Mean y = {y_test.mean():.2f} (Std = {y_test.std():.2f})")
    print("  -> Quantile stratification guarantees balanced target distribution across all splits.")

    # -------------------------------------------------------------------------
    # 5. ZERO DATA LEAKAGE FEATURE STANDARDIZATION
    # -------------------------------------------------------------------------
    # Formula: z = (x - mu) / sigma
    # Crucial ML Rule: StandardScaler is fitted SOLELY on X_train.
    # X_val and X_test are transformed using X_train's frozen statistics.
    scaler: StandardScaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Computes mu_train, sigma_train and scales X_train
    X_val_scaled = scaler.transform(X_val)          # Uses mu_train, sigma_train (no re-fitting!)
    X_test_scaled = scaler.transform(X_test)        # Uses mu_train, sigma_train (zero data leakage!)

    # -------------------------------------------------------------------------
    # 6. ARTIFACT PERSISTENCE
    # -------------------------------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save fitted scaler for inference in web server and CLI predictor
    joblib.dump(scaler, "data/processed/scaler.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # Save unscaled copies (critical for clinical review, web presets, and feature explainability)
    X_train.to_csv("data/processed/X_train_unscaled.csv", index=False)
    X_val.to_csv("data/processed/X_val_unscaled.csv", index=False)
    X_test.to_csv("data/processed/X_test_unscaled.csv", index=False)

    # Save scaled copies (for model fitting, matrix operations, and regression evaluation)
    pd.DataFrame(X_train_scaled, columns=feature_names).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_val_scaled, columns=feature_names).to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=feature_names).to_csv("data/processed/X_test.csv", index=False)

    # Save target vectors
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("\n✓ Serialized all scaled/unscaled splits and StandardScaler to 'data/processed/' and 'models/'.")


if __name__ == "__main__":
    main()

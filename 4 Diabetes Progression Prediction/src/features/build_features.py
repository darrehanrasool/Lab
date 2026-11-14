import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HANDLING DIFFERENT INPUT FEATURE TYPES:
    1. Categorical / Demographic:
       - 'sex' is originally coded as 1.0 (Female) and 2.0 (Male).
       - Leaving it as 1.0 vs 2.0 implies an arbitrary continuous scale where Male is 'twice' Female.
       - We map it to binary indicator [0.0 = Female, 1.0 = Male] so the model coefficient
         directly reflects the expected difference in progression for male patients relative to female patients.
    2. Continuous Anthropometric & Physiological:
       - 'age' (years), 'bmi' (kg/m²), 'bp' (mean arterial mmHg).
    3. Continuous Blood Serum Biomarkers:
       - 's1' (total cholesterol), 's2' (LDL), 's3' (HDL), 's4' (TCH ratio),
         's5' (log serum triglycerides - already log-transformed), 's6' (blood glucose).
    """
    df = df.copy()
    # Map sex to clean 0/1 binary indicator
    df['sex'] = df['sex'].map({1.0: 0.0, 2.0: 1.0})
    return df

def main():
    print("==========================================================")
    print("STEP 2: FEATURE PREPROCESSING, IMBALANCE MANAGEMENT & SPLIT")
    print("==========================================================")
    
    raw_path = 'data/raw/diabetes.csv'
    df_raw = pd.read_csv(raw_path)
    print(f"Loaded raw dataset: {df_raw.shape[0]} patients.")
    
    # 1. Process Input Features (Handle categorical binary sex vs continuous vitals)
    df_processed = process_input_features(df_raw)
    print("✓ Handled input features: Transformed 'sex' to binary indicator [0=Female, 1=Male].")
    print("✓ Retained 10 authentic clinical biomarkers with full physical interpretability.")
    
    X = df_processed.drop(columns=['progression'])
    y = df_processed['progression']
    feature_names = list(X.columns)
    
    # -------------------------------------------------------------
    # 2. MANAGING DATASET IMBALANCE VIA QUANTILE-STRATIFIED SPLIT
    # -------------------------------------------------------------
    # The continuous target 'progression' is right-skewed (+0.44 skewness).
    # Severe progression cases (>200) constitute only ~28.7% of the cohort.
    # Naive random splitting risks severe imbalance in the 10% test set (45 samples).
    # We partition progression into 5 quantile bins to enforce proportional stratification.
    progression_quantiles = pd.qcut(y, q=5, labels=False)
    
    # First split: 70% Train (309) vs 30% Temp (133)
    X_train, X_temp, y_train, y_temp, q_train, q_temp = train_test_split(
        X, y, progression_quantiles, test_size=133/442, random_state=42, stratify=progression_quantiles
    )
    
    # Second split: 30% Temp (133) -> 20% Val (88) and 10% Test (45)
    # Ratio: 45 / 133 = 0.33834586
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=45/133, random_state=42, stratify=q_temp
    )
    
    assert len(X_train) == 309, f"Train size {len(X_train)} != 309"
    assert len(X_val) == 88,   f"Val size {len(X_val)} != 88"
    assert len(X_test) == 45,  f"Test size {len(X_test)} != 45"
    
    print("\n✓ Stratified Dataset Partitioning Complete:")
    print(f"  - Training Set (70%):   {len(X_train)} patients | Mean Target: {y_train.mean():.2f} (Std: {y_train.std():.2f})")
    print(f"  - Validation Set (20%): {len(X_val)} patients   | Mean Target: {y_val.mean():.2f} (Std: {y_val.std():.2f})")
    print(f"  - Test Set (10%):       {len(X_test)} patients  | Mean Target: {y_test.mean():.2f} (Std: {y_test.std():.2f})")
    print("  -> Quantile stratification guarantees balanced distribution across all splits.")
    
    # -------------------------------------------------------------
    # 3. ZERO DATA LEAKAGE FEATURE STANDARDIZATION
    # -------------------------------------------------------------
    # Why scale? Ridge penalty (alpha * sum(w_j^2)) penalizes all coefficients equally.
    # Features on large scales (e.g., cholesterol ~200) would unfairly dominate features
    # on smaller scales (e.g., log triglycerides ~4.5) without z-score standardization.
    # GOLDEN RULE: Scaler is fitted EXCLUSIVELY on X_train.
    # X_val and X_test are transformed using X_train statistics (mean, std).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    # Serialize artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(scaler, 'data/processed/scaler.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save unscaled copies (for clinical review & web interface testing)
    X_train.to_csv('data/processed/X_train_unscaled.csv', index=False)
    X_val.to_csv('data/processed/X_val_unscaled.csv', index=False)
    X_test.to_csv('data/processed/X_test_unscaled.csv', index=False)
    
    # Save scaled copies (for model training and evaluation)
    pd.DataFrame(X_train_scaled, columns=feature_names).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=feature_names).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_names).to_csv('data/processed/X_test.csv', index=False)
    
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("\n✓ Serialized all scaled/unscaled splits and StandardScaler to data/processed/ and models/.")

if __name__ == '__main__':
    main()

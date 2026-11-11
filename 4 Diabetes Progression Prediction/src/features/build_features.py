import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def enrich_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DATA ENRICHMENT:
    Synthesizes domain-specific biochemical and metabolic interaction markers:
    1. INSULIN_RESISTANCE_PROXY: BMI * S6 (Obesity index * Fasting Blood Glucose)
    2. ATHEROGENIC_RATIO: S2 / S3 (LDL / HDL - vascular damage ratio)
    3. METABOLIC_RISK_SCORE: (BMI * BP) / S3 (Classic metabolic syndrome triad)
    """
    df = df.copy()
    
    # 1. Insulin Resistance Proxy (Glucose * BMI interaction)
    df['insulin_res_proxy'] = df['bmi'] * df['s6']
    
    # 2. Atherogenic Index (LDL to HDL Ratio)
    # S3 is HDL (good cholesterol), S2 is LDL (bad cholesterol)
    df['atherogenic_ratio'] = df['s2'] / (df['s3'] + 1e-5)
    
    # 3. Metabolic Risk Triad (BMI * Blood Pressure / Protective HDL)
    df['metabolic_risk_triad'] = (df['bmi'] * df['bp']) / (df['s3'] + 1e-5)
    
    return df

def main():
    print("==========================================================")
    print("STEP 2: FEATURE ENGINEERING, ENRICHMENT & 70/20/10 SPLIT")
    print("==========================================================")
    
    df_raw = pd.read_csv('data/raw/diabetes.csv')
    print(f"Loaded raw dataset with {df_raw.shape[0]} patients.")
    
    # Apply Data Enrichment
    df_enriched = enrich_clinical_features(df_raw)
    print(f"Data enriched: Features expanded from {df_raw.shape[1]-1} to {df_enriched.shape[1]-1} (added 3 clinical interaction terms).")
    
    feature_cols = [c for c in df_enriched.columns if c != 'progression']
    X = df_enriched[feature_cols]
    y = df_enriched['progression']
    
    # -------------------------------------------------------------
    # EXACT 70% / 20% / 10% DATASET PARTITIONING
    # Total N = 442
    # 70% Train: 309 samples
    # 20% Val:    88 samples
    # 10% Test:   45 samples
    # -------------------------------------------------------------
    # First split: 70% Train (309) vs 30% Temp (133)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    
    # Second split: 30% Temp (133) -> 20% Val (88) and 10% Test (45)
    # Ratio: 45 / 133 = 0.33834586
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=45/133, random_state=42
    )
    
    assert X_train.shape[0] == 309, f"Train size {X_train.shape[0]} != 309"
    assert X_val.shape[0] == 88,   f"Val size {X_val.shape[0]} != 88"
    assert X_test.shape[0] == 45,  f"Test size {X_test.shape[0]} != 45"
    
    print(f"✓ Split Verification:")
    print(f"  - Training Set (70%):   {X_train.shape[0]} patients")
    print(f"  - Validation Set (20%): {X_val.shape[0]} patients")
    print(f"  - Test Set (10%):       {X_test.shape[0]} patients")
    print(f"  - Total:                {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} patients")
    
    # -------------------------------------------------------------
    # ZERO DATA LEAKAGE FEATURE STANDARDIZATION
    # Scaler is fitted EXCLUSIVELY on X_train.
    # X_val and X_test are transformed using X_train statistics (mean, std).
    # -------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    # Persist artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(scaler, 'data/processed/scaler.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save unscaled copies for clinical inspection & web testing
    X_train.to_csv('data/processed/X_train_unscaled.csv', index=False)
    X_val.to_csv('data/processed/X_val_unscaled.csv', index=False)
    X_test.to_csv('data/processed/X_test_unscaled.csv', index=False)
    
    # Save scaled copies for model training
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv('data/processed/X_test.csv', index=False)
    
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("✓ All processed datasets and StandardScaler serialized to data/processed/ and models/.")

if __name__ == '__main__':
    main()

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("Building features for Diabetes Progression Prediction...")
    df = pd.read_csv('data/raw/diabetes.csv')
    
    feature_cols = [c for c in df.columns if c != 'progression']
    X = df[feature_cols]
    y = df['progression']
    
    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(scaler, 'data/processed/scaler.joblib')
    
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print(f"Features built successfully:")
    print(f"  Training samples:   {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples:       {X_test.shape[0]}")

if __name__ == '__main__':
    main()

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

def main():
    print("Training model for Diabetes Progression Prediction...")
    os.makedirs('models', exist_ok=True)
    
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    # Validation evaluation
    val_preds = model.predict(X_val)
    val_r2 = r2_score(y_val, val_preds)
    val_mse = mean_squared_error(y_val, val_preds)
    
    # Test evaluation
    test_preds = model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    # 5-fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Save model
    joblib.dump(model, 'models/diabetes_model.joblib')
    
    report_content = f"""Diabetes Progression Prediction Model Performance
===================================================
Algorithm: Random Forest Regressor (n_estimators=100, max_depth=6)

Validation Set Metrics:
  - R² Score: {val_r2:.4f}
  - Mean Squared Error (MSE): {val_mse:.4f}
  - Root Mean Squared Error (RMSE): {np.sqrt(val_mse):.4f}

Test Set Metrics:
  - R² Score: {test_r2:.4f}
  - Mean Squared Error (MSE): {test_mse:.4f}
  - Root Mean Squared Error (RMSE): {test_rmse:.4f}
  - Mean Absolute Error (MAE): {test_mae:.4f}

5-Fold Cross-Validation (R² on Training Set):
  - Fold Scores: {[round(s, 4) for s in cv_scores]}
  - Mean R²: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print("Model saved to models/diabetes_model.joblib and accuracy.txt created.")

if __name__ == '__main__':
    main()

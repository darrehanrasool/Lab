import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def main():
    print("Training model for Stock Prices Prediction...")
    os.makedirs('models', exist_ok=True)
    
    # Load processed matrices
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
    val_rmse = np.sqrt(val_mse)
    
    # Test evaluation
    test_preds = model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
    
    # Time Series Cross-Validation (Expanding Window)
    tscv = TimeSeriesSplit(n_splits=5)
    tscv_scores = []
    for train_idx, test_idx in tscv.split(X_train):
        m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        m.fit(X_train.iloc[train_idx], y_train[train_idx])
        preds_fold = m.predict(X_train.iloc[test_idx])
        tscv_scores.append(r2_score(y_train[test_idx], preds_fold))
        
    joblib.dump(model, 'models/stock_model.joblib')
    
    report_content = f"""Stock Prices Prediction Model Performance
===========================================
Algorithm: Random Forest Regressor (n_estimators=100, max_depth=6) with Technical Indicators

Validation Set Evaluation:
  - R² Score: {val_r2:.4f}
  - Root Mean Squared Error (RMSE): ${val_rmse:.2f}

Test Set Evaluation (Held-Out Future Window):
  - R² Score:                       {test_r2:.4f}
  - Root Mean Squared Error (RMSE): ${test_rmse:.2f}
  - Mean Absolute Error (MAE):       ${test_mae:.2f}
  - Mean Absolute % Error (MAPE):   {test_mape:.2f}%
  - Actual Price Mean:              ${np.mean(y_test):.2f}
  - Predicted Price Mean:           ${np.mean(test_preds):.2f}

Time Series Cross-Validation (TimeSeriesSplit Expanding Window R²):
  - Fold Scores: {[round(s, 4) for s in tscv_scores]}
  - Mean Temporal R²: {np.mean(tscv_scores):.4f} (+/- {np.std(tscv_scores):.4f})
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print("Model saved to models/stock_model.joblib and accuracy.txt generated.")

if __name__ == '__main__':
    main()

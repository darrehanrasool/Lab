import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import os

def main():
    print("Training models for Boston Housing Price Prediction (Syllabus: Linear Regression & Linear SVR)...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # 1. Simple Linear Regression (Baseline: RM -> MEDV)
    # Feature 5 is 'RM' (average number of rooms per dwelling)
    rm_idx = list(X_train.columns).index('RM') if 'RM' in X_train.columns else 5
    simple_lr = LinearRegression()
    simple_lr.fit(X_train.iloc[:, [rm_idx]], y_train)
    simple_preds = simple_lr.predict(X_test.iloc[:, [rm_idx]])
    simple_r2 = r2_score(y_test, simple_preds)
    simple_mse = mean_squared_error(y_test, simple_preds)
    
    # 2. Multiple Linear Regression (OLS - Ordinary Least Squares via Normal Equation)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    val_r2 = r2_score(y_val, val_preds)
    val_mse = mean_squared_error(y_val, val_preds)
    
    test_preds = model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    
    # 3. Linear Support Vector Regression (Linear SVR - Maximum Flatness & Epsilon-Tube, No Kernels)
    linear_svr = SVR(kernel='linear', C=1.0, epsilon=0.2)
    linear_svr.fit(X_train, y_train)
    svr_preds = linear_svr.predict(X_test)
    svr_r2 = r2_score(y_test, svr_preds)
    svr_mse = mean_squared_error(y_test, svr_preds)
    
    # 5-Fold Cross Validation on Training Data
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    os.makedirs('models', exist_ok=True)
    # Save Multiple Linear Regression as primary model for Flask serving
    joblib.dump(model, 'models/boston_model.joblib')
    joblib.dump(linear_svr, 'models/linear_svr_model.joblib')
    
    report_content = f"""Boston Housing Price Prediction Performance
==============================================
Algorithms: Multiple Linear Regression (OLS) & Linear SVR (No Kernels, No Regularization)

Simple Linear Regression (RM -> MEDV):
  - R2 Score: {simple_r2:.4f}
  - MSE:      {simple_mse:.4f}

Multiple Linear Regression (OLS - All 13 Features):
  - Validation R2:   {val_r2:.4f}
  - Validation MSE:  {val_mse:.4f}
  - Test R2 Score:   {test_r2:.4f} ({test_r2*100:.2f}% Variance Explained)
  - Test MSE:        {test_mse:.4f}
  - Test RMSE:       ${test_rmse*1000:,.2f}
  - Test MAE:        ${test_mae*1000:,.2f}

Linear Support Vector Regression (Linear SVR, C=1.0, eps=0.2):
  - Test R2 Score:   {svr_r2:.4f}
  - Test MSE:        {svr_mse:.4f}

5-Fold Cross-Validation (R2 on Training Set):
  - Mean CV R2:      {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print("Model saved to models/boston_model.joblib and accuracy.txt generated.")

if __name__ == '__main__':
    main()
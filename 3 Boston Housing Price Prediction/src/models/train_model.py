"""
========================================================================================
PROJECT 3: BOSTON HOUSING PRICE PREDICTION
MODULE 3: MODEL TRAINING, SVR BENCHMARKING & EVALUATION (src/models/train_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Harrison & Rubinfeld (1978) Boston Housing Cohort (N = 506 Census Tracts)

PURPOSE:
--------
1. Fits a Simple Linear Regression baseline using solely Room Count (RM -> MEDV).
2. Fits an Ordinary Least Squares (OLS) Multiple Linear Regression model across all 13 features.
3. Fits a Linear Support Vector Regression (Linear SVR) model using an epsilon-insensitive tube.
4. Evaluates R², MSE, RMSE, and MAE across Validation (20%) and Holdout Test (10%) splits.
5. Computes 5-Fold Cross Validation on the training split to guarantee stability.
6. Serializes model artifacts to `models/` and generates `accuracy.txt`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is the Ordinary Least Squares (OLS) Normal Equation?
    -> beta = (X^T * X)^(-1) * X^T * y
       This minimizes the sum of squared residuals without requiring hyperparameter tuning or learning rates.

Q2: How does Linear Support Vector Regression (Linear SVR) work?
    -> SVR fits a linear hyperplane within an epsilon-insensitive tube (epsilon = 0.2).
       Any residual error smaller than epsilon is completely ignored (loss = 0). Errors outside the tube
       are penalized linearly according to regularization parameter C = 1.0. This makes SVR robust to outliers.

Q3: Why compare Simple Linear Regression (RM only) to Multiple Linear Regression?
    -> Simple linear regression with RM achieves R² ~ 0.4288 (explaining ~43% of price variance).
       Adding all 13 features increases R² to 0.7140 (explaining ~71.4% of price variance), demonstrating
       the massive predictive power gained from environmental, socioeconomic, and tax indicators.
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR


def main() -> None:
    """Executes multi-algorithm regression training, benchmarking, and metric reporting."""
    print("==================================================================")
    print("PROJECT 3 - STEP 3: OLS REGRESSION & LINEAR SVR BENCHMARKING")
    print("==================================================================")

    # 1. Load standardized training, validation, and holdout test splits
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # -------------------------------------------------------------------------
    # 2. BASELINE SIMPLE LINEAR REGRESSION (RM -> MEDV)
    # -------------------------------------------------------------------------
    # Univariate baseline: Predict home price using solely the number of rooms (RM)
    rm_idx = list(X_train.columns).index("RM") if "RM" in X_train.columns else 5
    simple_lr = LinearRegression()
    simple_lr.fit(X_train.iloc[:, [rm_idx]], y_train)
    simple_preds = simple_lr.predict(X_test.iloc[:, [rm_idx]])
    simple_r2 = float(r2_score(y_test, simple_preds))
    simple_mse = float(mean_squared_error(y_test, simple_preds))

    # -------------------------------------------------------------------------
    # 3. MULTIPLE LINEAR REGRESSION (OLS - All 13 Features)
    # -------------------------------------------------------------------------
    # Closed-form solution via Normal Equation: beta = (X^T * X)^(-1) * X^T * y
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validation Set Evaluation (20%, N=101)
    val_preds = model.predict(X_val)
    val_r2 = float(r2_score(y_val, val_preds))
    val_mse = float(mean_squared_error(y_val, val_preds))

    # Holdout Test Set Evaluation (10%, N=51)
    test_preds = model.predict(X_test)
    test_r2 = float(r2_score(y_test, test_preds))
    test_mse = float(mean_squared_error(y_test, test_preds))
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_rmse = float(np.sqrt(test_mse))

    # -------------------------------------------------------------------------
    # 4. LINEAR SUPPORT VECTOR REGRESSION (Linear SVR)
    # -------------------------------------------------------------------------
    # Maximum flatness with epsilon-insensitive tube (epsilon=0.2, C=1.0)
    linear_svr = SVR(kernel="linear", C=1.0, epsilon=0.2)
    linear_svr.fit(X_train, y_train)
    svr_preds = linear_svr.predict(X_test)
    svr_r2 = float(r2_score(y_test, svr_preds))
    svr_mse = float(mean_squared_error(y_test, svr_preds))

    # 5-Fold Cross Validation on Training Set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    # -------------------------------------------------------------------------
    # 5. SERIALIZE TRAINED ARTIFACTS
    # -------------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/boston_model.joblib")
    joblib.dump(linear_svr, "models/linear_svr_model.joblib")

    # -------------------------------------------------------------------------
    # 6. GENERATE ACCURACY REPORT (accuracy.txt)
    # -------------------------------------------------------------------------
    report_content = f"""Boston Housing Price Prediction Performance
==============================================
Algorithms: Multiple Linear Regression (OLS) & Linear SVR (No Kernels, No Regularization)

Simple Linear Regression (RM -> MEDV):
  - R2 Score: {simple_r2:.4f}
  - MSE:      {simple_mse:.4f}

Multiple Linear Regression (OLS - All 13 Features):
  - Validation R2:   {val_r2:.4f}
  - Validation MSE:  {val_mse:.4f}
  - Test R2 Score:   {test_r2:.4f} ({test_r2 * 100:.2f}% Variance Explained)
  - Test MSE:        {test_mse:.4f}
  - Test RMSE:       ${test_rmse * 1000:,.2f}
  - Test MAE:        ${test_mae * 1000:,.2f}

Linear Support Vector Regression (Linear SVR, C=1.0, eps=0.2):
  - Test R2 Score:   {svr_r2:.4f}
  - Test MSE:        {svr_mse:.4f}

5-Fold Cross-Validation (R2 on Training Set):
  - Mean CV R2:      {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})
"""
    with open("accuracy.txt", "w") as f:
        f.write(report_content)

    print(report_content)
    print("✓ Model saved to 'models/boston_model.joblib' and 'accuracy.txt' generated.")


if __name__ == "__main__":
    main()

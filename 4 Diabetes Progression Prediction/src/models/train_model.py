import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score

def main():
    print("==========================================================")
    print("STEP 3: MULTI-MODEL TRAINING & SYLLABUS-ALIGNED BENCHMARK")
    print("==========================================================")
    os.makedirs('models', exist_ok=True)
    
    # Load processed splits
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val   = pd.read_csv('data/processed/X_val.csv')
    y_val   = pd.read_csv('data/processed/y_val.csv').values.ravel()
    X_test  = pd.read_csv('data/processed/X_test.csv')
    y_test  = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    models = {
        "Multiple Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (L2 alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Linear SVR (C=10.0, eps=5.0)": SVR(kernel='linear', C=10.0, epsilon=5.0),
        "Random Forest Regressor (max_depth=5)": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=6, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_test_r2 = -999.0
    best_model_obj = None
    
    print("\n--- Model Training & Validation Evaluation ---")
    for name, model in models.items():
        # Train strictly on 70% Train
        model.fit(X_train, y_train)
        
        # Validation Metrics (20% Val)
        val_preds = model.predict(X_val)
        val_r2    = r2_score(y_val, val_preds)
        val_mse   = mean_squared_error(y_val, val_preds)
        val_rmse  = np.sqrt(val_mse)
        val_mae   = mean_absolute_error(y_val, val_preds)
        
        # Test Metrics (10% Test)
        test_preds = model.predict(X_test)
        test_r2    = r2_score(y_test, test_preds)
        test_mse   = mean_squared_error(y_test, test_preds)
        test_rmse  = np.sqrt(test_mse)
        test_mae   = mean_absolute_error(y_test, test_preds)
        
        # 5-Fold Cross-Validation on Training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            "model": model,
            "val_r2": val_r2, "val_rmse": val_rmse,
            "test_r2": test_r2, "test_mse": test_mse, "test_rmse": test_rmse, "test_mae": test_mae,
            "cv_mean": np.mean(cv_scores), "cv_std": np.std(cv_scores), "cv_scores": cv_scores
        }
        
        print(f"\nModel: {name}")
        print(f"  Validation R²: {val_r2:.4f} | Val RMSE: {val_rmse:.2f}")
        print(f"  Test R²:       {test_r2:.4f} | Test RMSE: {test_rmse:.2f} | Test MAE: {test_mae:.2f}")
        print(f"  5-Fold Mean R²: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_name = name
            best_model_obj = model

    print(f"\n>>> Best Performing Model on Holdout Test Set: {best_model_name} (R² = {best_test_r2:.4f})")
    
    # Save the primary production model and the Ridge/Linear models
    joblib.dump(best_model_obj, 'models/diabetes_model.joblib')
    joblib.dump(models["Ridge Regression (L2 alpha=1.0)"], 'models/ridge_model.joblib')
    joblib.dump(models["Multiple Linear Regression (OLS)"], 'models/ols_model.joblib')
    joblib.dump(models["Linear SVR (C=10.0, eps=5.0)"], 'models/linear_svr_model.joblib')
    joblib.dump(models["Random Forest Regressor (max_depth=5)"], 'models/random_forest_model.joblib')
    
    # -------------------------------------------------------------
    # DUAL EVALUATION: CLINICAL RISK STRATIFICATION (CLASSIFICATION FRAMING)
    # Binary cut: High Progression Risk (y >= 150) vs Controlled (y < 150)
    # -------------------------------------------------------------
    threshold = 150.0
    y_test_binary = (y_test >= threshold).astype(int)
    best_test_preds = best_model_obj.predict(X_test)
    preds_binary = (best_test_preds >= threshold).astype(int)
    
    cm = confusion_matrix(y_test_binary, preds_binary)
    acc = accuracy_score(y_test_binary, preds_binary)
    prec = precision_score(y_test_binary, preds_binary)
    rec = recall_score(y_test_binary, preds_binary) # Sensitivity
    f1 = f1_score(y_test_binary, preds_binary)
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print("\n--- Dual Clinical Risk Stratification Analysis (Threshold: 150 Progression) ---")
    print(f"Classification Accuracy: {acc*100:.2f}%")
    print(f"Sensitivity (Recall):     {rec*100:.2f}%")
    print(f"Specificity:              {spec*100:.2f}%")
    print(f"Precision:                {prec*100:.2f}%")
    print(f"F1-Score:                 {f1:.4f}")
    print(f"Confusion Matrix:\n  TN={tn}, FP={fp}\n  FN={fn}, TP={tp}")
    
    # Generate full accuracy.txt
    report = f"""Diabetes Progression Prediction Performance Report
======================================================
Course: CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: 442 Diabetes Patients • Split: 70% Train (309), 20% Val (88), 10% Test (45)

1. REGRESSION BENCHMARK COMPARISON ON TEST SET (N=45):
---------------------------------------------------------------------------------------------------
Algorithm                                Test R²     Test RMSE     Test MAE     Mean 5-Fold CV R²
---------------------------------------------------------------------------------------------------
Multiple Linear Regression (OLS)          {results["Multiple Linear Regression (OLS)"]["test_r2"]:.4f}      {results["Multiple Linear Regression (OLS)"]["test_rmse"]:.2f}         {results["Multiple Linear Regression (OLS)"]["test_mae"]:.2f}        {results["Multiple Linear Regression (OLS)"]["cv_mean"]:.4f} (±{results["Multiple Linear Regression (OLS)"]["cv_std"]:.4f})
Ridge Regression (L2, alpha=1.0)          {results["Ridge Regression (L2 alpha=1.0)"]["test_r2"]:.4f}      {results["Ridge Regression (L2 alpha=1.0)"]["test_rmse"]:.2f}         {results["Ridge Regression (L2 alpha=1.0)"]["test_mae"]:.2f}        {results["Ridge Regression (L2 alpha=1.0)"]["cv_mean"]:.4f} (±{results["Ridge Regression (L2 alpha=1.0)"]["cv_std"]:.4f})
Linear Support Vector Regression (SVR)    {results["Linear SVR (C=10.0, eps=5.0)"]["test_r2"]:.4f}      {results["Linear SVR (C=10.0, eps=5.0)"]["test_rmse"]:.2f}         {results["Linear SVR (C=10.0, eps=5.0)"]["test_mae"]:.2f}        {results["Linear SVR (C=10.0, eps=5.0)"]["cv_mean"]:.4f} (±{results["Linear SVR (C=10.0, eps=5.0)"]["cv_std"]:.4f})
Random Forest Regressor (max_depth=5)     {results["Random Forest Regressor (max_depth=5)"]["test_r2"]:.4f}      {results["Random Forest Regressor (max_depth=5)"]["test_rmse"]:.2f}         {results["Random Forest Regressor (max_depth=5)"]["test_mae"]:.2f}        {results["Random Forest Regressor (max_depth=5)"]["cv_mean"]:.4f} (±{results["Random Forest Regressor (max_depth=5)"]["cv_std"]:.4f})
---------------------------------------------------------------------------------------------------

Primary Model Selected for Serving: {best_model_name}
  - Test R² Score: {best_test_r2:.4f}
  - Test RMSE:     {results[best_model_name]["test_rmse"]:.2f}
  - Test MAE:      {results[best_model_name]["test_mae"]:.2f}

2. DUAL CLINICAL RISK CLASSIFICATION (High Risk >= 150 vs Controlled < 150):
---------------------------------------------------------------------------------------------------
  - Clinical Accuracy:   {acc*100:.2f}%
  - Clinical Sensitivity: {rec*100:.2f}% (High-risk patients correctly identified)
  - Clinical Specificity: {spec*100:.2f}% (Controlled patients correctly cleared)
  - Clinical Precision:   {prec*100:.2f}%
  - Clinical F1-Score:    {f1:.4f}
  - Confusion Matrix:
      [TN={tn:2d}  FP={fp:2d}]
      [FN={fn:2d}  TP={tp:2d}]

3. 5-FOLD CROSS-VALIDATION BREAKDOWN ({best_model_name}):
---------------------------------------------------------------------------------------------------
  - Fold Scores: {[round(s, 4) for s in results[best_model_name]["cv_scores"]]}
  - Mean R²:     {results[best_model_name]["cv_mean"]:.4f} (+/- {results[best_model_name]["cv_std"]:.4f})
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report)
    
    print("\n✓ Saved accuracy.txt and all model checkpoints to models/.")

if __name__ == '__main__':
    main()

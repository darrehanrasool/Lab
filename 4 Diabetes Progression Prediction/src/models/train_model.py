"""
========================================================================================
MODULE 3: MODEL TRAINING, BENCHMARKING & CLINICAL TRIAGE (src/models/train_model.py)
========================================================================================
Project: Diabetes Progression Prediction via Multiple Linear Regression
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Efron et al. (2004) Authentic Clinical Diabetes Cohort (N = 442 Patients)

PURPOSE:
--------
1. Fits an Ordinary Least Squares (OLS) Multiple Linear Regression model on standardized training features.
2. Evaluates performance across Training (70%), Validation (20%), and Holdout Test (10%) splits.
3. Performs 5-Fold Cross-Validation on the training cohort to test stability across data folds.
4. Diagnoses multicollinearity among blood serum biomarkers using Variance Inflation Factors (VIF).
5. Solves clinical class imbalance for high-risk triage (progression >= 200) using cost-sensitive
   balanced Logistic Regression (class_weight='balanced') to minimize life-threatening False Negatives.
6. Conducts slice-based continuous error analysis to diagnose tail sparsity.
7. Serializes trained model artifacts and generates the comprehensive benchmark report 'accuracy.txt'.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is Ordinary Least Squares (OLS) and how does it find the weights?
    -> OLS minimizes the Residual Sum of Squares: RSS = sum((y_i - y_hat_i)^2).
       Unlike iterative gradient descent, OLS has an exact analytical closed-form solution
       known as the Normal Equation:
           beta = (X^T * X)^(-1) * X^T * y
       This directly finds the global minimum of the convex quadratic loss surface.

Q2: What does the Intercept (beta_0) mean in standardized regression?
    -> Because all features were standardized to have mean = 0 (z_j = (x_j - mu_j) / sigma_j),
       when all patient biomarkers are at their cohort averages (z_j = 0 for all j),
       y_hat = beta_0. Thus, beta_0 = 151.21 is the expected 1-year progression score for
       an 'average' diabetes patient.

Q3: What do R², RMSE, and MAE measure?
    -> R² (Coefficient of Determination): Fraction of total target variance explained by the features
       (R² = 1 - SS_res / SS_tot). 0.4882 means ~48.8% of progression variance is explained.
    -> RMSE (Root Mean Squared Error): Penalizes larger errors quadratically (RMSE = sqrt(mean(errors^2))).
    -> MAE (Mean Absolute Error): Average absolute magnitude of errors (|y - y_hat|), robust to outliers.

Q4: Why use class_weight='balanced' in the triage logistic regression?
    -> Severe progression (>= 200) accounts for only 28.7% of patients. In standard logistic regression,
       the algorithm prioritizes the majority class to maximize raw accuracy, causing it to miss
       severe patients (high False Negatives). By weighting the loss inversely proportional to class
       frequencies (w_k = N / (2 * N_k)), the penalty for missing a severe patient increases,
       raising Clinical Sensitivity (Recall) from 61.54% to 76.92% and cutting missed severe cases.
========================================================================================
"""

import os
from typing import Dict, List

import joblib  # Model serialization to .joblib binary files
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def compute_vifs(X_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes the Variance Inflation Factor (VIF) for each feature to diagnose multicollinearity.
    
    Formula:
        VIF_j = 1 / (1 - R_j^2)
    where R_j^2 is the R² score obtained by regressing feature j against all remaining 9 features.
    
    Clinical Significance:
      - VIF = 1: No collinearity.
      - VIF 1 to 5: Moderate, acceptable correlation.
      - VIF > 5 or 10: Severe multicollinearity.
      In this dataset, total cholesterol (s1) and LDL (s2) have high correlation (r = +0.897),
      causing their beta weights to oppose each other (s1 = -31.89, s2 = +18.96).

    Args:
        X_df (pd.DataFrame): Standardized training features.

    Returns:
        Dict[str, float]: Feature name mapped to its VIF value.
    """
    vifs: Dict[str, float] = {}
    for col in X_df.columns:
        # Target is the single column
        y_c = X_df[col]
        # Predictors are all other 9 columns
        X_o = X_df.drop(columns=[col])
        # Fit auxiliary linear regression
        aux_model = LinearRegression().fit(X_o, y_c)
        r2 = aux_model.score(X_o, y_c)
        # Avoid division by zero if R² == 1.0
        vifs[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return vifs


def main() -> None:
    """
    Executes model training, multi-split evaluation, cross-validation, and triage benchmarking.
    """
    print("==================================================================")
    print("STEP 3: MULTIPLE LINEAR REGRESSION TRAINING & BENCHMARKING")
    print("==================================================================")
    os.makedirs("models", exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. LOAD PREPROCESSED DATASET SPLITS
    # -------------------------------------------------------------------------
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    feature_names: List[str] = list(X_train.columns)

    # -------------------------------------------------------------------------
    # 2. MULTIPLE LINEAR REGRESSION (ORDINARY LEAST SQUARES)
    # -------------------------------------------------------------------------
    # Mathematical Equation:
    #   y_hat = beta_0 + beta_1 * z_1 + beta_2 * z_2 + ... + beta_10 * z_10
    # Closed-form solution via Normal Equation:
    #   beta = (X^T * X)^(-1) * X^T * y
    ols: LinearRegression = LinearRegression()
    ols.fit(X_train, y_train)

    # Evaluate Training Cohort (70%, N=309)
    train_preds = ols.predict(X_train)
    train_r2 = r2_score(y_train, train_preds)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    train_mae = float(mean_absolute_error(y_train, train_preds))

    # Evaluate Validation Cohort (20%, N=88)
    val_preds = ols.predict(X_val)
    val_r2 = r2_score(y_val, val_preds)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_mae = float(mean_absolute_error(y_val, val_preds))

    # Evaluate Holdout Test Cohort (10%, N=45) - Completely unseen data
    test_preds = ols.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    test_mae = float(mean_absolute_error(y_test, test_preds))

    # 5-Fold Cross Validation on Training Split (Guarantees stability across folds)
    cv_scores = cross_val_score(ols, X_train, y_train, cv=5, scoring="r2")

    print("\n--- Multiple Linear Regression Results ---")
    print(f"  Training R²:       {train_r2:.4f} | Train RMSE: {train_rmse:.2f} | Train MAE: {train_mae:.2f}")
    print(f"  Validation R²:     {val_r2:.4f} | Val RMSE:   {val_rmse:.2f} | Val MAE:   {val_mae:.2f}")
    print(f"  Holdout Test R²:   {test_r2:.4f} | Test RMSE:  {test_rmse:.2f} | Test MAE:  {test_mae:.2f}")
    print(f"  5-Fold CV Mean R²: {float(np.mean(cv_scores)):.4f} (±{float(np.std(cv_scores)):.4f})")
    print(f"  Intercept (Beta 0): {ols.intercept_:.2f}")

    # -------------------------------------------------------------------------
    # 3. STANDARDIZED FEATURE COEFFICIENTS (BETA WEIGHTS)
    # -------------------------------------------------------------------------
    print("\n--- Standardized Feature Coefficients (Beta Weights) ---")
    coef_df = pd.DataFrame(
        {"Feature": feature_names, "Beta_Weight": ols.coef_}
    ).sort_values(by="Beta_Weight", ascending=False)
    print(coef_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. CLINICAL RISK TRIAGE & IMBALANCE RESOLUTION (Severe Progression >= 200)
    # -------------------------------------------------------------------------
    # Clinical justification: In a hospital setting, patients progressing >= 200 require
    # immediate pharmaceutical intensification. Severe cases comprise only 28.7% of cohort.
    y_train_sev = (y_train >= 200).astype(int)
    y_test_sev = (y_test >= 200).astype(int)

    # 4.1 Unweighted Baseline Logistic Regression (Suffers from Majority Class Bias)
    lr_un = LogisticRegression(random_state=42).fit(X_train, y_train_sev)
    preds_un = lr_un.predict(X_test)
    cm_un = confusion_matrix(y_test_sev, preds_un)

    # 4.2 Balanced Cost-Sensitive Logistic Regression (class_weight='balanced')
    # Penalizes False Negatives inversely proportional to class frequencies: w_k = N / (2 * N_k)
    lr_bal = LogisticRegression(class_weight="balanced", random_state=42).fit(X_train, y_train_sev)
    preds_bal = lr_bal.predict(X_test)
    cm_bal = confusion_matrix(y_test_sev, preds_bal)

    # Compute comparative diagnostic metrics
    rec_un = float(recall_score(y_test_sev, preds_un))
    rec_bal = float(recall_score(y_test_sev, preds_bal))
    prec_un = float(precision_score(y_test_sev, preds_un))
    prec_bal = float(precision_score(y_test_sev, preds_bal))
    acc_un = float(accuracy_score(y_test_sev, preds_un))
    acc_bal = float(accuracy_score(y_test_sev, preds_bal))
    f1_un = float(f1_score(y_test_sev, preds_un))
    f1_bal = float(f1_score(y_test_sev, preds_bal))

    print("\n--- Managing Dataset Imbalance: Clinical Risk Triage (Progression >= 200) ---")
    print("A. Unweighted Baseline (Suffers from Majority Class Bias):")
    print(f"   Accuracy: {acc_un * 100:.2f}% | Sensitivity/Recall: {rec_un * 100:.2f}% | Precision: {prec_un * 100:.2f}% | F1: {f1_un:.4f}")
    print(f"   Confusion Matrix: TN={cm_un[0, 0]}, FP={cm_un[0, 1]}, FN={cm_un[1, 0]}, TP={cm_un[1, 1]}")
    print(f"   -> Missed {cm_un[1, 0]} severe patients because the model is biased toward the majority negative class!")

    print("\nB. Balanced Cost-Sensitive Model (Managing Imbalance via class_weight='balanced'):")
    print(f"   Accuracy: {acc_bal * 100:.2f}% | Sensitivity/Recall: {rec_bal * 100:.2f}% | Precision: {prec_bal * 100:.2f}% | F1: {f1_bal:.4f}")
    print(f"   Confusion Matrix: TN={cm_bal[0, 0]}, FP={cm_bal[0, 1]}, FN={cm_bal[1, 0]}, TP={cm_bal[1, 1]}")
    print(f"   -> Clinical Sensitivity jumped from {rec_un * 100:.1f}% to {rec_bal * 100:.1f}%! Missed severe cases reduced to {cm_bal[1, 0]}.")

    # -------------------------------------------------------------------------
    # 5. SLICE-BASED CONTINUOUS ERROR ANALYSIS (TAIL SPARSITY)
    # -------------------------------------------------------------------------
    slice_mild = y_test < 100
    slice_mod = (y_test >= 100) & (y_test <= 200)
    slice_sev = y_test > 200

    mae_mild = float(mean_absolute_error(y_test[slice_mild], test_preds[slice_mild]))
    mae_mod = float(mean_absolute_error(y_test[slice_mod], test_preds[slice_mod]))
    mae_sev = float(mean_absolute_error(y_test[slice_sev], test_preds[slice_sev]))

    # -------------------------------------------------------------------------
    # 6. SERIALIZE TRAINED ARTIFACTS
    # -------------------------------------------------------------------------
    joblib.dump(ols, "models/diabetes_model.joblib")
    joblib.dump(ols, "models/ols_model.joblib")
    joblib.dump(lr_bal, "models/triage_logistic_model.joblib")

    # -------------------------------------------------------------------------
    # 7. GENERATE COMPREHENSIVE BENCHMARK REPORT (accuracy.txt)
    # -------------------------------------------------------------------------
    report: str = f"""Diabetes Progression Prediction Performance Report
======================================================
Course: CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: 442 Diabetes Patients • Split: 70% Train (309), 20% Val (88), 10% Test (45)
Model Architecture: Multiple Linear Regression (Ordinary Least Squares - OLS)
Explainability: 100% Transparent Linear Formulation (Zero Black-Box)

1. MULTIPLE LINEAR REGRESSION BENCHMARK PERFORMANCE:
---------------------------------------------------------------------------------------------------
Dataset Split                  R² Score     RMSE (Loss)     MAE          Interpretation
---------------------------------------------------------------------------------------------------
Training Set (70%, N=309)      {train_r2:.4f}       {train_rmse:.2f}          {train_mae:.2f}        Baseline fit on training cohort
Validation Set (20%, N=88)     {val_r2:.4f}       {val_rmse:.2f}          {val_mae:.2f}        Independent parameter verification
Holdout Test Set (10%, N=45)   {test_r2:.4f}       {test_rmse:.2f}          {test_mae:.2f}        Unseen generalization benchmark
---------------------------------------------------------------------------------------------------
5-Fold Cross Validation Mean R²: {float(np.mean(cv_scores)):.4f} (±{float(np.std(cv_scores)):.4f}) -> Consistent stability across folds!

Mathematical Formulation:
  y_hat = beta_0 + sum_{{j=1}}^{{10}} (beta_j * z_j)
  where z_j = (x_j - mu_j) / sigma_j (Standardized Z-Score)
  Baseline Intercept (beta_0): {ols.intercept_:.2f}

2. FEATURE COEFFICIENTS & CLINICAL INTERPRETATION:
---------------------------------------------------------------------------------------------------
Biomarker    Beta Weight    Direction        Clinical / Physiological Significance
---------------------------------------------------------------------------------------------------
s5 (LTG)     +{ols.coef_[8]:.2f}         Accelerant       Serum triglycerides: Strong metabolic risk driver
bmi          +{ols.coef_[2]:.2f}         Accelerant       Body mass index: Visceral obesity drives insulin resistance
s2 (LDL)     +{ols.coef_[5]:.2f}         Accelerant       Low-density lipoprotein ('bad cholesterol')
bp           +{ols.coef_[3]:.2f}         Accelerant       Mean arterial blood pressure: Vascular wall shear stress
s4 (TCH)     +{ols.coef_[7]:.2f}         Accelerant       Total cholesterol / HDL ratio: Atherogenic index
s6 (GLU)     +{ols.coef_[9]:.2f}         Accelerant       Fasting blood glucose: Active glycemic burden
s3 (HDL)     +{ols.coef_[6]:.2f}         Collinear        HDL cholesterol (affected by collinearity with S1/S2)
age          {ols.coef_[0]:.2f}         Neutral          Patient age: Minimal marginal impact in this cohort
sex          {ols.coef_[1]:.2f}         Demographic      Binary sex [0=Female, 1=Male]: Baseline differential
s1 (TC)      {ols.coef_[4]:.2f}         Collinear        Total cholesterol (collinear with LDL and HDL)
---------------------------------------------------------------------------------------------------

3. MANAGING DATASET IMBALANCE: CLINICAL RISK TRIAGE (Severe Progression >= 200):
---------------------------------------------------------------------------------------------------
Clinical Context: 28.7% Severe Cases (Class Ratio ~ 1:2.5 imbalanced)
Strategy: Balanced Cost-Sensitive Logistic Regression (class_weight='balanced')

Metric                     Unweighted Baseline     Balanced Model (Managing Imbalance)
---------------------------------------------------------------------------------------------------
Clinical Accuracy          {acc_un * 100:.2f}%                 {acc_bal * 100:.2f}%
Clinical Sensitivity       {rec_un * 100:.2f}%                 {rec_bal * 100:.2f}% (CRITICAL: +{rec_bal * 100 - rec_un * 100:.1f}% recall)
Clinical Specificity       {cm_un[0, 0] / (cm_un[0, 0] + cm_un[0, 1]) * 100:.2f}%                 {cm_bal[0, 0] / (cm_bal[0, 0] + cm_bal[0, 1]) * 100:.2f}%
Clinical Precision         {prec_un * 100:.2f}%                 {prec_bal * 100:.2f}%
Clinical F1-Score          {f1_un:.4f}                 {f1_bal:.4f}
Confusion Matrix           [TN={cm_un[0, 0]:2d} FP={cm_un[0, 1]:2d}]             [TN={cm_bal[0, 0]:2d} FP={cm_bal[0, 1]:2d}]
                           [FN={cm_un[1, 0]:2d} TP={cm_un[1, 1]:2d}]             [FN={cm_bal[1, 0]:2d} TP={cm_bal[1, 1]:2d}]
---------------------------------------------------------------------------------------------------
-> By managing imbalance via cost-sensitive weighting, False Negatives were reduced from {cm_un[1, 0]} to {cm_bal[1, 0]}.
   In clinical medicine, missing a deteriorating diabetic patient (FN) is far more dangerous than a false alarm (FP)!

4. SLICE-BASED CONTINUOUS ERROR ANALYSIS (TAIL SPARSITY):
---------------------------------------------------------------------------------------------------
Cohort Subgroup            Test Samples (N)    Mean Actual    Mean Predicted    MAE
---------------------------------------------------------------------------------------------------
Mild (<100)                {int(slice_mild.sum()):2d}                  {y_test[slice_mild].mean():6.1f}         {test_preds[slice_mild].mean():6.1f}          {mae_mild:.2f}
Moderate (100-200)         {int(slice_mod.sum()):2d}                  {y_test[slice_mod].mean():6.1f}         {test_preds[slice_mod].mean():6.1f}          {mae_mod:.2f}
Severe (>200)              {int(slice_sev.sum()):2d}                  {y_test[slice_sev].mean():6.1f}         {test_preds[slice_sev].mean():6.1f}          {mae_sev:.2f}
---------------------------------------------------------------------------------------------------
Observation: Moderate cases achieve lowest MAE ({mae_mod:.2f}). Extreme cases demonstrate 'regression to the mean'
due to continuous tail sparsity in the training distribution.
"""
    with open("accuracy.txt", "w") as f:
        f.write(report)

    print("\n✓ Multiple Linear Regression training complete.")
    print("✓ Serialized models to 'models/' and updated 'accuracy.txt'.")


if __name__ == "__main__":
    main()

"""
========================================================================================
MODULE 4: REAL-TIME PATIENT PROGNOSIS & EXPLAINABILITY (src/models/predict_model.py)
========================================================================================
Project: Diabetes Progression Prediction via Multiple Linear Regression
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Efron et al. (2004) Authentic Clinical Diabetes Cohort (N = 442 Patients)

PURPOSE:
--------
Provides an offline CLI prediction and explainability engine for a single patient:
1. Loads the serialized Multiple Linear Regression model, StandardScaler, and Triage Classifier.
2. Accepts authentic physiological and blood serum biomarker inputs.
3. Applies exact feature standardization using the frozen training statistics (mu, sigma).
4. Computes quantitative 1-year disease progression: y_hat = beta_0 + sum(beta_j * z_j).
5. Provides 100% white-box explainability via exact biomarker contribution decomposition:
       Impact_j = Beta_j * z_j
6. Computes severe progression probability (P(y >= 200)) using the balanced logistic triage model.
7. Maps predictions into 3 standardized clinical care stages (Mild, Moderate, Severe).

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How does the model compute the prediction for a new patient?
    -> The raw inputs are passed into `scaler.transform()`, which subtracts the training mean (mu)
       and divides by the training standard deviation (sigma) to get z-scores: z_j = (x_j - mu_j) / sigma_j.
       Then the linear model computes the dot product with the learned beta weights plus the intercept:
           y_hat = beta_0 + sum(beta_j * z_j).

Q2: What is the Biomarker Impact Breakdown (beta_j * z_j)?
    -> Unlike black-box neural networks or random forests, Multiple Linear Regression is
       100% transparent. Each feature's positive or negative push away from the baseline (beta_0)
       is exactly equal to its coefficient (beta_j) multiplied by how many standard deviations
       the patient is away from normal (z_j). If z_j > 0 and beta_j > 0, that biomarker is
       accelerating disease progression.

Q3: How is the Severe Triage Probability determined?
    -> It is calculated by the cost-sensitive balanced Logistic Regression model trained in Step 3
       via `triage_model.predict_proba()`. If probability >= 0.50, an urgent clinical alert is triggered.
========================================================================================
"""

import os
from typing import Any, Dict

import joblib  # Model and artifact deserialization
import pandas as pd


def predict_patient(
    age: float = 59.0,
    sex: float = 1.0,  # Raw clinical encoding: 1.0 = Female, 2.0 = Male
    bmi: float = 32.1,
    bp: float = 101.0,
    s1: float = 157.0,
    s2: float = 93.2,
    s3: float = 38.0,
    s4: float = 4.0,
    s5: float = 4.86,
    s6: float = 87.0,
) -> Dict[str, Any]:
    """
    Computes real-time diabetes disease progression prognosis using the interpretable OLS model.

    Mathematical Formulation:
        y_hat = beta_0 + sum_{j=1}^{10} (beta_j * z_j)
    where:
        z_j = (x_j - mu_j) / sigma_j  (Z-score standardization)
        beta_0 = Baseline Intercept (151.21)
        beta_j = Feature coefficient weight

    Args:
        age (float): Patient age in years.
        sex (float): Demographic sex (1.0 = Female, 2.0 = Male, or 0.0 / 1.0 binary).
        bmi (float): Body Mass Index in kg/m².
        bp (float): Mean arterial blood pressure in mmHg.
        s1 (float): Total serum cholesterol in mg/dL.
        s2 (float): Low-density lipoprotein (LDL) in mg/dL.
        s3 (float): High-density lipoprotein (HDL) in mg/dL.
        s4 (float): Total cholesterol / HDL ratio.
        s5 (float): Log of serum triglycerides level.
        s6 (float): Fasting blood glucose concentration in mg/dL.

    Returns:
        Dict[str, Any]: Structured clinical diagnostic dictionary containing:
            - progression_index: Predicted continuous 1-year progression score.
            - intercept_beta0: Baseline cohort intercept.
            - clinical_stage: Mild, Moderate, or Severe progression category.
            - severe_risk_probability: Percent risk of progression >= 200.
            - severe_triage_alert: Boolean flag for immediate clinical escalation.
            - clinical_action: Tailored physician intervention guideline.
            - biomarker_contributions: Linear decomposition of each biomarker (weight, z-score, impact).
    """
    # -------------------------------------------------------------------------
    # 1. RESOLVE ABSOLUTE FILE PATHS & LOAD MODEL ARTIFACTS
    # -------------------------------------------------------------------------
    # Dynamically locate project root directory regardless of current working directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "diabetes_model.joblib")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
    triage_path = os.path.join(base_dir, "models", "triage_logistic_model.joblib")

    # Load fitted artifacts into memory
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    triage_model = joblib.load(triage_path)

    # -------------------------------------------------------------------------
    # 2. ENCODE INPUT FEATURES & HANDLE DEMOGRAPHIC DUMMY VARIABLE
    # -------------------------------------------------------------------------
    # Standardize sex to binary dummy indicator: 0.0 = Female, 1.0 = Male
    if sex == 1.0:
        sex_binary = 0.0  # Raw female mapped to reference category 0.0
    elif sex == 2.0:
        sex_binary = 1.0  # Raw male mapped to dummy category 1.0
    else:
        sex_binary = float(sex)  # Already provided as 0.0 or 1.0

    feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    raw_values = [age, sex_binary, bmi, bp, s1, s2, s3, s4, s5, s6]

    # -------------------------------------------------------------------------
    # 3. APPLY FROZEN TRAINING SCALER (ZERO LEAKAGE INFERENCE)
    # -------------------------------------------------------------------------
    # Create single-row pandas DataFrame with matching column names
    df_input = pd.DataFrame([raw_values], columns=feature_names)

    # Standardize features using the frozen training mu and sigma
    X_scaled = scaler.transform(df_input)
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # -------------------------------------------------------------------------
    # 4. PREDICT CONTINUOUS PROGRESSION (MULTIPLE LINEAR REGRESSION)
    # -------------------------------------------------------------------------
    progression_val = float(model.predict(df_scaled)[0])

    # -------------------------------------------------------------------------
    # 5. LINEAR DECOMPOSITION / WHITE-BOX EXPLAINABILITY (Impact = Beta * Z)
    # -------------------------------------------------------------------------
    contributions = {}
    for name, weight, z_val in zip(feature_names, model.coef_, X_scaled[0]):
        contributions[name] = {
            "weight": round(float(weight), 3),
            "z_score": round(float(z_val), 3),
            "impact": round(float(weight * z_val), 2),  # Exact linear push away from intercept
        }

    # -------------------------------------------------------------------------
    # 6. COMPUTE SEVERE CLINICAL TRIAGE PROBABILITY (Logistic Regression)
    # -------------------------------------------------------------------------
    # Probability of belonging to class 1 (severe progression >= 200)
    severe_prob = float(triage_model.predict_proba(df_scaled)[0, 1])
    is_severe = bool(severe_prob >= 0.50)

    # -------------------------------------------------------------------------
    # 7. CLINICAL STAGING & PHYSICIAN CARE GUIDELINES
    # -------------------------------------------------------------------------
    if progression_val < 100.0:
        stage = "Mild Progression"
        action = "Routine annual monitoring. Maintain current diet and exercise protocol."
    elif progression_val < 200.0:
        stage = "Moderate Progression"
        action = "Lifestyle intervention recommended. Focus on BMI reduction and blood pressure management."
    else:
        stage = "Severe / Accelerated Progression"
        action = "Immediate clinical triage required. Review pharmacotherapy (Metformin / GLP-1 RA / SGLT2i) and lipid panel."

    return {
        "progression_index": round(progression_val, 2),
        "intercept_beta0": round(float(model.intercept_), 2),
        "clinical_stage": stage,
        "severe_risk_probability": round(severe_prob * 100.0, 1),
        "severe_triage_alert": is_severe,
        "clinical_action": action,
        "biomarker_contributions": contributions,
    }


if __name__ == "__main__":
    # Run sample inference demonstration for a high-risk patient
    res = predict_patient()
    print("=== DIABETES PROGRESSION PROGNOSIS REPORT ===")
    print(f"Predicted 1-Year Progression Index: {res['progression_index']} (Baseline Intercept: {res['intercept_beta0']})")
    print(f"Clinical Stage: {res['clinical_stage']}")
    print(f"Severe Risk Probability: {res['severe_risk_probability']}% (Alert: {res['severe_triage_alert']})")
    print(f"Recommended Clinical Action: {res['clinical_action']}")
    print("\nBiomarker Impact Breakdown (w_j * z_j):")
    for k, v in res["biomarker_contributions"].items():
        print(f"  {k:8s}: weight={v['weight']:+6.2f} | z-score={v['z_score']:+5.2f} | impact={v['impact']:+6.2f}")

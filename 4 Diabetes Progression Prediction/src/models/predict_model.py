import os
import joblib
import numpy as np
import pandas as pd

def predict_patient(age=59.0, sex=1.0, bmi=32.1, bp=101.0, s1=157.0, s2=93.2, s3=38.0, s4=4.0, s5=4.86, s6=87.0):
    """
    Computes real-time diabetes disease progression prognosis using the interpretable Ridge model.
    Demonstrates the exact closed-form linear combination:
        y_hat = w0 + sum(w_j * z_j)
    where z_j is the z-score standardized feature: z_j = (x_j - mu_j) / sigma_j.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'diabetes_model.joblib')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    triage_path = os.path.join(base_dir, 'models', 'triage_logistic_model.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    triage_model = joblib.load(triage_path)
    
    # Handle sex encoding: if passed as 2.0 (Male) or 1.0 (Female), map to binary 0.0 or 1.0
    sex_binary = 1.0 if sex == 2.0 or sex == 1.0 and sex != 0.0 and sex != 1.0 else (1.0 if sex == 1.0 else 0.0)
    # If the user passed raw sex: 1=Female, 2=Male
    if sex in [1.0, 2.0]:
        sex_binary = 0.0 if sex == 1.0 else 1.0
    else:
        sex_binary = float(sex)
        
    feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    raw_values = [age, sex_binary, bmi, bp, s1, s2, s3, s4, s5, s6]
    
    # Create DataFrame for scaler and model
    df_input = pd.DataFrame([raw_values], columns=feature_names)
    X_scaled = scaler.transform(df_input)
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Predict quantitative progression using Ridge
    progression_val = float(model.predict(df_scaled)[0])
    
    # Feature-by-feature contribution breakdown: contribution_j = w_j * z_j
    contributions = {}
    for name, weight, z_val in zip(feature_names, model.coef_, X_scaled[0]):
        contributions[name] = {
            'weight': round(float(weight), 3),
            'z_score': round(float(z_val), 3),
            'impact': round(float(weight * z_val), 2)
        }
    
    # Predict Severe Progression Triage probability (>= 200)
    severe_prob = float(triage_model.predict_proba(df_scaled)[0, 1])
    is_severe   = bool(severe_prob >= 0.50)
    
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
        'progression_index': round(progression_val, 2),
        'intercept_w0': round(float(model.intercept_), 2),
        'clinical_stage': stage,
        'severe_risk_probability': round(severe_prob * 100.0, 1),
        'severe_triage_alert': is_severe,
        'clinical_action': action,
        'biomarker_contributions': contributions
    }

if __name__ == '__main__':
    res = predict_patient()
    print("=== DIABETES PROGRESSION PROGNOSIS REPORT ===")
    print(f"Predicted 1-Year Progression Index: {res['progression_index']} (Baseline Intercept: {res['intercept_w0']})")
    print(f"Clinical Stage: {res['clinical_stage']}")
    print(f"Severe Risk Probability: {res['severe_risk_probability']}% (Alert: {res['severe_triage_alert']})")
    print(f"Recommended Clinical Action: {res['clinical_action']}")
    print("\nBiomarker Impact Breakdown (w_j * z_j):")
    for k, v in res['biomarker_contributions'].items():
        print(f"  {k:8s}: weight={v['weight']:+6.2f} | z-score={v['z_score']:+5.2f} | impact={v['impact']:+6.2f}")

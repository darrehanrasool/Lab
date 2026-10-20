import os
import joblib
import numpy as np

def predict(age=0.038, sex=0.05, bmi=0.061, bp=0.021, s1=-0.044, s2=-0.034, s3=-0.043, s4=-0.002, s5=0.019, s6=-0.017):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'diabetes_model.joblib')
    scaler_path = os.path.join(base_dir, 'data', 'processed', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    features = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    features_scaled = scaler.transform(features)
    
    progression_val = float(model.predict(features_scaled)[0])
    
    # Clinical risk categorization based on typical progression ranges (25 to 346)
    if progression_val < 100:
        risk = "Low Progression Risk"
    elif progression_val < 200:
        risk = "Moderate Progression Risk"
    else:
        risk = "High Progression Risk"
        
    return {
        'progression_index': round(progression_val, 2),
        'risk_category': risk,
        'clinical_interpretation': f"Predicted diabetes progression metric is {progression_val:.2f} ({risk})."
    }

if __name__ == '__main__':
    result = predict()
    print("Sample Diabetes Prediction:", result)

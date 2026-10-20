import os
import joblib
import numpy as np

def predict(crim=0.1, zn=18.0, indus=2.31, chas=0, nox=0.538, rm=6.575,
            age=65.2, dis=4.09, rad=1, tax=296, ptratio=15.3, b=396.9, lstat=4.98):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'boston_model.joblib')
    scaler_path = os.path.join(base_dir, 'data', 'processed', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    features_scaled = scaler.transform(features)
    
    pred_val = model.predict(features_scaled)[0]
    return {
        'predicted_price_k': float(pred_val),
        'formatted_price': f"${pred_val * 1000:,.2f}"
    }

if __name__ == '__main__':
    result = predict()
    print("Sample Boston Housing Price Prediction:", result)

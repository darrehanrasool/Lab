import os
import joblib
import numpy as np

def predict(pclass, sex, age, sibsp, parch, fare):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'titanic_model.joblib')
    scaler_path = os.path.join(base_dir, 'data', 'processed', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    sex_num = 1 if str(sex).lower() in ['female', '1', 'f'] else 0
    features = np.array([[pclass, sex_num, age, sibsp, parch, fare]])
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    return {
        'survived': bool(pred == 1),
        'label': 'Survived' if pred == 1 else 'Not Survived',
        'probability_survived': float(prob[1]),
        'probability_not_survived': float(prob[0])
    }

if __name__ == '__main__':
    res = predict(pclass=3, sex='female', age=22, sibsp=1, parch=0, fare=7.25)
    print("Sample Titanic Prediction:", res)

import os
import joblib
import numpy as np

def predict(sepal_length, sepal_width, petal_length, petal_width):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'logistic_regression_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    
    pred_idx = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    return {
        'species': class_names[pred_idx],
        'class_index': int(pred_idx),
        'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    }

if __name__ == '__main__':
    sample = predict(5.1, 3.5, 1.4, 0.2)
    print("Sample Prediction (Expected Setosa):", sample)

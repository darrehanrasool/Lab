import os
import joblib
import numpy as np

def predict(pixel_array=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'mnist_model.joblib')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    if pixel_array is None:
        # Load a default test sample
        test_data_path = os.path.join(base_dir, 'data', 'processed', 'X_test.csv')
        sample = np.genfromtxt(test_data_path, delimiter=',', skip_header=1, max_rows=1)
        # unscale to test raw pipeline
        pixel_array = scaler.inverse_transform([sample])[0]
        
    pixel_array = np.array(pixel_array).flatten().reshape(1, -1)
    if pixel_array.shape[1] != 784:
        raise ValueError(f"Expected 784 pixels, got {pixel_array.shape[1]}")
        
    pixel_scaled = scaler.transform(pixel_array)
    pred_digit = int(model.predict(pixel_scaled)[0])
    probs = model.predict_proba(pixel_scaled)[0]
    
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{'digit': int(d), 'probability': round(float(probs[d]), 4)} for d in top3_idx]
    
    return {
        'predicted_digit': pred_digit,
        'confidence': f"{probs[pred_digit] * 100:.2f}%",
        'top_candidates': top3,
        'all_probabilities': [round(float(p), 4) for p in probs]
    }

if __name__ == '__main__':
    result = predict()
    print("Sample MNIST Digit Prediction:", result)

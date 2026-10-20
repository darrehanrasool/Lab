import os
import joblib
import numpy as np

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def predict(pixel_array=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'fashion_mnist_model.joblib')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    if pixel_array is None:
        # Load a default test sample
        test_data_path = os.path.join(base_dir, 'data', 'processed', 'X_test.csv')
        sample = np.genfromtxt(test_data_path, delimiter=',', skip_header=1, max_rows=1)
        pixel_array = scaler.inverse_transform([sample])[0]
        
    pixel_array = np.array(pixel_array).flatten().reshape(1, -1)
    if pixel_array.shape[1] != 784:
        raise ValueError(f"Expected 784 pixels, got {pixel_array.shape[1]}")
        
    pixel_scaled = scaler.transform(pixel_array)
    pred_idx = int(model.predict(pixel_scaled)[0])
    probs = model.predict_proba(pixel_scaled)[0]
    
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{'class': CLASS_NAMES[i], 'probability': round(float(probs[i]), 4)} for i in top3_idx]
    
    return {
        'predicted_class_id': pred_idx,
        'predicted_category': CLASS_NAMES[pred_idx],
        'confidence': f"{probs[pred_idx] * 100:.2f}%",
        'top_candidates': top3,
        'all_probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(10)}
    }

if __name__ == '__main__':
    result = predict()
    print("Sample Fashion-MNIST Classification:", result)

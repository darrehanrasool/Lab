from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict_fashion():
    try:
        data = request.get_json(force=True)
        pixels = data.get('pixels', [])
        
        if len(pixels) != 784:
            return jsonify({'error': f'Expected 784 pixels, got {len(pixels)}'}), 400
            
        model_path = os.path.join(BASE_DIR, 'models', 'fashion_mnist_model.joblib')
        scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        pixel_array = np.array(pixels, dtype=np.float32).reshape(1, -1)
        pixel_scaled = scaler.transform(pixel_array)
        
        pred_idx = int(model.predict(pixel_scaled)[0])
        probs = model.predict_proba(pixel_scaled)[0]
        
        return jsonify({
            'class_id': pred_idx,
            'category': CLASS_NAMES[pred_idx],
            'confidence': f"{probs[pred_idx] * 100:.1f}%",
            'probabilities': [round(float(p), 4) for p in probs]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample/<int:class_id>', methods=['GET'])
def get_sample(class_id):
    try:
        raw_path = os.path.join(BASE_DIR, 'data', 'raw', 'fashion_mnist.csv')
        df = pd.read_csv(raw_path)
        match = df[df['label'] == class_id].iloc[0]
        pixels = match.drop('label').values.tolist()
        return jsonify({
            'class_id': class_id,
            'category': CLASS_NAMES[class_id],
            'pixels': pixels
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5007)

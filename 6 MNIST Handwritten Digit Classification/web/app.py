from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_digit():
    try:
        data = request.get_json(force=True)
        pixels = data.get('pixels', [])
        
        if len(pixels) != 784:
            return jsonify({'error': f'Expected 784 pixels, got {len(pixels)}'}), 400
            
        model_path = os.path.join(BASE_DIR, 'models', 'mnist_model.joblib')
        scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        pixel_array = np.array(pixels, dtype=np.float32).reshape(1, -1)
        pixel_scaled = scaler.transform(pixel_array)
        
        pred = int(model.predict(pixel_scaled)[0])
        probs = model.predict_proba(pixel_scaled)[0]
        
        return jsonify({
            'prediction': pred,
            'confidence': f"{probs[pred] * 100:.1f}%",
            'probabilities': [round(float(p), 4) for p in probs]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample/<int:digit>', methods=['GET'])
def get_sample(digit):
    try:
        raw_path = os.path.join(BASE_DIR, 'data', 'raw', 'mnist.csv')
        import pandas as pd
        df = pd.read_csv(raw_path)
        match = df[df['label'] == digit].iloc[0]
        pixels = match.drop('label').values.tolist()
        return jsonify({'digit': digit, 'pixels': pixels})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)

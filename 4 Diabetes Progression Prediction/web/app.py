from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    risk_level = None
    input_data = {}
    
    if request.method == 'POST':
        try:
            fields = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
            features = []
            for field in fields:
                val = float(request.form.get(field, 0.0))
                features.append(val)
                input_data[field] = val
                
            model_path = os.path.join(BASE_DIR, 'models', 'diabetes_model.joblib')
            scaler_path = os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            features_scaled = scaler.transform([features])
            pred_val = float(model.predict(features_scaled)[0])
            
            prediction = f"{pred_val:.1f}"
            if pred_val < 100:
                risk_level = "Low Risk (Mild Progression)"
            elif pred_val < 200:
                risk_level = "Moderate Risk (Moderate Progression)"
            else:
                risk_level = "High Risk (Rapid Progression)"
        except Exception as e:
            prediction = f"Error: {str(e)}"
            risk_level = "Calculation Failed"
            
    return render_template('index.html', prediction=prediction, risk_level=risk_level, input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True, port=5003)

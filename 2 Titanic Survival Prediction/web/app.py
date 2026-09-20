import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['pclass']), 
                float(request.form['sex']),
                float(request.form['age']), 
                float(request.form['sibsp']),
                float(request.form['parch']), 
                float(request.form['fare'])
            ]
            
            # Use absolute paths to load the model and scaler
            model_path = os.path.join(BASE_DIR, 'models', 'titanic_model.joblib')
            scaler_path = os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]
            prediction = "Survived" if pred == 1 else "Did Not Survive"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
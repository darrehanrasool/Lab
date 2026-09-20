from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Absolute path handling to prevent "File Not Found" errors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Taking 5 main features for the form, padding the rest with 0.0
            features = [
                float(request.form['crim']), float(request.form['zn']),
                float(request.form['indus']), float(request.form['chas']),
                float(request.form['nox']), float(request.form['rm']),
                float(request.form['age']), float(request.form['dis']),
                float(request.form['rad']), float(request.form['tax']),
                float(request.form['ptratio']), float(request.form['b']),
                float(request.form['lstat'])
            ]
            
            model_path = os.path.join(BASE_DIR, 'models', 'boston_model.joblib')
            scaler_path = os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]
            prediction = f"${pred * 1000:,.2f}" # Boston dataset target is in $1000s
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
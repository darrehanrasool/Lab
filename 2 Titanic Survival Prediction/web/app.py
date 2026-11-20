"""
========================================================================================
PROJECT 2: TITANIC SURVIVAL PREDICTION
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: RMS Titanic Passenger Manifest (N = 891 Samples, Binary Classification)

PURPOSE:
--------
Serves as the deployment and interaction interface for Project 2:
1. Web UI: Interactive passenger survival prognostic dashboard at GET/POST /.
2. REST API: Programmatic endpoint at POST /predict for clinical or external integration.
3. In-Memory Asset Caching: Loads the Random Forest and StandardScaler once at application startup.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why cache the model globally at startup instead of inside the `home()` route?
    -> Loading from disk on every POST request introduces 40-80ms of disk I/O latency. Pre-loading
       into RAM gives instant sub-millisecond response times.
========================================================================================
"""

import os
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib
import pandas as pd

app = Flask(__name__)

# Resolve absolute paths to persistent assets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "titanic_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.joblib")

# Load model and scaler into RAM on startup (zero cold-start latency)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Successfully loaded Titanic Random Forest model and scaler into RAM.")
except Exception as err:
    print(f"Warning: Failed to load Titanic assets: {err}")
    model = None
    scaler = None

FEATURE_NAMES = ["pclass", "sex", "age", "sibsp", "parch", "fare"]


@app.route("/", methods=["GET", "POST"])
def home():
    """Renders the HTML web interface and handles browser form submissions."""
    prediction = None
    prob_survived = None
    if request.method == "POST":
        try:
            pclass = float(request.form["pclass"])
            sex = float(request.form["sex"])
            age = float(request.form["age"])
            sibsp = float(request.form["sibsp"])
            parch = float(request.form["parch"])
            fare = float(request.form["fare"])

            # Format into DataFrame with correct feature names
            df_input = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], columns=FEATURE_NAMES)
            features_scaled = scaler.transform(df_input)
            df_scaled = pd.DataFrame(features_scaled, columns=FEATURE_NAMES)

            pred = int(model.predict(df_scaled)[0])
            prob = model.predict_proba(df_scaled)[0]

            prediction = "Survived" if pred == 1 else "Did Not Survive"
            prob_survived = round(float(prob[1]) * 100.0, 1)
        except Exception as err:
            prediction = f"Error: {err!s}"

    return render_template("index.html", prediction=prediction, prob_survived=prob_survived)


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic passenger survival prediction.
    Accepts JSON: { pclass, sex, age, sibsp, parch, fare }
    Returns JSON with survival prediction and probability distribution.
    """
    try:
        data = request.get_json(force=True)
        sex_val = 1.0 if str(data.get("sex", "male")).strip().lower() in ["female", "1", "f"] else 0.0
        df_input = pd.DataFrame([[
            float(data.get("pclass", 3)),
            sex_val,
            float(data.get("age", 28.0)),
            float(data.get("sibsp", 0)),
            float(data.get("parch", 0)),
            float(data.get("fare", 14.45)),
        ]], columns=FEATURE_NAMES)

        features_scaled = scaler.transform(df_input)
        df_scaled = pd.DataFrame(features_scaled, columns=FEATURE_NAMES)
        pred = int(model.predict(df_scaled)[0])
        probabilities = model.predict_proba(df_scaled)[0]

        return jsonify({
            "status": "success",
            "prediction": "Survived" if pred == 1 else "Not Survived",
            "survived": bool(pred == 1),
            "probability_survived": round(float(probabilities[1]) * 100.0, 2),
            "probability_perished": round(float(probabilities[0]) * 100.0, 2),
        }), 200
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

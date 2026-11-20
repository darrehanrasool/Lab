"""
========================================================================================
PROJECT 3: BOSTON HOUSING PRICE PREDICTION
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Harrison & Rubinfeld (1978) Boston Housing Cohort (N = 506 Census Tracts)

PURPOSE:
--------
Serves as the deployment and interaction interface for Project 3:
1. Web UI: Interactive property valuation dashboard at GET/POST /.
2. REST API: Programmatic endpoint at POST /predict for real estate integration.
3. Pre-loads model and scaler into memory on startup for instant inference.
========================================================================================
"""

import os
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib
import pandas as pd

app = Flask(__name__)

# Absolute path handling to prevent "File Not Found" errors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "boston_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.joblib")

# Load model and scaler into RAM on startup (zero cold-start latency)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Successfully loaded Boston Housing OLS model and scaler into RAM.")
except Exception as err:
    print(f"Warning: Failed to load Boston Housing assets: {err}")
    model = None
    scaler = None

FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]


@app.route("/", methods=["GET", "POST"])
def home():
    """Renders the HTML web interface and handles browser form submissions."""
    prediction = None
    if request.method == "POST":
        try:
            # Parse all 13 features from form input
            features = [
                float(request.form.get("crim", 0.1)),
                float(request.form.get("zn", 18.0)),
                float(request.form.get("indus", 2.31)),
                float(request.form.get("chas", 0.0)),
                float(request.form.get("nox", 0.538)),
                float(request.form.get("rm", 6.575)),
                float(request.form.get("age", 65.2)),
                float(request.form.get("dis", 4.09)),
                float(request.form.get("rad", 1.0)),
                float(request.form.get("tax", 296.0)),
                float(request.form.get("ptratio", 15.3)),
                float(request.form.get("b", 396.9)),
                float(request.form.get("lstat", 4.98)),
            ]

            df_input = pd.DataFrame([features], columns=FEATURE_NAMES)
            features_scaled = scaler.transform(df_input)
            df_scaled = pd.DataFrame(features_scaled, columns=FEATURE_NAMES)

            pred = float(model.predict(df_scaled)[0])
            prediction = f"${pred * 1000:,.2f}"  # Boston target is in $1,000s
        except Exception as err:
            prediction = f"Error: {err!s}"

    return render_template("index.html", prediction=prediction)


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic property valuation.
    Accepts JSON with any or all 13 features.
    Returns JSON with predicted price in thousands and formatted dollars.
    """
    try:
        data = request.get_json(force=True)
        features = [
            float(data.get("crim", 0.1)),
            float(data.get("zn", 18.0)),
            float(data.get("indus", 2.31)),
            float(data.get("chas", 0.0)),
            float(data.get("nox", 0.538)),
            float(data.get("rm", 6.575)),
            float(data.get("age", 65.2)),
            float(data.get("dis", 4.09)),
            float(data.get("rad", 1.0)),
            float(data.get("tax", 296.0)),
            float(data.get("ptratio", 15.3)),
            float(data.get("b", 396.9)),
            float(data.get("lstat", 4.98)),
        ]

        df_input = pd.DataFrame([features], columns=FEATURE_NAMES)
        features_scaled = scaler.transform(df_input)
        df_scaled = pd.DataFrame(features_scaled, columns=FEATURE_NAMES)

        pred = float(model.predict(df_scaled)[0])

        return jsonify({
            "status": "success",
            "predicted_price_k": round(pred, 2),
            "formatted_price": f"${pred * 1000:,.2f}",
        }), 200
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)

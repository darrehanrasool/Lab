"""
========================================================================================
PROJECT 1: IRIS SPECIES CLASSIFICATION
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Fisher's Famous 1936 Iris Flower Dataset (N = 150 Samples, 3 Classes)

PURPOSE:
--------
Serves as the interactive web deployment layer for Project 1:
1. Web UI: Renders interactive flower classification form at GET /.
2. REST API: Programmatic endpoint at POST /predict returning species and Softmax probabilities.
3. Pre-loads model and scaler into memory on startup for instant inference.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why use `request.get_json()` in `/predict`?
    -> It parses incoming JSON payloads sent by asynchronous AJAX/Fetch calls from the frontend
       without requiring a full page refresh.
Q2: What is Softmax?
    -> Softmax converts raw linear logits (z_k = w_k^T * x + b_k) into normalized probabilities
       that sum to 100%, allowing the web UI to show confidence percentages for each species.
========================================================================================
"""

import os
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Resolve model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# In-memory asset caching at startup
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Class species names
SPECIES = ["Setosa", "Versicolor", "Virginica"]


@app.route("/")
def index() -> str:
    """Renders the HTML user interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API endpoint for programmatic inference.
    Accepts JSON: { sepal_length, sepal_width, petal_length, petal_width }
    Returns JSON with predicted species and Softmax probability distribution.
    """
    try:
        data = request.get_json(force=True)
        # Parse 4 morphological measurements into DataFrame matching scaler schema
        feature_names = getattr(scaler, "feature_names_in_", ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
        df_features = pd.DataFrame([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"]),
        ]], columns=feature_names)

        # Scale features using frozen training statistics
        features_scaled = scaler.transform(df_features)

        # Predict class index and probabilities
        pred_idx = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]

        return jsonify({
            "status": "success",
            "prediction": SPECIES[pred_idx],
            "class_index": pred_idx,
            "probabilities": {
                SPECIES[i]: round(float(probabilities[i]) * 100, 2)
                for i in range(3)
            },
        }), 200
    except (ValueError, KeyError, TypeError, OSError) as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

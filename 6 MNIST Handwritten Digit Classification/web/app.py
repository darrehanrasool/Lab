"""
========================================================================================
PROJECT 6: MNIST HANDWRITTEN DIGIT CLASSIFICATION
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: LeCun et al. (1998) Modified National Institute of Standards & Technology (MNIST)

PURPOSE:
--------
Serves as the interactive canvas web deployment layer for Project 6:
1. Web UI: Interactive HTML5 canvas drawing pad at GET /.
2. REST API: Programmatic endpoint at POST /predict accepting 784 pixel floats and returning digit classification.
3. Pre-loads the neural network and scaler into memory at startup for zero latency.
========================================================================================
"""

import os
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# Load neural network and scaler into RAM on startup (zero cold-start latency)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Successfully loaded MNIST Neural Network and scaler into RAM.")
except Exception as err:
    print(f"Warning: Failed to load MNIST assets: {err}")
    model = None
    scaler = None

PIXEL_COLS = [f"pixel_{i}" for i in range(784)]


@app.route("/", methods=["GET"])
def home() -> str:
    """Renders the HTML5 canvas digit drawing interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_digit():
    """
    REST API endpoint for digit image recognition.
    Accepts JSON: { "pixels": [p_0, p_1, ..., p_783] }
    Returns JSON with predicted digit and probability distribution.
    """
    try:
        data = request.get_json(force=True)
        pixels = data.get("pixels", [])

        if len(pixels) != 784:
            return jsonify({"status": "error", "message": f"Expected 784 pixels, got {len(pixels)}"}), 400

        # Construct DataFrame matching training schema
        df_pixels = pd.DataFrame([pixels], columns=PIXEL_COLS)
        pixel_scaled = scaler.transform(df_pixels)
        df_scaled = pd.DataFrame(pixel_scaled, columns=PIXEL_COLS)

        pred = int(model.predict(df_scaled)[0])
        probs = model.predict_proba(df_scaled)[0]

        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [{"digit": int(d), "probability": round(float(probs[d]) * 100.0, 2)} for d in top3_idx]

        return jsonify({
            "status": "success",
            "digit": pred,
            "confidence": f"{float(probs[pred]) * 100.0:.1f}%",
            "top3": top3,
            "probabilities": [round(float(p) * 100.0, 2) for p in probs],
        }), 200
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False)

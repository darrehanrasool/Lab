"""
========================================================================================
PROJECT 5: SPAM EMAIL / SMS DETECTION
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: UCI / Almeida et al. SMS & Email Spam Collection (N = 5,572 Messages)

PURPOSE:
--------
Serves as the deployment and interaction interface for Project 5:
1. Web UI: Interactive email scanner interface at GET/POST /.
2. REST API: Programmatic endpoint at POST /predict for mail server integrations.
3. Pre-loads Multinomial Naive Bayes model and TfidfVectorizer into memory at boot time.
========================================================================================
"""

import os
import re
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib

app = Flask(__name__)

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")

# Load model and vectorizer into RAM on startup (zero cold-start latency)
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ Successfully loaded Spam Detection Naive Bayes model and TF-IDF vectorizer into RAM.")
except Exception as err:
    print(f"Warning: Failed to load Spam Detection assets: {err}")
    model = None
    vectorizer = None


def clean_text(text: str) -> str:
    """Standardizes input string by stripping URLs, digits, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    """Renders the HTML web interface and handles browser message submissions."""
    prediction = None
    confidence = None
    spam_prob = None
    message_text = ""

    if request.method == "POST":
        message_text = request.form.get("message", "").strip()
        if message_text:
            try:
                cleaned = clean_text(message_text)
                features = vectorizer.transform([cleaned])
                pred = int(model.predict(features)[0])
                probs = model.predict_proba(features)[0]

                prediction = "SPAM" if pred == 1 else "HAM (Legitimate)"
                confidence = f"{float(max(probs)) * 100:.1f}%"
                spam_prob = round(float(probs[1]) * 100.0, 1)
            except Exception as err:
                prediction = f"Error: {err!s}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        spam_prob=spam_prob,
        message=message_text,
    )


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic email spam classification.
    Accepts JSON: { "message": "Text to analyze..." }
    Returns JSON with classification status and posterior probability.
    """
    try:
        data = request.get_json(force=True)
        raw_text = data.get("message", "")
        cleaned = clean_text(raw_text)
        features = vectorizer.transform([cleaned])

        pred = int(model.predict(features)[0])
        probs = model.predict_proba(features)[0]

        is_spam = bool(pred == 1)
        return jsonify({
            "status": "success",
            "is_spam": is_spam,
            "label": "SPAM" if is_spam else "HAM",
            "spam_probability": round(float(probs[1]) * 100.0, 2),
            "ham_probability": round(float(probs[0]) * 100.0, 2),
            "confidence": f"{float(max(probs)) * 100.0:.2f}%",
        }), 200
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=False)

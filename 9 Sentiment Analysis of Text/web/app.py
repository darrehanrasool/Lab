"""
========================================================================================
PROJECT 9: SENTIMENT ANALYSIS OF TEXT
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Maas et al. (2011) Stanford IMDB Large Movie Review Dataset (N = 6,000 Reviews)

PURPOSE:
--------
Serves as the interactive web deployment layer for Project 9:
1. Web UI: Interactive sentiment scanner dashboard at GET/POST /.
2. REST API: Programmatic endpoint at POST /predict for opinion mining integrations.
3. Pre-loads the Logistic Regression model and Sublinear TF-IDF vectorizer into memory at boot time.
========================================================================================
"""

import os
import re
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")

# Load model and vectorizer into RAM on startup (zero cold-start latency)
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ Successfully loaded Sentiment Logistic Regression model and TF-IDF vectorizer into RAM.")
except Exception as err:
    print(f"Warning: Failed to load Sentiment assets: {err}")
    model = None
    vectorizer = None


def clean_text(text: str) -> str:
    """Standardizes input string by stripping HTML breaks, URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    """Renders the HTML web interface and handles browser text submissions."""
    prediction = None
    confidence = None
    pos_prob = None
    neg_prob = None
    polarity_score = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()
        if review_text:
            try:
                cleaned = clean_text(review_text)
                features = vectorizer.transform([cleaned])
                pred = int(model.predict(features)[0])
                probs = model.predict_proba(features)[0]

                prediction = "POSITIVE" if pred == 1 else "NEGATIVE"
                confidence = f"{float(max(probs)) * 100:.1f}%"
                pos_prob = round(float(probs[1]) * 100.0, 1)
                neg_prob = round(float(probs[0]) * 100.0, 1)
                polarity_score = round(float(probs[1] - probs[0]), 3)
            except Exception as err:
                prediction = f"Error: {err!s}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        pos_prob=pos_prob,
        neg_prob=neg_prob,
        polarity_score=polarity_score,
        review=review_text,
    )


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic sentiment classification.
    Accepts JSON: { "review": "Text to analyze..." }
    Returns JSON with sentiment label, polarity score, and probabilities.
    """
    try:
        data = request.get_json(force=True)
        raw_text = data.get("review", "")
        cleaned = clean_text(raw_text)
        features = vectorizer.transform([cleaned])

        pred = int(model.predict(features)[0])
        probs = model.predict_proba(features)[0]

        is_positive = bool(pred == 1)
        return jsonify({
            "status": "success",
            "is_positive": is_positive,
            "sentiment": "POSITIVE" if is_positive else "NEGATIVE",
            "positive_probability": round(float(probs[1]) * 100.0, 2),
            "negative_probability": round(float(probs[0]) * 100.0, 2),
            "confidence": f"{float(max(probs)) * 100.0:.1f}%",
            "polarity_score": round(float(probs[1] - probs[0]), 4),
        }), 200
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, debug=False)

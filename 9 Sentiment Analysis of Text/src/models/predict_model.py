"""
========================================================================================
PROJECT 9: SENTIMENT ANALYSIS OF TEXT
MODULE 4: REAL-TIME OPINION INFERENCE ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Maas et al. (2011) Stanford IMDB Large Movie Review Dataset (N = 6,000 Reviews)

PURPOSE:
--------
Provides offline command-line prediction for classifying the sentiment of any raw movie review or text string:
1. Loads the serialized Logistic Regression classifier and fitted Sublinear TfidfVectorizer.
2. Applies the exact regex cleaning pipeline used during training.
3. Computes the Sigmoid probability score P(Positive | text).
4. Outputs human-readable label (POSITIVE vs NEGATIVE), probability percentages, and polarity confidence.
========================================================================================
"""

import os
import re
from typing import Any, Dict
import joblib


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


def predict(
    text: str = "This film was an absolute masterpiece with phenomenal acting and breathtaking cinematography!",
) -> Dict[str, Any]:
    """
    Classifies raw review text as POSITIVE or NEGATIVE.

    Args:
        text (str): Raw natural language review or user feedback.

    Returns:
        Dict[str, Any]: Contains binary flag, sentiment label, and probability distribution.
    """
    # 1. Resolve project file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "sentiment_model.joblib")
    vectorizer_path = os.path.join(base_dir, "models", "vectorizer.joblib")

    # 2. Deserialize model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # 3. Clean input text
    cleaned = clean_text(text)

    # 4. Transform to Sublinear TF-IDF features
    features = vectorizer.transform([cleaned])

    # 5. Forward inference
    pred = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]

    is_positive = bool(pred == 1)
    return {
        "sentiment": "POSITIVE" if is_positive else "NEGATIVE",
        "is_positive": is_positive,
        "confidence": f"{float(max(probs)) * 100:.1f}%",
        "positive_probability": round(float(probs[1]), 4),
        "negative_probability": round(float(probs[0]), 4),
    }


if __name__ == "__main__":
    sample_review = "The movie started with great potential, but quickly collapsed into incoherent writing, dreadful pacing, and uninspired performances."
    res = predict(sample_review)
    print("=== PROJECT 9: SENTIMENT ANALYSIS REPORT ===")
    print(f"Review Content: '{sample_review}'")
    print(f"Sentiment:      {res['sentiment']} (Positive: {res['is_positive']})")
    print(f"Confidence:     {res['confidence']}")
    print(f"Posterior Probabilities:")
    print(f"  - Positive Probability: {res['positive_probability'] * 100:.2f}%")
    print(f"  - Negative Probability: {res['negative_probability'] * 100:.2f}%")

"""
========================================================================================
PROJECT 5: SPAM EMAIL / SMS DETECTION
MODULE 4: REAL-TIME TEXT CLASSIFICATION INFERENCE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: UCI / Almeida et al. SMS & Email Spam Collection (N = 5,572 Messages)

PURPOSE:
--------
Provides offline command-line prediction for classifying any raw email or SMS text:
1. Loads the serialized Multinomial Naive Bayes model and fitted TfidfVectorizer.
2. Cleans raw user input using the exact regex pipeline used during training.
3. Transforms text to a sparse TF-IDF feature vector over the 3,000-term vocabulary.
4. Outputs binary classification (SPAM vs HAM), class probabilities, and confidence score.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What happens if a user submits a sentence with words not in the training vocabulary?
    -> Out-of-vocabulary (OOV) words are simply ignored by the vectorizer (assigned weight 0).
       Known keywords in the sentence (like 'cash', 'free', 'win') still activate their corresponding
       TF-IDF weights. Laplace smoothing guarantees the posterior probability remains mathematically sound.
========================================================================================
"""

import os
import re
from typing import Any, Dict
import joblib


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


def predict(
    email_text: str = "Congratulations! You have won a $1000 Walmart gift card. Click here to claim your prize.",
) -> Dict[str, Any]:
    """
    Classifies a raw text message as SPAM or HAM.

    Args:
        email_text (str): Raw email or SMS message content.

    Returns:
        Dict[str, Any]: Contains boolean is_spam flag, label, and probability distribution.
    """
    # 1. Resolve project file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "spam_model.joblib")
    vectorizer_path = os.path.join(base_dir, "models", "vectorizer.joblib")

    # 2. Deserialize model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # 3. Clean raw text using training pipeline
    cleaned = clean_text(email_text)

    # 4. Transform to TF-IDF feature representation
    features = vectorizer.transform([cleaned])

    # 5. Predict discrete label and posterior class probabilities
    pred = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]

    is_spam = bool(pred == 1)
    return {
        "is_spam": is_spam,
        "label": "SPAM" if is_spam else "HAM (Legitimate)",
        "spam_probability": round(float(probs[1]), 4),
        "ham_probability": round(float(probs[0]), 4),
        "confidence_percentage": f"{float(max(probs)) * 100:.1f}%",
    }


if __name__ == "__main__":
    # Test sample with typical promotional spam patterns
    sample_text = "URGENT! You have won a guaranteed £1000 cash prize. Call 09061701461 to claim your reward."
    res = predict(sample_text)
    print("=== PROJECT 5: SPAM EMAIL CLASSIFICATION REPORT ===")
    print(f"Message Content: '{sample_text}'")
    print(f"Classification:  {res['label']} (Spam: {res['is_spam']})")
    print(f"Confidence:      {res['confidence_percentage']}")
    print(f"Posterior Probabilities:")
    print(f"  - Spam Probability: {res['spam_probability'] * 100:.2f}%")
    print(f"  - Ham Probability:  {res['ham_probability'] * 100:.2f}%")

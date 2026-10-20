import os
import re
import joblib

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text="This film was an absolute masterpiece with phenomenal acting and breathtaking cinematography!"):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'sentiment_model.joblib')
    vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.joblib')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    
    pred = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]
    
    is_positive = bool(pred == 1)
    return {
        'sentiment': 'POSITIVE' if is_positive else 'NEGATIVE',
        'is_positive': is_positive,
        'confidence': f"{max(probs) * 100:.1f}%",
        'positive_probability': round(float(probs[1]), 4),
        'negative_probability': round(float(probs[0]), 4),
        'polarity_score': round(float(probs[1] - probs[0]), 4)
    }

if __name__ == '__main__':
    pos = predict("Incredible movie, brilliant performance and deeply moving script!")
    print("Positive Sample:", pos)
    
    neg = predict("Complete waste of time, utterly boring and terrible plot.")
    print("Negative Sample:", neg)

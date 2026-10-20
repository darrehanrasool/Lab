import os
import re
import joblib

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(email_text="Congratulations! You have won a $1000 Walmart gift card. Click here to claim your prize."):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'spam_model.joblib')
    vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.joblib')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    cleaned = clean_text(email_text)
    features = vectorizer.transform([cleaned])
    
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    is_spam = bool(pred == 1)
    return {
        'is_spam': is_spam,
        'label': 'SPAM' if is_spam else 'HAM (Legitimate)',
        'spam_probability': round(float(probs[1]), 4),
        'ham_probability': round(float(probs[0]), 4),
        'confidence_percentage': f"{max(probs) * 100:.1f}%"
    }

if __name__ == '__main__':
    sample = predict()
    print("Sample Spam Prediction:", sample)
    
    legit_sample = predict("Hey John, are we still meeting tomorrow at 10 AM for the lab session?")
    print("Sample Legitimate Prediction:", legit_sample)

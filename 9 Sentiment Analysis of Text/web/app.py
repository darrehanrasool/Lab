from flask import Flask, render_template, request
import joblib
import os
import re

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    pos_prob = None
    neg_prob = None
    polarity_score = None
    review_text = ""
    
    if request.method == 'POST':
        review_text = request.form.get('review', '').strip()
        if review_text:
            try:
                model_path = os.path.join(BASE_DIR, 'models', 'sentiment_model.joblib')
                vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.joblib')
                
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                
                cleaned = clean_text(review_text)
                features = vectorizer.transform([cleaned])
                
                pred = int(model.predict(features)[0])
                probs = model.predict_proba(features)[0]
                
                prediction = "POSITIVE" if pred == 1 else "NEGATIVE"
                confidence = f"{max(probs) * 100:.1f}%"
                pos_prob = f"{probs[1] * 100:.1f}%"
                neg_prob = f"{probs[0] * 100:.1f}%"
                polarity_score = round(float(probs[1] - probs[0]), 2)
            except Exception as e:
                prediction = f"Error: {str(e)}"
                confidence = "N/A"
                
    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           pos_prob=pos_prob,
                           neg_prob=neg_prob,
                           polarity_score=polarity_score,
                           review_text=review_text)

if __name__ == '__main__':
    app.run(debug=True, port=5008)

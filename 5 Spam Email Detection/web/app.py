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
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    spam_prob = None
    message_text = ""
    
    if request.method == 'POST':
        message_text = request.form.get('message', '').strip()
        if message_text:
            try:
                model_path = os.path.join(BASE_DIR, 'models', 'spam_model.joblib')
                vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.joblib')
                
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                
                cleaned = clean_text(message_text)
                features = vectorizer.transform([cleaned])
                
                pred = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                
                prediction = "SPAM" if pred == 1 else "HAM (Legitimate)"
                confidence = f"{max(probs) * 100:.1f}%"
                spam_prob = f"{probs[1] * 100:.1f}%"
            except Exception as e:
                prediction = f"Error: {str(e)}"
                confidence = "N/A"
                spam_prob = "N/A"
                
    return render_template('index.html', prediction=prediction, confidence=confidence,
                           spam_prob=spam_prob, message_text=message_text)

if __name__ == '__main__':
    app.run(debug=True, port=5004)

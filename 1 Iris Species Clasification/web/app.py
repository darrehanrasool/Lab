# app.py – Flask backend for Iris prediction
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Class names
SPECIES = ["Setosa", "Versicolor", "Virginica"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the form
        data = request.get_json()
        features = np.array([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]])
        # Scale the features using the same scaler
        features_scaled = scaler.transform(features)
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        return jsonify({
            "prediction": SPECIES[prediction],
            "probabilities": {
                SPECIES[i]: round(float(probability[i]) * 100, 2)
                for i in range(3)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
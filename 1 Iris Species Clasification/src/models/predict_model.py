"""
========================================================================================
PROJECT 1: IRIS SPECIES CLASSIFICATION
MODULE 4: REAL-TIME INFERENCE & PROBABILITY ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Fisher's Famous 1936 Iris Flower Dataset (N = 150 Samples, 3 Classes)

PURPOSE:
--------
Provides offline command-line inference for classifying any single iris flower sample:
1. Loads the serialized Multinomial Logistic Regression model and StandardScaler.
2. Takes 4 continuous physical measurements (Sepal Length, Sepal Width, Petal Length, Petal Width).
3. Standardizes inputs using frozen training parameters (z = (x - mu) / sigma).
4. Computes class index and calibrated Softmax probabilities across all 3 flower classes.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How does Logistic Regression handle 3 classes (Multiclass)?
    -> It uses Multinomial Logistic Regression with the Softmax activation function:
           P(y = k | x) = exp(w_k^T * x + b_k) / sum_{j=1}^3 exp(w_j^T * x + b_j)
       This ensures the 3 predicted probabilities are non-negative and sum to exactly 1.0 (100%).

Q2: Why must the single test sample pass through `scaler.transform()`?
    -> The model weights were trained on standardized z-scores. Passing unscaled centimeters
       directly to the dot product would cause massive prediction errors.
========================================================================================
"""

import os
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd


def predict(
    sepal_length: float = 5.1,
    sepal_width: float = 3.5,
    petal_length: float = 1.4,
    petal_width: float = 0.2,
) -> Dict[str, Any]:
    """
    Predicts the Iris flower species given 4 botanical measurements.

    Args:
        sepal_length (float): Sepal length in centimeters.
        sepal_width (float): Sepal width in centimeters.
        petal_length (float): Petal length in centimeters.
        petal_width (float): Petal width in centimeters.

    Returns:
        Dict[str, Any]: Contains predicted species name, integer index, and class probabilities.
    """
    # 1. Resolve project root and locate serialized artifacts
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "logistic_regression_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

    # 2. Deserialize model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Format input into DataFrame with feature names to match scaler schema
    feature_names = getattr(scaler, "feature_names_in_", ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
    df_features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

    # 4. Standardize using frozen training statistics (mu, sigma)
    features_scaled = scaler.transform(df_features)

    # 5. Predict discrete class index (Argmax over Softmax probabilities)
    pred_idx = int(model.predict(features_scaled)[0])

    # 6. Extract calibrated Softmax posterior probabilities: P(Class_k | Features)
    probabilities = model.predict_proba(features_scaled)[0]

    class_names = ["Setosa", "Versicolor", "Virginica"]
    return {
        "species": class_names[pred_idx],
        "class_index": pred_idx,
        "probabilities": {class_names[i]: float(prob) for i, prob in enumerate(probabilities)},
    }


if __name__ == "__main__":
    # Test sample with textbook Setosa characteristics
    sample = predict(5.1, 3.5, 1.4, 0.2)
    print("=== PROJECT 1: IRIS FLOWER PREDICTION ===")
    print("Inputs: Sepal [5.1, 3.5] cm, Petal [1.4, 0.2] cm")
    print(f"Predicted Species: {sample['species']} (Class Index: {sample['class_index']})")
    print("Softmax Probabilities:")
    for sp, prob in sample["probabilities"].items():
        print(f"  - {sp:10s}: {prob * 100:.2f}%")

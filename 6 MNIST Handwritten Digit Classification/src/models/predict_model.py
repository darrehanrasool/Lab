"""
========================================================================================
PROJECT 6: MNIST HANDWRITTEN DIGIT CLASSIFICATION
MODULE 4: REAL-TIME IMAGE INFERENCE ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: LeCun et al. (1998) Modified National Institute of Standards & Technology (MNIST)

PURPOSE:
--------
Provides offline command-line prediction for classifying any single 28x28 handwritten digit image:
1. Loads the serialized Multi-Layer Perceptron neural network and StandardScaler.
2. Accepts a 784-element array of grayscale pixel intensities (or loads an authentic test sample).
3. Standardizes input using frozen training parameters (z = (x - mu) / sigma).
4. Computes forward pass, outputs top predicted digit, confidence percentage, and top-3 candidate rankings.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How does the network output probabilities?
    -> The final layer applies the Softmax function over the 10 raw logits:
           P(digit = k | x) = exp(z_k) / sum_{j=0}^9 exp(z_j)
       The sum of all 10 probabilities is guaranteed to equal exactly 1.0 (100%).
========================================================================================
"""

import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd


def predict(pixel_array: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Predicts the handwritten digit given 784 grayscale pixel intensity values.

    Args:
        pixel_array (Optional[List[float]]): 784-element list/array of pixels in [0, 255].
            If None, loads a default unseen test image from disk.

    Returns:
        Dict[str, Any]: Contains predicted digit, confidence, top 3 candidates, and full probabilities.
    """
    # 1. Resolve project file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "mnist_model.joblib")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")

    # 2. Deserialize model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Handle default test sample fallback
    if pixel_array is None:
        test_data_path = os.path.join(base_dir, "data", "processed", "X_test.csv")
        sample = np.genfromtxt(test_data_path, delimiter=",", skip_header=1, max_rows=1)
        # Unscale back to raw pixel scale [0, 255] for demonstration
        pixel_array = scaler.inverse_transform([sample])[0]

    # Validate dimensionality: exactly 784 pixels
    flat_pixels = np.array(pixel_array, dtype=np.float32).flatten()
    if len(flat_pixels) != 784:
        raise ValueError(f"Dimensionality Error: Expected 784 pixels, got {len(flat_pixels)}")

    # 4. Standardize using training statistics
    pixel_cols = getattr(scaler, "feature_names_in_", [f"pixel_{i}" for i in range(784)])
    df_raw = pd.DataFrame([flat_pixels], columns=pixel_cols)
    pixel_scaled = scaler.transform(df_raw)
    df_scaled = pd.DataFrame(pixel_scaled, columns=pixel_cols)

    # 5. Forward pass inference
    pred_digit = int(model.predict(df_scaled)[0])
    probs = model.predict_proba(df_scaled)[0]

    # 6. Extract Top-3 most probable digits
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{"digit": int(d), "probability": round(float(probs[d]), 4)} for d in top3_idx]

    return {
        "predicted_digit": pred_digit,
        "confidence": f"{float(probs[pred_digit]) * 100:.2f}%",
        "top_candidates": top3,
        "all_probabilities": [round(float(p), 4) for p in probs],
    }


if __name__ == "__main__":
    result = predict()
    print("=== PROJECT 6: MNIST DIGIT CLASSIFICATION REPORT ===")
    print(f"Predicted Digit: {result['predicted_digit']}")
    print(f"Model Confidence: {result['confidence']}")
    print("Top-3 Probable Digits:")
    for rank, cand in enumerate(result["top_candidates"], 1):
        print(f"  {rank}. Digit '{cand['digit']}': {cand['probability'] * 100:.2f}%")

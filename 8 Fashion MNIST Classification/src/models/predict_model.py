"""
========================================================================================
PROJECT 8: FASHION-MNIST CLOTHING CLASSIFICATION
MODULE 4: REAL-TIME APPAREL INFERENCE ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Xiao, Rasul & Vollgraf (2017) Zalando Fashion-MNIST (10 Clothing Categories)

PURPOSE:
--------
Provides offline command-line prediction for classifying any single 28x28 clothing article:
1. Loads the serialized Multi-Layer Perceptron neural network and StandardScaler.
2. Accepts a 784-element array of grayscale pixel intensities (or loads an authentic test sample).
3. Standardizes inputs using frozen training parameters (z = (x - mu) / sigma).
4. Computes forward pass, outputs top predicted clothing category, confidence percentage, and top-3 candidate rankings.
========================================================================================
"""

import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def predict(pixel_array: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Predicts the clothing category given 784 grayscale pixel intensity values.

    Args:
        pixel_array (Optional[List[float]]): 784-element list/array of pixels in [0, 255].
            If None, loads a default unseen test image from disk.

    Returns:
        Dict[str, Any]: Contains predicted category name, confidence, top 3 candidates, and full probabilities.
    """
    # 1. Resolve project file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "fashion_mnist_model.joblib")
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
    pred_idx = int(model.predict(df_scaled)[0])
    probs = model.predict_proba(df_scaled)[0]

    # 6. Extract Top-3 most probable clothing classes
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{"class": CLASS_NAMES[i], "probability": round(float(probs[i]), 4)} for i in top3_idx]

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "class_index": pred_idx,
        "confidence": f"{float(probs[pred_idx]) * 100:.2f}%",
        "top_candidates": top3,
        "all_probabilities": [round(float(p), 4) for p in probs],
    }


if __name__ == "__main__":
    result = predict()
    print("=== PROJECT 8: FASHION-MNIST CLASSIFICATION REPORT ===")
    print(f"Predicted Category: {result['predicted_class']} (Class {result['class_index']})")
    print(f"Model Confidence:   {result['confidence']}")
    print("Top-3 Probable Categories:")
    for rank, cand in enumerate(result["top_candidates"], 1):
        print(f"  {rank}. {cand['class']:12s}: {cand['probability'] * 100:.2f}%")

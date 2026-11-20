"""
========================================================================================
PROJECT 2: TITANIC SURVIVAL PREDICTION
MODULE 4: REAL-TIME PASSENGER INFERENCE ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: RMS Titanic Passenger Manifest (N = 891 Samples, Binary Classification)

PURPOSE:
--------
Provides offline command-line prediction for any single Titanic passenger:
1. Loads the serialized Random Forest Classifier and StandardScaler.
2. Accepts passenger socio-demographic features (Class, Sex, Age, Siblings/Spouses, Parents/Children, Fare).
3. Encodes gender and standardizes numerical vitals using frozen training statistics.
4. Returns binary prediction (Survived vs Not Survived) and posterior probability estimates.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How does a Random Forest output a probability estimate?
    -> Each of the 100 decision trees outputs a binary vote (0 or 1). The predicted probability
       P(Survived = 1 | X) is the fraction of individual trees voting 'Survived' (e.g., 82 / 100 = 0.82).

Q2: Why must the input be scaled before passing into the Random Forest?
    -> While raw decision trees are technically scale-invariant (they split on single features),
       our pipeline trained the forest on standardized features (mean=0, std=1) to ensure consistency
       and compatibility with other linear/distance-based models. Therefore, inference inputs must
       match the exact training feature space.
========================================================================================
"""

import os
from typing import Any, Dict, Union
import joblib
import pandas as pd


def predict(
    pclass: int = 3,
    sex: Union[str, int] = "female",
    age: float = 22.0,
    sibsp: int = 1,
    parch: int = 0,
    fare: float = 7.25,
) -> Dict[str, Any]:
    """
    Predicts survival outcome for a single passenger.

    Args:
        pclass (int): Ticket class: 1 (1st = Upper), 2 (2nd = Middle), 3 (3rd = Lower).
        sex (Union[str, int]): Gender ('female', 'male', 1, or 0).
        age (float): Passenger age in years.
        sibsp (int): Number of siblings/spouses aboard the Titanic.
        parch (int): Number of parents/children aboard the Titanic.
        fare (float): Passenger ticket fare in British Pounds (£).

    Returns:
        Dict[str, Any]: Contains survival boolean, human label, and probability breakdown.
    """
    # 1. Resolve absolute file paths to serialized assets
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "titanic_model.joblib")
    scaler_path = os.path.join(base_dir, "data", "processed", "scaler.joblib")

    # 2. Deserialize model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Clean and encode gender (Female = 1, Male = 0)
    sex_num = 1 if str(sex).strip().lower() in ["female", "1", "f"] else 0

    # 4. Construct DataFrame matching scaler feature schema
    feature_names = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    raw_df = pd.DataFrame([[pclass, sex_num, age, sibsp, parch, fare]], columns=feature_names)

    # 5. Standardize using training parameters
    features_scaled = scaler.transform(raw_df)
    df_scaled = pd.DataFrame(features_scaled, columns=feature_names)

    # 6. Predict binary outcome and voting probability
    pred = int(model.predict(df_scaled)[0])
    probabilities = model.predict_proba(df_scaled)[0]

    return {
        "survived": bool(pred == 1),
        "label": "Survived" if pred == 1 else "Not Survived",
        "probability_survived": round(float(probabilities[1]), 4),
        "probability_not_survived": round(float(probabilities[0]), 4),
    }


if __name__ == "__main__":
    # Test passenger: 3rd Class 22-year-old female traveling with 1 sibling, fare £7.25
    sample = predict(pclass=3, sex="female", age=22.0, sibsp=1, parch=0, fare=7.25)
    print("=== PROJECT 2: TITANIC SURVIVAL PREDICTION ===")
    print("Passenger Profile: 3rd Class, Female, 22 Years Old, Fare: £7.25")
    print(f"Prediction: {sample['label']} (Survived: {sample['survived']})")
    print(f"Survival Probability:     {sample['probability_survived'] * 100:.2f}%")
    print(f"Perishing Probability:    {sample['probability_not_survived'] * 100:.2f}%")

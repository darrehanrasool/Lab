"""
========================================================================================
PROJECT 3: BOSTON HOUSING PRICE PREDICTION
MODULE 4: REAL-TIME VALUATION INFERENCE ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Harrison & Rubinfeld (1978) Boston Housing Cohort (N = 506 Census Tracts)

PURPOSE:
--------
Provides offline command-line prediction for estimating home valuation across any Boston suburb:
1. Loads the serialized Multiple Linear Regression model and StandardScaler.
2. Accepts 13 neighborhood socio-economic and environmental predictors.
3. Standardizes inputs using frozen training parameters (z = (x - mu) / sigma).
4. Computes predicted median home value (MEDV) in both thousands and formatted US Dollars ($).

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How does the model compute predicted price?
    -> Using the linear combination: y_hat = beta_0 + sum_{j=1}^{13} (beta_j * z_j).
       The raw prediction is in $1,000s (e.g. 29.84 means $29,840).

Q2: Which features increase price the most, and which decrease it?
    -> Average rooms (RM) has a strong positive coefficient (more rooms = higher price).
       Lower status percentage (LSTAT) and pupil-teacher ratio (PTRATIO) have strong negative
       coefficients (higher poverty or crowded schools = lower property values).
========================================================================================
"""

import os
from typing import Any, Dict
import joblib
import pandas as pd


def predict(
    crim: float = 0.1,
    zn: float = 18.0,
    indus: float = 2.31,
    chas: int = 0,
    nox: float = 0.538,
    rm: float = 6.575,
    age: float = 65.2,
    dis: float = 4.09,
    rad: int = 1,
    tax: float = 296.0,
    ptratio: float = 15.3,
    b: float = 396.9,
    lstat: float = 4.98,
) -> Dict[str, Any]:
    """
    Predicts the median home price in a Boston suburb given 13 community attributes.

    Args:
        crim (float): Per capita crime rate by town.
        zn (float): Proportion of residential land zoned for lots over 25,000 sq.ft.
        indus (float): Proportion of non-retail business acres per town.
        chas (int): Charles River dummy variable (1 if tract bounds river; 0 otherwise).
        nox (float): Nitric oxides concentration (parts per 10 million).
        rm (float): Average number of rooms per dwelling.
        age (float): Proportion of owner-occupied units built prior to 1940.
        dis (float): Weighted distances to five Boston employment centres.
        rad (int): Index of accessibility to radial highways.
        tax (float): Full-value property-tax rate per $10,000.
        ptratio (float): Pupil-teacher ratio by town.
        b (float): 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents.
        lstat (float): Percentage of lower status of the population.

    Returns:
        Dict[str, Any]: Contains predicted price in $1,000s and formatted dollar string.
    """
    # 1. Resolve project directory and model file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "boston_model.joblib")
    scaler_path = os.path.join(base_dir, "data", "processed", "scaler.joblib")

    # 2. Deserialize model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Format input into DataFrame with matching feature names
    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]
    raw_df = pd.DataFrame(
        [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]],
        columns=feature_names,
    )

    # 4. Standardize using frozen training statistics
    features_scaled = scaler.transform(raw_df)
    df_scaled = pd.DataFrame(features_scaled, columns=feature_names)

    # 5. Compute continuous regression prediction
    pred_val = float(model.predict(df_scaled)[0])

    return {
        "predicted_price_k": round(pred_val, 2),
        "formatted_price": f"${pred_val * 1000:,.2f}",
    }


if __name__ == "__main__":
    # Test sample with suburban characteristics (6.5 rooms, low crime, low poverty)
    result = predict()
    print("=== PROJECT 3: BOSTON HOUSING VALUATION REPORT ===")
    print(f"Predicted Median Value: {result['predicted_price_k']} ($1,000s)")
    print(f"Estimated Market Price: {result['formatted_price']}")

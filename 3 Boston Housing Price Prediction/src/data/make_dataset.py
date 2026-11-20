"""
========================================================================================
PROJECT 3: BOSTON HOUSING PRICE PREDICTION
MODULE 1: DATA INGESTION & DEFENSIVE VALIDATION (src/data/make_dataset.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Harrison & Rubinfeld (1978) Boston Housing Cohort (N = 506 Census Tracts)

PURPOSE:
--------
Step 1 ingests the authentic Boston Housing dataset (506 tracts, 13 demographic/environmental
features, and 1 continuous target MEDV in $1,000s), validates schema integrity and physiological
bounds, and persists the raw matrix to `data/raw/boston.csv`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What are the 13 predictive features in Boston Housing?
    -> 1. CRIM: Per capita crime rate by town
       2. ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
       3. INDUS: Proportion of non-retail business acres per town
       4. CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
       5. NOX: Nitric oxides concentration (parts per 10 million)
       6. RM: Average number of rooms per dwelling (strong positive correlation)
       7. AGE: Proportion of owner-occupied units built prior to 1940
       8. DIS: Weighted distances to five Boston employment centres
       9. RAD: Index of accessibility to radial highways
       10. TAX: Full-value property-tax rate per $10,000
       11. PTRATIO: Pupil-teacher ratio by town (school quality proxy)
       12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
       13. LSTAT: Percentage of lower socioeconomic status of the population (strong negative correlation)

Q2: What is the target variable MEDV?
    -> Median value of owner-occupied homes in $1,000s (e.g. MEDV = 24.0 means $24,000).
       Capped at $50,000 (right-censored at 50.0).

Q3: Why defensive assertions?
    -> Ensures matrix dimensions match exactly (506, 14), has zero null values, and all prices are positive.
========================================================================================
"""

import os
import numpy as np
import pandas as pd


def main() -> None:
    """Ingests, validates, and persists the raw Boston Housing dataset."""
    print("==================================================================")
    print("PROJECT 3 - STEP 1: INGESTION OF RAW BOSTON HOUSING DATA")
    print("==================================================================")

    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "boston.csv")

    # If raw file already exists locally, load from disk; otherwise fetch from CMU repository
    if os.path.exists(raw_path):
        print(f"Loading from existing local file: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print("Fetching Boston Housing dataset from StatLib repository...")
        url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
        # CMU StatLib alternates two rows per record: 11 values on row 1, 3 on row 2
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        columns = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
        ]
        df = pd.DataFrame(data, columns=columns)
        df["MEDV"] = target

    # Defensive schema and range validation
    assert df.shape == (506, 14), f"Unexpected shape {df.shape}; expected (506, 14)"
    assert not df.isnull().any().any(), "Data Quality Error: Missing values detected in Boston dataset!"
    assert (df["MEDV"] > 0.0).all(), "Physiological Error: Non-positive housing prices detected!"
    assert (df["RM"] >= 2.0).all() and (df["RM"] <= 12.0).all(), "Physiological Error: Unrealistic room counts!"

    # Save clean raw CSV
    df.to_csv(raw_path, index=False)

    print(f"✓ Ingestion Successful: {df.shape[0]} census tracts across {df.shape[1] - 1} predictors.")
    print(f"✓ Target MEDV Summary: Min=${df['MEDV'].min() * 1000:,.0f}, Max=${df['MEDV'].max() * 1000:,.0f}, Mean=${df['MEDV'].mean() * 1000:,.0f}")
    print(f"✓ Saved raw dataset to: {raw_path}")


if __name__ == "__main__":
    main()

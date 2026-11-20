"""
========================================================================================
PROJECT 2: TITANIC SURVIVAL PREDICTION
MODULE 1: DATA INGESTION & DEFENSIVE VALIDATION (src/data/make_dataset.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: RMS Titanic Passenger Manifest (N = 891 Samples, Binary Classification)

PURPOSE:
--------
Step 1 ingests the authentic Titanic passenger manifest, validates schema integrity
and historical survival boundaries, and saves the raw dataset to `data/raw/titanic.csv`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is the target variable and what does it represent?
    -> 'survived': Binary indicator where 0 = Perished and 1 = Survived.
Q2: What are the key demographic and socio-economic features?
    -> pclass (1st, 2nd, 3rd class ticket), sex (male/female), age (years),
       sibsp (siblings / spouses aboard), parch (parents / children aboard), fare (ticket price in £).
Q3: Why is defensive data engineering essential here?
    -> Real-world passenger manifests contain anomalies. We verify the presence of all 891 records,
       confirm that survival is strictly {0, 1}, and check that fares are non-negative.
========================================================================================
"""

import os
import pandas as pd
import seaborn as sns


def main() -> None:
    """Ingests, validates, and persists the raw Titanic passenger dataset."""
    print("==================================================================")
    print("PROJECT 2 - STEP 1: INGESTION OF RAW TITANIC PASSENGER DATA")
    print("==================================================================")

    # 1. Ingest official Seaborn Titanic dataset
    df: pd.DataFrame = sns.load_dataset("titanic")

    # 2. Defensive schema and domain boundary validation
    # Verify passenger count matches canonical 891 manifest entries
    assert df.shape[0] == 891, f"Unexpected sample count {df.shape[0]}; expected 891 passengers"

    # Verify presence of essential predictive columns
    required_cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]
    for col in required_cols:
        assert col in df.columns, f"Schema Error: Missing required column '{col}'"

    # Verify target variable is strictly binary {0, 1}
    assert set(df["survived"].unique()).issubset({0, 1}), "Target Error: 'survived' must be binary (0 or 1)"

    # Verify passenger fares are non-negative
    assert (df["fare"].dropna() >= 0.0).all(), "Physiological Error: Negative ticket fare detected!"

    # 3. Create persistent storage directory
    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/titanic.csv"
    df.to_csv(raw_path, index=False)

    survival_rate = df["survived"].mean() * 100.0
    print(f"✓ Ingestion Successful: {df.shape[0]} passengers, {df.shape[1]} raw attributes.")
    print(f"✓ Historical Baseline: {survival_rate:.1f}% overall survival rate (61.6% perished, 38.4% survived).")
    print(f"✓ Saved raw dataset to: {raw_path}")


if __name__ == "__main__":
    main()

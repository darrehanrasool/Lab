"""
========================================================================================
PROJECT 1: IRIS SPECIES CLASSIFICATION
MODULE 1: DATA INGESTION & DEFENSIVE VALIDATION (src/data/make_dataset.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Fisher's Famous 1936 Iris Flower Dataset (N = 150 Samples, 3 Classes)

PURPOSE:
--------
Step 1 ingests the authentic Iris botanical measurements, validates schema integrity
and physiological dimensions, and saves the immutable raw dataset to `data/raw/Iris.csv`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What are the 4 features in the Iris dataset?
    -> 1. Sepal Length (cm)
       2. Sepal Width (cm)
       3. Petal Length (cm)
       4. Petal Width (cm)
Q2: What are the target classes?
    -> 3 balanced species of Iris flowers (50 samples each):
       0: Iris Setosa (linearly separable from the other two)
       1: Iris Versicolor (non-linearly overlapping with Virginica)
       2: Iris Virginica
Q3: Why are defensive assertions needed?
    -> In data engineering, assertions guarantee that upstream data has exactly 150 rows,
       5 columns (4 features + 1 target), zero missing values, and plausible positive lengths.
========================================================================================
"""

import os
import pandas as pd
from sklearn.datasets import load_iris


def main() -> None:
    """Loads, validates, and persists the raw Iris flower dataset."""
    print("==================================================================")
    print("PROJECT 1 - STEP 1: INGESTION OF RAW IRIS BOTANICAL DATA")
    print("==================================================================")

    # 1. Ingest official Scikit-Learn Iris dataset as a structured DataFrame
    iris = load_iris(as_frame=True)
    df: pd.DataFrame = iris.frame.copy()

    # 2. Defensive schema and physiological boundary validation
    # Verify exact shape: 150 flower samples across 5 columns (4 features + target)
    assert df.shape == (150, 5), f"Unexpected shape {df.shape}; expected (150, 5)"

    # Verify zero missing or corrupted values
    assert not df.isnull().any().any(), "Data Quality Error: Missing values detected in Iris dataset!"

    # Verify all morphological measurements are strictly positive human-measurable centimeters
    feature_cols = [c for c in df.columns if c != "target"]
    for col in feature_cols:
        assert (df[col] > 0.0).all(), f"Physiological Error: Non-positive value found in {col}!"

    # Verify balanced multiclass distribution: exactly 50 instances per species
    class_counts = df["target"].value_counts().to_dict()
    assert class_counts == {0: 50, 1: 50, 2: 50}, f"Imbalanced class distribution: {class_counts}"

    # 3. Create persistent raw storage directory
    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/Iris.csv"
    df.to_csv(raw_path, index=False)

    print(f"✓ Success: Ingested {df.shape[0]} botanical samples across {len(feature_cols)} features.")
    print("✓ Class Distribution: 50 Setosa (0), 50 Versicolor (1), 50 Virginica (2).")
    print(f"✓ Saved raw dataset to: {raw_path}")


if __name__ == "__main__":
    main()

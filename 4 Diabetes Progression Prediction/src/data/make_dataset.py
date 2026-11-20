"""
========================================================================================
MODULE 1: DATA INGESTION & DEFENSIVE VALIDATION (src/data/make_dataset.py)
========================================================================================
Project: Diabetes Progression Prediction via Multiple Linear Regression
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Efron et al. (2004) Authentic Clinical Diabetes Cohort (N = 442 Patients)

PURPOSE:
--------
In production machine learning pipelines, Step 1 is always the ingestion of raw,
untouched data from an authoritative source, accompanied by strict schema validation
and physiological boundary checks (defensive data engineering) before any transformations.

KEY CONCEPTS SIR MIGHT ASK ABOUT:
---------------------------------
1. Why 'scaled=False'?
   -> Scikit-learn's default `load_diabetes()` returns pre-standardized, unitless
      z-score floats (mean=0, std=1/sqrt(N)). Setting `scaled=False` extracts the
      original, real-world biological units (years for age, kg/m² for BMI, mmHg for BP,
      and mg/dL for blood serum lipids). This ensures full physical interpretability.
2. Why 'as_frame=True'?
   -> Returns a pandas DataFrame with column names intact rather than an untyped NumPy
      array, preventing column index mismatch bugs.
3. Why strict assertions / physiological validation?
   -> Real-world clinical ML models must fail fast if upstream data is corrupted,
      contains null values, or has physiologically impossible vital signs.
========================================================================================
"""

import os

import pandas as pd
from sklearn.datasets import load_diabetes  # Official Scikit-Learn clinical dataset


def main() -> None:
    """
    Executes the Step 1 data ingestion pipeline:
    1. Fetches authentic unscaled physical and biochemical patient data.
    2. Validates schema dimensions, absence of missing values, and human vital ranges.
    3. Persists the clean raw dataset to `data/raw/diabetes.csv`.
    """
    print("==================================================================")
    print("STEP 1: INGESTION & DEFENSIVE VALIDATION OF RAW CLINICAL DATA")
    print("==================================================================")

    # -------------------------------------------------------------------------
    # 1. LOAD RAW DATASET WITH AUTHENTIC PHYSICAL & BIOCHEMICAL UNITS
    # -------------------------------------------------------------------------
    # Feature dictionary and clinical units:
    #   - age: Patient chronological age in years (Range: 19 to 79)
    #   - sex: Biological sex: 1.0 = Female, 2.0 = Male
    #   - bmi: Body Mass Index in kg/m² (Range: 18.5 to 42.2)
    #   - bp:  Mean arterial blood pressure in mmHg (Range: 62.0 to 133.0)
    #   - s1:  Total serum cholesterol in mg/dL (Range: 97.0 to 301.0)
    #   - s2:  Low-density lipoprotein ('bad' LDL cholesterol) in mg/dL (Range: 59.6 to 242.4)
    #   - s3:  High-density lipoprotein ('good' HDL cholesterol) in mg/dL (Range: 22.0 to 99.0)
    #   - s4:  Total cholesterol / HDL ratio (TCH ratio; Range: 2.0 to 9.09)
    #   - s5:  Logarithm of serum triglycerides level (LTG; Range: 3.25 to 6.11)
    #   - s6:  Fasting blood glucose concentration in mg/dL (Range: 58.0 to 124.0)
    #   - target: Quantitative diabetes disease progression 1 year post-baseline (25 to 346)
    data = load_diabetes(scaled=False, as_frame=True)
    df: pd.DataFrame = data.frame.copy()

    # -------------------------------------------------------------------------
    # 2. SEMANTIC COLUMN RENAMING
    # -------------------------------------------------------------------------
    # Rename default generic column 'target' to 'progression' for clear clinical interpretation
    df = df.rename(columns={"target": "progression"})

    # -------------------------------------------------------------------------
    # 3. DEFENSIVE DATA VALIDATION & PHYSIOLOGICAL BOUNDARY CHECKS
    # -------------------------------------------------------------------------
    # Check 3.1: Verify dataset dimensionality (442 patient rows, 11 columns)
    assert df.shape == (442, 11), f"Unexpected shape {df.shape}; expected (442, 11)"

    # Check 3.2: Verify zero missing / null values across entire matrix
    assert not df.isnull().any().any(), (
        "Data Quality Error: Dataset contains missing values!"
    )

    # Check 3.3: Verify target progression score lies within historical clinical boundaries [25, 346]
    assert (df["progression"] >= 25).all() and (df["progression"] <= 346).all(), (
        "Range Error: Progression target values fall outside valid bounds [25, 346]."
    )

    # Check 3.4: Verify Body Mass Index falls within plausible living human boundaries [15, 50] kg/m²
    assert (df["bmi"] >= 15.0).all() and (df["bmi"] <= 50.0).all(), (
        "Physiological Error: BMI values fall outside human metabolic boundaries [15, 50] kg/m²."
    )

    # Check 3.5: Verify Mean Arterial Blood Pressure falls within valid clinical boundaries [50, 160] mmHg
    assert (df["bp"] >= 50.0).all() and (df["bp"] <= 160.0).all(), (
        "Physiological Error: Blood pressure falls outside viable hemodynamic boundaries [50, 160] mmHg."
    )

    # -------------------------------------------------------------------------
    # 4. IMMUTABLE PERSISTENCE TO LOCAL STORAGE
    # -------------------------------------------------------------------------
    # Ensure raw data folder exists
    raw_dir: str = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_file_path: str = os.path.join(raw_dir, "diabetes.csv")

    # Save to CSV without exporting artificial pandas numeric row index
    df.to_csv(raw_file_path, index=False)

    # -------------------------------------------------------------------------
    # 5. CONSOLE AUDIT & SUMMARY REPORT
    # -------------------------------------------------------------------------
    print(
        f"✓ Ingestion Successful: {df.shape[0]} patients, {df.shape[1]} clinical variables."
    )
    print(
        f"✓ Target 'progression' statistics: "
        f"Min={df['progression'].min():.1f}, "
        f"Max={df['progression'].max():.1f}, "
        f"Mean={df['progression'].mean():.2f}, "
        f"Median={df['progression'].median():.2f}"
    )
    print(f"✓ Raw dataset saved to: {raw_file_path}")


if __name__ == "__main__":
    # Standard Python entrypoint guard to allow safe importing as a module or direct script execution
    main()

import os

from sklearn.datasets import load_diabetes  # type: ignore[import-untyped]


def main():
    print("==========================================================")
    print("STEP 1: INGESTION OF UNTOUCHED RAW CLINICAL DIABETES DATA")
    print("==========================================================")

    # Load unscaled authentic physical and biochemical measurements:
    #   age: Patient age in years (19 to 79)
    #   sex: Demographic biological sex: 1.0 (Female), 2.0 (Male)
    #   bmi: Body Mass Index in kg/m^2 (18.5 to 42.2)
    #   bp:  Mean arterial blood pressure in mmHg (62.0 to 133.0)
    #   s1:  Total serum cholesterol in mg/dL (97.0 to 301.0)
    #   s2:  Low-density lipoproteins (LDL / 'bad' cholesterol) in mg/dL (59.6 to 242.4)
    #   s3:  High-density lipoproteins (HDL / 'good' cholesterol) in mg/dL (22.0 to 99.0)
    #   s4:  Total cholesterol / HDL ratio (2.0 to 9.09)
    #   s5:  Log of serum triglycerides level (3.25 to 6.11)
    #   s6:  Fasting blood glucose concentration in mg/dL (58.0 to 124.0)
    #   progression: Quantitative diabetes disease progression 1 year post baseline (25.0 to 346.0)
    data = load_diabetes(scaled=False, as_frame=True)
    df = data.frame.copy()

    # Rename target column to 'progression' for clear clinical interpretation
    df = df.rename(columns={"target": "progression"})

    # Strict schema and physiological range validation
    assert df.shape == (442, 11), f"Unexpected shape {df.shape}; expected (442, 11)"
    assert not df.isnull().any().any(), "Dataset contains missing values!"
    assert (df["progression"] >= 25).all() and (df["progression"] <= 346).all(), (
        "Progression values out of bounds"
    )
    assert (df["bmi"] >= 15.0).all() and (df["bmi"] <= 50.0).all(), (
        "BMI outside physiological human bounds"
    )
    assert (df["bp"] >= 50.0).all() and (df["bp"] <= 160.0).all(), (
        "Blood pressure outside physiological bounds"
    )

    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/diabetes.csv"
    df.to_csv(raw_path, index=False)

    print(
        f"✓ Success: Ingested {df.shape[0]} patients and {df.shape[1]} clinical variables."
    )
    print(
        f"✓ Target 'progression' summary: Min={df['progression'].min()}, Max={df['progression'].max()}, Mean={df['progression'].mean():.2f}, Median={df['progression'].median():.2f}"
    )
    print(f"✓ Saved raw dataset to: {raw_path}")


if __name__ == "__main__":
    main()

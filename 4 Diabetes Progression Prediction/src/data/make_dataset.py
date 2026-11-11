import os
import pandas as pd
from sklearn.datasets import load_diabetes

def main():
    print("==========================================================")
    print("STEP 1: INGESTION OF UNTOUCHED RAW DIABETES CLINICAL DATA")
    print("==========================================================")
    
    # Load authentic unscaled clinical measurements
    # scaled=False preserves original physical and medical units:
    #   age: years (19 to 79)
    #   sex: 1 (female), 2 (male)
    #   bmi: kg/m^2 (18.5 to 42.2)
    #   bp: average arterial blood pressure in mmHg (62 to 133)
    #   s1 (tc): total serum cholesterol in mg/dL (97 to 301)
    #   s2 (ldl): low-density lipoproteins in mg/dL (59.6 to 242.4)
    #   s3 (hdl): high-density lipoproteins in mg/dL (22 to 99)
    #   s4 (tch): total cholesterol / HDL ratio (2 to 9.09)
    #   s5 (ltg): log of serum triglycerides (3.25 to 6.11)
    #   s6 (glu): fasting blood glucose in mg/dL (58 to 124)
    #   progression: quantitative disease progression 1 year post baseline (25 to 346)
    data = load_diabetes(scaled=False, as_frame=True)
    df = data.frame.copy()
    
    # Rename target column to 'progression' for clinical clarity
    df.rename(columns={'target': 'progression'}, inplace=True)
    
    # Assertions for immutable integrity
    assert df.shape == (442, 11), f"Unexpected shape {df.shape}; expected (442, 11)"
    assert not df.isnull().any().any(), "Raw dataset contains unexpected missing values!"
    assert (df['progression'] >= 25).all() and (df['progression'] <= 346).all(), "Target out of expected bounds"
    
    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/diabetes.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Success: Ingested authentic unscaled clinical cohort: {df.shape[0]} patients, {df.shape[1]} columns.")
    print(f"✓ Target 'progression' range: [{df['progression'].min()}, {df['progression'].max()}] | Mean: {df['progression'].mean():.2f}")
    print(f"✓ Saved immutable raw copy to: {csv_path}")

if __name__ == '__main__':
    main()

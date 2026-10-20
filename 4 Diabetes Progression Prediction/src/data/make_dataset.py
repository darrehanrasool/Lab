import os
import pandas as pd
from sklearn.datasets import load_diabetes

def main():
    print("Loading Diabetes dataset...")
    # Load dataset with original unscaled / meaningful values if desired, or load scikit-learn diabetes
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    
    # Rename target to target_progression for clarity
    df.rename(columns={'target': 'progression'}, inplace=True)
    
    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/diabetes.csv'
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path} with shape {df.shape}")

if __name__ == '__main__':
    main()

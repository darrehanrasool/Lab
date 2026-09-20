import pandas as pd
import numpy as np
import os

def main():
    print("Loading Boston Housing dataset...")
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(data, columns=columns)
    df['MEDV'] = target
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/boston.csv', index=False)
    print("Dataset saved to data/raw/boston.csv")

if __name__ == '__main__':
    main()
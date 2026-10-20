import os
import pandas as pd
from sklearn.datasets import load_iris

def main():
    print("Loading Iris dataset...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/Iris.csv', index=False)
    print("Saved raw dataset to data/raw/Iris.csv")

if __name__ == '__main__':
    main()

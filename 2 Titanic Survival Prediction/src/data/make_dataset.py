import pandas as pd
import seaborn as sns
import os

def main():
    print("Loading Titanic dataset...")
    df = sns.load_dataset('titanic')
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/titanic.csv', index=False)
    print("Dataset saved to data/raw/titanic.csv")

if __name__ == '__main__':
    main()
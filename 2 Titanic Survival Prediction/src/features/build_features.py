import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    df = pd.read_csv('data/raw/titanic.csv')
    
    # Preprocessing
    df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(scaler, 'data/processed/scaler.joblib')
    
    pd.DataFrame(X_train_scaled).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print("Features built and saved.")

if __name__ == '__main__':
    main()
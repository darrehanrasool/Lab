import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def main():
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/boston_model.joblib')
    
    with open('accuracy.txt', 'w') as f:
        f.write(f"R2 Score: {r2:.4f}\nMean Squared Error: {mse:.4f}")
    print(f"Model trained. R2 Score: {r2:.4f}, MSE: {mse:.4f}")

if __name__ == '__main__':
    main()
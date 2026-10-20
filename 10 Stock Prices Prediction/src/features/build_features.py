import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def main():
    print("Building features and technical indicators for Stock Prices Prediction...")
    df = pd.read_csv('data/raw/stock_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 1. Target variable: Next day's Close price (t+1)
    df['Target_Next_Close'] = df['Close'].shift(-1)
    
    # 2. Moving Averages & Trend Indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # 3. Volatility & Momentum Indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['Intraday_Spread'] = df['High'] - df['Low']
    df['Open_Close_Diff'] = df['Close'] - df['Open']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    
    # 4. Lagged Historical Prices
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    
    # Drop warm-up NA rows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Feature columns
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'EMA_10',
        'Daily_Return', 'Intraday_Spread', 'Open_Close_Diff',
        'Volume_Change', 'RSI_14', 'Close_Lag1', 'Close_Lag2', 'Close_Lag3'
    ]
    
    X = df[feature_cols]
    y = df['Target_Next_Close']
    dates = df['Date']
    
    # STRICT TEMPORAL CHRONOLOGICAL SPLIT (NO SHUFFLING TO PREVENT DATA LEAKAGE)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train, dates_train = X.iloc[:train_end], y.iloc[:train_end], dates.iloc[:train_end]
    X_val, y_val, dates_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end], dates.iloc[train_end:val_end]
    X_test, y_test, dates_test = X.iloc[val_end:], y.iloc[val_end:], dates.iloc[val_end:]
    
    # Standardize features (fitted strictly on training chronological window)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(scaler, 'data/processed/scaler.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save processed CSV splits
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save dates for evaluation plotting
    dates_train.to_csv('data/processed/dates_train.csv', index=False)
    dates_val.to_csv('data/processed/dates_val.csv', index=False)
    dates_test.to_csv('data/processed/dates_test.csv', index=False)
    
    # Save full feature DataFrame for visualization
    df.to_csv('data/processed/featured_stock_data.csv', index=False)
    
    print(f"Stock features built successfully:")
    print(f"  Training samples:   {len(X_train)} ({dates_train.min().strftime('%Y-%m-%d')} to {dates_train.max().strftime('%Y-%m-%d')})")
    print(f"  Validation samples: {len(X_val)} ({dates_val.min().strftime('%Y-%m-%d')} to {dates_val.max().strftime('%Y-%m-%d')})")
    print(f"  Test samples:       {len(X_test)} ({dates_test.min().strftime('%Y-%m-%d')} to {dates_test.max().strftime('%Y-%m-%d')})")

if __name__ == '__main__':
    main()

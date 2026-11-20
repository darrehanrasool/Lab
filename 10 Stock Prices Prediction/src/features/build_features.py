"""
========================================================================================
PROJECT 10: STOCK PRICES PREDICTION
MODULE 2: TECHNICAL INDICATORS & TEMPORAL PARTITIONING (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Apple Inc. (NASDAQ: AAPL) Daily OHLCV Financial Market Data

PURPOSE:
--------
1. Defines the forecasting target: Next Trading Session's Close Price (Close_{t+1}).
2. Engineers 17 market trend, momentum, volatility, and autoregressive lag indicators:
   - Simple Moving Averages (SMA-5, SMA-10, SMA-20) and Exponential Moving Average (EMA-10).
   - Momentum: Relative Strength Index (RSI-14), Daily Percentage Return, Volume Change.
   - Volatility: Intraday Spread (High - Low) and Daily Body Spread (Close - Open).
   - Autoregressive Lags: Close_{t-1}, Close_{t-2}, Close_{t-3}.
3. Enforces Strict Chronological Splitting (NO random shuffling):
   - Earliest 70% -> Training Cohort
   - Middle 15%   -> Validation Cohort
   - Latest 15%   -> Unseen Future Test Cohort
4. Standardizes features with Zero Look-Ahead Bias: StandardScaler is fitted strictly on `X_train`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why is random shuffling (`shuffle=True`) strictly forbidden in Time Series Forecasting?
    -> In cross-sectional data (like Iris or Titanic), observations are independent.
       In financial time series, prices have strong temporal auto-correlation. If we shuffled rows randomly,
       future prices (t+5, t+10) would be present in the training set to predict past prices (t+1),
       creating severe Look-Ahead Bias and false 99.9% accuracy that crashes completely in real trading.

Q2: What is the Relative Strength Index (RSI-14)?
    -> A momentum oscillator measuring the speed and magnitude of recent price changes:
           RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss over 14 days.
       Values > 70 signal an Overbought asset (due for downward correction), while values < 30
       signal an Oversold asset (due for rebound).

Q3: What does `shift(-1)` do?
    -> `df['Target_Next_Close'] = df['Close'].shift(-1)` aligns the features computed on day t
       with the actual closing price observed on day t+1, allowing the model to predict tomorrow's price today.
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes the 14-day Relative Strength Index (RSI) momentum indicator.

    Formula:
        RS = Average_Gain / Average_Loss
        RSI = 100 - (100 / (1 + RS))
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def main() -> None:
    """Orchestrates technical feature engineering, temporal chronological splitting, and scaling."""
    print("==================================================================")
    print("PROJECT 10 - STEP 2: TECHNICAL INDICATORS & TEMPORAL PARTITIONING")
    print("==================================================================")

    # 1. Load clean raw stock data from Step 1
    raw_path = "data/raw/stock_prices.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run 'python3 src/data/make_dataset.py' first!")

    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 2. Target Variable: Next day's Close price (t+1)
    df["Target_Next_Close"] = df["Close"].shift(-1)

    # 3. Moving Averages & Trend Indicators
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # 4. Volatility & Momentum Indicators
    df["Daily_Return"] = df["Close"].pct_change()
    df["Intraday_Spread"] = df["High"] - df["Low"]
    df["Open_Close_Diff"] = df["Close"] - df["Open"]
    df["Volume_Change"] = df["Volume"].pct_change()
    df["RSI_14"] = compute_rsi(df["Close"], period=14)

    # 5. Autoregressive Lagged Prices
    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag3"] = df["Close"].shift(3)

    # Drop warm-up rows (NaN from rolling windows) and the final row (NaN from shift(-1))
    df_clean = df.dropna().reset_index(drop=True)

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_5", "SMA_10", "SMA_20", "EMA_10",
        "Daily_Return", "Intraday_Spread", "Open_Close_Diff",
        "Volume_Change", "RSI_14", "Close_Lag1", "Close_Lag2", "Close_Lag3"
    ]

    X = df_clean[feature_cols]
    y = df_clean["Target_Next_Close"]
    dates = df_clean["Date"]

    # 6. Strict Chronological Temporal Splitting (70% Train, 15% Val, 15% Test)
    n = len(df_clean)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]

    print(f"✓ Chronological Temporal Partition Complete:")
    print(f"  - Training Cohort (Past 70%):   {len(X_train)} days ({dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[train_idx-1].strftime('%Y-%m-%d')})")
    print(f"  - Validation Cohort (Mid 15%):  {len(X_val)} days ({dates.iloc[train_idx].strftime('%Y-%m-%d')} to {dates.iloc[val_idx-1].strftime('%Y-%m-%d')})")
    print(f"  - Test Cohort (Future 15%):     {len(X_test)} days ({dates.iloc[val_idx].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')})")

    # 7. Zero Look-Ahead Bias Standardization (Fit exclusively on X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 8. Serialize artifacts to disk
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "data/processed/scaler.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv("data/processed/X_test.csv", index=False)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"✓ Successfully engineered {len(feature_cols)} technical indicators and saved splits.")


if __name__ == "__main__":
    main()

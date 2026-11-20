"""
========================================================================================
PROJECT 10: STOCK PRICES PREDICTION
MODULE 1: DATA INGESTION & CHRONOLOGICAL VALIDATION (src/data/make_dataset.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Apple Inc. (NASDAQ: AAPL) Daily OHLCV Financial Market Data

PURPOSE:
--------
Step 1 ingests historical daily Open, High, Low, Close, Volume (OHLCV) records for Apple Inc.,
enforces strict chronological sorting, validates financial market price invariants, and saves
the clean raw dataset to `data/raw/stock_prices.csv`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is OHLCV data?
    -> 1. Open: The price of the asset at the market opening bell (9:30 AM EST).
       2. High: The peak maximum price reached during trading hours.
       3. Low: The trough minimum price reached during trading hours.
       4. Close: The final benchmark price when the market closes (4:00 PM EST).
       5. Volume: Total number of shares transacted throughout the trading day.

Q2: What is the Fundamental Price Invariant in Financial Engineering?
    -> Low <= min(Open, Close) and High >= max(Open, Close).
       Defensive validation verifies that every trading session satisfies these physical market realities.

Q3: Why must financial data be strictly sorted chronologically?
    -> Time flows forward. If data rows are shuffled, temporal dependencies are broken,
       and technical indicators (like 20-day moving averages) become invalid.
========================================================================================
"""

import os
import urllib.request
import pandas as pd


def main() -> None:
    """Ingests, sorts chronologically, validates financial invariants, and persists raw OHLCV stock data."""
    print("==================================================================")
    print("PROJECT 10 - STEP 1: INGESTION OF RAW HISTORICAL STOCK PRICES")
    print("==================================================================")

    os.makedirs("data/raw", exist_ok=True)
    csv_path = "data/raw/stock_prices.csv"

    # If raw file already exists locally, load from disk; otherwise fetch from repository
    if os.path.exists(csv_path):
        print(f"Loading existing local raw dataset: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Downloading historical AAPL OHLCV data from Plotly Financial repository...")
        url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            df = pd.read_csv(response)

        rename_dict = {
            "Date": "Date",
            "AAPL.Open": "Open",
            "AAPL.High": "High",
            "AAPL.Low": "Low",
            "AAPL.Close": "Close",
            "AAPL.Volume": "Volume",
        }
        df = df[list(rename_dict.keys())].rename(columns=rename_dict)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Defensive schema and financial price invariant validation
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns), "Schema Error: Missing OHLCV columns"
    assert (df["High"] >= df["Low"]).all(), "Market Anomaly: High price lower than Low price!"
    assert (df["High"] >= df["Open"]).all() and (df["High"] >= df["Close"]).all(), "Market Anomaly: High violated"
    assert (df["Low"] <= df["Open"]).all() and (df["Low"] <= df["Close"]).all(), "Market Anomaly: Low violated"
    assert (df["Volume"] >= 0).all(), "Market Anomaly: Negative trading volume detected"

    df.to_csv(csv_path, index=False)

    print(f"✓ Ingestion Successful: {df.shape[0]} daily trading sessions.")
    print(f"✓ Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"✓ Close Price Range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    print(f"✓ Saved raw dataset to: {csv_path}")


if __name__ == "__main__":
    main()

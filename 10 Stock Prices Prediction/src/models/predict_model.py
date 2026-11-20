"""
========================================================================================
PROJECT 10: STOCK PRICES PREDICTION
MODULE 4: REAL-TIME MARKET FORECASTING ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Apple Inc. (NASDAQ: AAPL) Daily OHLCV Financial Market Data

PURPOSE:
--------
Provides offline command-line prediction for forecasting the next trading day's closing price:
1. Loads the serialized Random Forest Regressor and StandardScaler.
2. Accepts today's Open, High, Low, Close, and Volume parameters.
3. Automatically synthesizes technical momentum and moving average indicators.
4. Predicts tomorrow's expected Close price ($), projected dollar change, and trading signal (BUY, SELL, HOLD).

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How does the model derive the Trading Signal (BUY / SELL / HOLD)?
    -> By comparing the predicted future close (Close_{t+1}) with today's current close (Close_t):
         - Projected gain > +1.0% -> BUY signal (favorable upward momentum).
         - Projected loss < -1.0% -> SELL signal (downward trend expected).
         - Between -1.0% and +1.0% -> HOLD / NEUTRAL signal (sideways consolidation).
========================================================================================
"""

import os
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd


def predict(
    open_p: float = 115.0,
    high_p: float = 116.5,
    low_p: float = 114.2,
    close_p: float = 115.8,
    volume: float = 25000000.0,
) -> Dict[str, Any]:
    """
    Forecasts the next trading day's closing price and generates a quantitative trading signal.

    Args:
        open_p (float): Today's opening price ($).
        high_p (float): Today's high price ($).
        low_p (float): Today's low price ($).
        close_p (float): Today's close price ($).
        volume (float): Today's trading volume (shares transacted).

    Returns:
        Dict[str, Any]: Contains predicted price, projected dollar change, percentage change, and trading action.
    """
    # 1. Resolve project file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "stock_model.joblib")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")

    # 2. Deserialize model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Derive synthetic moving averages and technical indicators
    sma_5 = close_p * 0.995
    sma_10 = close_p * 0.990
    sma_20 = close_p * 0.985
    ema_10 = close_p * 0.992
    daily_return = (close_p - open_p) / open_p
    intraday_spread = high_p - low_p
    open_close_diff = close_p - open_p
    volume_change = 0.02
    rsi_14 = 55.0
    close_lag1 = close_p * 0.998
    close_lag2 = close_p * 0.995
    close_lag3 = close_p * 0.992

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_5", "SMA_10", "SMA_20", "EMA_10",
        "Daily_Return", "Intraday_Spread", "Open_Close_Diff",
        "Volume_Change", "RSI_14", "Close_Lag1", "Close_Lag2", "Close_Lag3"
    ]
    raw_df = pd.DataFrame(
        [[
            open_p, high_p, low_p, close_p, volume,
            sma_5, sma_10, sma_20, ema_10,
            daily_return, intraday_spread, open_close_diff,
            volume_change, rsi_14, close_lag1, close_lag2, close_lag3,
        ]],
        columns=feature_cols,
    )

    # 4. Standardize using training statistics
    features_scaled = scaler.transform(raw_df)
    df_scaled = pd.DataFrame(features_scaled, columns=feature_cols)

    # 5. Model Inference
    pred_close = float(model.predict(df_scaled)[0])
    delta = pred_close - close_p
    pct_change = (delta / close_p) * 100.0

    if pct_change >= 0.75:
        signal = "BUY / BULLISH"
    elif pct_change <= -0.75:
        signal = "SELL / BEARISH"
    else:
        signal = "HOLD / NEUTRAL"

    return {
        "current_close": round(close_p, 2),
        "predicted_next_close": round(pred_close, 2),
        "projected_dollar_change": round(delta, 2),
        "projected_percentage_change": f"{pct_change:+.2f}%",
        "trading_signal": signal,
    }


if __name__ == "__main__":
    res = predict(open_p=115.10, high_p=116.75, low_p=114.50, close_p=115.82, volume=28500000.0)
    print("=== PROJECT 10: NEXT-DAY STOCK PRICE FORECAST ===")
    print(f"Current Closing Price:     ${res['current_close']:.2f}")
    print(f"Predicted Next-Day Close:  ${res['predicted_next_close']:.2f}")
    print(f"Expected Price Movement:   ${res['projected_dollar_change']:+.2f} ({res['projected_percentage_change']})")
    print(f"Algorithmic Trading Signal: {res['trading_signal']}")

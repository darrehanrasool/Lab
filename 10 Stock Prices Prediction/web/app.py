"""
========================================================================================
PROJECT 10: STOCK PRICES PREDICTION
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Apple Inc. (NASDAQ: AAPL) Daily OHLCV Financial Market Data

PURPOSE:
--------
Serves as the interactive web deployment layer for Project 10:
1. Web UI: Interactive algorithmic trading dashboard with price sliders and trading signals at GET/POST /.
2. REST API: Programmatic endpoint at POST /predict for algorithmic trading bots and financial portals.
3. Pre-loads Random Forest Regressor and StandardScaler into memory on startup for instant inference.
========================================================================================
"""

import os
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# Load model and scaler into RAM on startup (zero cold-start latency)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Successfully loaded Stock Prediction Random Forest model and scaler into RAM.")
except Exception as err:
    print(f"Warning: Failed to load Stock Prediction assets: {err}")
    model = None
    scaler = None

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_5", "SMA_10", "SMA_20", "EMA_10",
    "Daily_Return", "Intraday_Spread", "Open_Close_Diff",
    "Volume_Change", "RSI_14", "Close_Lag1", "Close_Lag2", "Close_Lag3"
]


def execute_forecast(open_p, high_p, low_p, close_p, volume):
    """Core forecasting logic synthesizing technical indicators."""
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

    raw_df = pd.DataFrame(
        [[
            open_p, high_p, low_p, close_p, volume,
            sma_5, sma_10, sma_20, ema_10,
            daily_return, intraday_spread, open_close_diff,
            volume_change, rsi_14, close_lag1, close_lag2, close_lag3,
        ]],
        columns=FEATURE_COLS,
    )

    features_scaled = scaler.transform(raw_df)
    df_scaled = pd.DataFrame(features_scaled, columns=FEATURE_COLS)

    pred_close = float(model.predict(df_scaled)[0])
    delta = pred_close - close_p
    pct_change = (delta / close_p) * 100.0

    if pct_change >= 0.75:
        signal = "BUY / BULLISH"
    elif pct_change <= -0.75:
        signal = "SELL / BEARISH"
    else:
        signal = "HOLD / NEUTRAL"

    return pred_close, delta, pct_change, signal


@app.route("/", methods=["GET", "POST"])
def home():
    """Renders the HTML web interface and processes user price input."""
    prediction = None
    expected_change = None
    expected_pct = None
    trading_signal = None
    current_close = 115.82

    input_data = {
        "open_p": 115.10,
        "high_p": 116.75,
        "low_p": 114.50,
        "close_p": 115.82,
        "volume": 28500000,
    }

    if request.method == "POST":
        try:
            open_p = float(request.form.get("open_p", 115.10))
            high_p = float(request.form.get("high_p", 116.75))
            low_p = float(request.form.get("low_p", 114.50))
            close_p = float(request.form.get("close_p", 115.82))
            volume = float(request.form.get("volume", 28500000))
            current_close = close_p

            input_data = {
                "open_p": open_p,
                "high_p": high_p,
                "low_p": low_p,
                "close_p": close_p,
                "volume": int(volume),
            }

            pred, delta, pct, signal = execute_forecast(open_p, high_p, low_p, close_p, volume)
            prediction = f"${pred:.2f}"
            expected_change = f"{delta:+.2f}"
            expected_pct = f"{pct:+.2f}%"
            trading_signal = signal
        except Exception as err:
            prediction = f"Error: {err!s}"

    return render_template(
        "index.html",
        prediction=prediction,
        expected_change=expected_change,
        expected_pct=expected_pct,
        trading_signal=trading_signal,
        current_close=current_close,
        inputs=input_data,
    )


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic next-day stock price forecasting.
    Accepts JSON: { "open_p", "high_p", "low_p", "close_p", "volume" }
    Returns JSON with predicted close, projected change, and trading recommendation.
    """
    try:
        data = request.get_json(force=True)
        open_p = float(data.get("open_p", 115.10))
        high_p = float(data.get("high_p", 116.75))
        low_p = float(data.get("low_p", 114.50))
        close_p = float(data.get("close_p", 115.82))
        volume = float(data.get("volume", 28500000.0))

        pred, delta, pct, signal = execute_forecast(open_p, high_p, low_p, close_p, volume)

        return jsonify({
            "status": "success",
            "current_close": round(close_p, 2),
            "predicted_next_close": round(pred, 2),
            "projected_dollar_change": round(delta, 2),
            "projected_percentage_change": f"{pct:+.2f}%",
            "trading_signal": signal,
        }), 200
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5009, debug=False)

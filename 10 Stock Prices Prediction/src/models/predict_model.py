import os
import joblib
import numpy as np
import pandas as pd

def predict(open_p=115.0, high_p=116.5, low_p=114.2, close_p=115.8, volume=25000000):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'stock_model.joblib')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Estimate derived indicators from input parameters
    sma_5 = close_p * 0.995
    sma_10 = close_p * 0.99
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
    
    features = np.array([[
        open_p, high_p, low_p, close_p, volume,
        sma_5, sma_10, sma_20, ema_10,
        daily_return, intraday_spread, open_close_diff,
        volume_change, rsi_14, close_lag1, close_lag2, close_lag3
    ]])
    
    features_scaled = scaler.transform(features)
    predicted_next_close = float(model.predict(features_scaled)[0])
    
    expected_change = predicted_next_close - close_p
    expected_return_pct = (expected_change / close_p) * 100
    
    if expected_return_pct > 0.5:
        signal = "BUY (Bullish Momentum)"
    elif expected_return_pct < -0.5:
        signal = "SELL (Bearish Correction)"
    else:
        signal = "HOLD (Consolidation)"
        
    return {
        'current_close': round(close_p, 2),
        'predicted_next_close': round(predicted_next_close, 2),
        'expected_change': round(expected_change, 2),
        'expected_return_pct': f"{expected_return_pct:+.2f}%",
        'trading_signal': signal
    }

if __name__ == '__main__':
    res = predict(open_p=115.0, high_p=117.0, low_p=114.5, close_p=116.5, volume=28000000)
    print("Sample Stock Prediction:", res)

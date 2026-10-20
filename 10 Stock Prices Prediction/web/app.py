from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    expected_change = None
    expected_pct = None
    trading_signal = None
    current_close = 115.82
    
    input_data = {
        'open_p': 115.10,
        'high_p': 116.75,
        'low_p': 114.50,
        'close_p': 115.82,
        'volume': 28500000
    }
    
    if request.method == 'POST':
        try:
            open_p = float(request.form.get('open_p', 115.10))
            high_p = float(request.form.get('high_p', 116.75))
            low_p = float(request.form.get('low_p', 114.50))
            close_p = float(request.form.get('close_p', 115.82))
            volume = float(request.form.get('volume', 28500000))
            
            input_data = {
                'open_p': open_p,
                'high_p': high_p,
                'low_p': low_p,
                'close_p': close_p,
                'volume': volume
            }
            current_close = close_p
            
            model_path = os.path.join(BASE_DIR, 'models', 'stock_model.joblib')
            scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
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
            predicted_price = float(model.predict(features_scaled)[0])
            
            diff = predicted_price - close_p
            pct = (diff / close_p) * 100
            
            prediction = f"${predicted_price:.2f}"
            expected_change = f"{diff:+.2f}"
            expected_pct = f"{pct:+.2f}%"
            
            if pct > 0.5:
                trading_signal = "BUY (Bullish Trend)"
            elif pct < -0.5:
                trading_signal = "SELL (Bearish Correction)"
            else:
                trading_signal = "HOLD (Consolidation)"
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template('index.html',
                           prediction=prediction,
                           expected_change=expected_change,
                           expected_pct=expected_pct,
                           trading_signal=trading_signal,
                           current_close=current_close,
                           input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True, port=5009)

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

def main():
    print("Generating visualizations for Stock Prices Prediction...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA) & TECHNICAL ANALYSIS
    # -------------------------------------------------------------
    df = pd.read_csv('data/processed/featured_stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Price History & Volume Multi-Panel
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    ax1.plot(df['Date'], df['Close'], color='#2b6cb0', lw=2, label='Close Price ($)')
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.set_title('Historical Stock Price and Trading Volume (AAPL)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    ax2.bar(df['Date'], df['Volume'] / 1e6, color='#718096', width=1.5, alpha=0.7)
    ax2.set_ylabel('Volume (M)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/price_history_volume.png')
    plt.close()
    
    # 2. Technical Indicators (SMA, EMA, RSI)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    ax1.plot(df['Date'], df['Close'], color='black', lw=1.5, label='Close Price')
    ax1.plot(df['Date'], df['SMA_10'], color='#3182ce', lw=1.5, linestyle='--', label='10-Day SMA')
    ax1.plot(df['Date'], df['SMA_20'], color='#dd6b20', lw=1.5, linestyle=':', label='20-Day SMA')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Technical Analysis: Moving Averages & RSI Momentum', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.plot(df['Date'], df['RSI_14'], color='#805ad5', lw=1.5, label='RSI (14)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/technical_indicators.png')
    plt.close()
    
    # 3. Correlation Heatmap
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI_14', 'Target_Next_Close']
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Stock Market Features Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()
    
    # -------------------------------------------------------------
    # PART 2: MODEL EVALUATION DIAGNOSTICS
    # -------------------------------------------------------------
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        dates_test = pd.to_datetime(pd.read_csv('data/processed/dates_test.csv').values.ravel())
        model = joblib.load('models/stock_model.joblib')
    except Exception as e:
        print(f"Error loading model or test data: {e}")
        return
        
    preds = model.predict(X_test)
    
    # 4. Actual vs Predicted Prices Over Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual Next-Day Close', color='#2b6cb0', lw=2.5)
    plt.plot(dates_test, preds, label='Random Forest Predicted Close', color='#e53e3e', lw=2, linestyle='--')
    plt.fill_between(dates_test, y_test, preds, color='gray', alpha=0.2, label='Forecast Error')
    plt.xlabel('Date (Held-Out Test Window)', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title('Actual vs Predicted Stock Price (Chronological Evaluation)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted_prices.png')
    plt.close()
    
    # 5. Residual Error Plot
    residuals = y_test - preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(preds, residuals, color='#805ad5', alpha=0.7, edgecolors='black')
    ax1.axhline(0, color='red', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Close Price ($)', fontsize=12)
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax1.set_title('Residuals vs Predicted Price', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    sns.histplot(residuals, kde=True, ax=ax2, color='#319795', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Residual Error ($)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/residual_plot.png')
    plt.close()
    
    # 6. Time-Series Learning Curve
    subset_sizes = [int(len(X_train) * f) for f in [0.2, 0.4, 0.6, 0.8, 1.0]]
    train_r2, val_r2 = [], []
    for s in subset_sizes:
        m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        m.fit(X_train.iloc[:s], y_train[:s])
        train_r2.append(m.score(X_train.iloc[:s], y_train[:s]))
        val_r2.append(m.score(X_test, y_test))
        
    plt.figure(figsize=(8, 5))
    plt.plot(subset_sizes, train_r2, 'o-', color='crimson', lw=2, label='Training R²')
    plt.plot(subset_sizes, val_r2, 's-', color='seagreen', lw=2, label='Out-of-Sample Test R²')
    plt.xlabel('Historical Training Sample Size', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Time-Series Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()
    
    # 7. TimeSeriesSplit Expanding Window Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    tscv_scores = []
    for train_idx, test_idx in tscv.split(X_train):
        m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        m.fit(X_train.iloc[train_idx], y_train[train_idx])
        tscv_scores.append(m.score(X_train.iloc[test_idx], y_train[test_idx]))
        
    plt.figure(figsize=(7, 5))
    plt.bar(range(1, 6), tscv_scores, color='#d69e2e', edgecolor='black', width=0.55)
    plt.axhline(np.mean(tscv_scores), color='red', linestyle='--', lw=2, label=f'Mean R²: {np.mean(tscv_scores):.3f}')
    plt.xlabel('Temporal Expanding Fold', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('TimeSeriesSplit Expanding Window CV', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation_tscv.png')
    plt.close()
    
    # 8. Feature Importances
    importances = model.feature_importances_
    features = X_train.columns
    sorted_idx = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], color='#2b6cb0', edgecolor='black')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx], fontsize=11)
    plt.xlabel('Relative Feature Importance (MDI)', fontsize=12)
    plt.title('Top 10 Technical Indicators Driving Stock Price Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importances.png')
    plt.close()
    
    print("All Stock Prices Prediction visualizations saved in reports/figures/.")

if __name__ == '__main__':
    main()

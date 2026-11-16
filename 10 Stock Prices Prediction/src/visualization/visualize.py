import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

def main():
    print("Generating visualizations for Stock Prices Prediction (Uniform 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA) & TECHNICAL ANALYSIS
    # -------------------------------------------------------------
    df = pd.read_csv('data/processed/featured_stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Price History & Volume Multi-Panel (10x7.5 at 120 DPI -> 1200x900)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=120, sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    ax1.plot(df['Date'], df['Close'], color='#2b6cb0', lw=2.2, label='AAPL Close Price ($)')
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.set_title('Historical Stock Price and Trading Volume (Apple Inc. - AAPL)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=11)
    
    ax2.bar(df['Date'], df['Volume'] / 1e6, color='#718096', width=1.5, alpha=0.7)
    ax2.set_ylabel('Volume (Millions)', fontsize=11)
    ax2.set_xlabel('Trading Date', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/price_history_volume.png', dpi=120)
    plt.close()
    
    # 2. Technical Indicators (SMA, EMA, RSI) (10x7.5 at 120 DPI -> 1200x900)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=120, sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    ax1.plot(df['Date'], df['Close'], color='#1a202c', lw=1.8, label='Close Price')
    ax1.plot(df['Date'], df['SMA_10'], color='#3182ce', lw=1.8, linestyle='--', label='10-Day SMA')
    ax1.plot(df['Date'], df['SMA_20'], color='#dd6b20', lw=1.8, linestyle=':', label='20-Day SMA')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Technical Analysis: Moving Averages & RSI Momentum Signals', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.plot(df['Date'], df['RSI_14'], color='#805ad5', lw=1.8, label='RSI (14-Period)')
    ax2.axhline(70, color='#e53e3e', linestyle='--', alpha=0.8, label='Overbought Threshold (70)')
    ax2.axhline(30, color='#38a169', linestyle='--', alpha=0.8, label='Oversold Threshold (30)')
    ax2.set_ylabel('RSI', fontsize=11)
    ax2.set_xlabel('Trading Date', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9, ncol=3)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/technical_indicators.png', dpi=120)
    plt.close()
    
    # 3. Correlation Heatmap (10x7.5 at 120 DPI -> 1200x900)
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI_14', 'Target_Next_Close']
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.8, cbar=True)
    plt.title('Stock Market Technical Features Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png', dpi=120)
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
    
    # 4. Actual vs Predicted Prices Over Time Series (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(dates_test, y_test, label='Actual Next-Day Close', color='#2b6cb0', lw=2.5)
    plt.plot(dates_test, preds, label='Random Forest Predicted Close', color='#e53e3e', lw=2, linestyle='--')
    plt.fill_between(dates_test, y_test, preds, color='gray', alpha=0.2, label='Forecast Residual Gap')
    plt.xlabel('Date (Held-Out Future Window)', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title('Out-of-Sample Actual vs Predicted Stock Prices (Chronological Split)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted_prices.png', dpi=120)
    plt.close()
    
    # 5. Residual Error Plot (10x7.5 at 120 DPI -> 1200x900)
    residuals = y_test - preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7.5), dpi=120)
    ax1.scatter(preds, residuals, color='#805ad5', alpha=0.7, edgecolors='black', s=50)
    ax1.axhline(0, color='#e53e3e', linestyle='--', lw=2.2)
    ax1.set_xlabel('Predicted Close Price ($)', fontsize=12)
    ax1.set_ylabel('Residual Error ($)', fontsize=12)
    ax1.set_title('Residuals vs. Predicted Price', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    sns.histplot(residuals, kde=True, ax=ax2, color='#319795', edgecolor='black')
    ax2.axvline(0, color='#e53e3e', linestyle='--', lw=2.2)
    ax2.set_xlabel('Residual Error ($)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Error Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.suptitle('Residual Diagnostics: Homoscedasticity & Normal Dispersion', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/residual_plot.png', dpi=120)
    plt.close()
    
    # 6. Time-Series Learning Curve (10x7.5 at 120 DPI -> 1200x900)
    subset_sizes = [int(len(X_train) * f) for f in [0.2, 0.4, 0.6, 0.8, 1.0]]
    train_r2, val_r2 = [], []
    for s in subset_sizes:
        m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        m.fit(X_train.iloc[:s], y_train[:s])
        train_r2.append(m.score(X_train.iloc[:s], y_train[:s]))
        val_r2.append(m.score(X_test, y_test))
        
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(subset_sizes, train_r2, 'o-', color='#e53e3e', lw=2.5, label='Training R² Score')
    plt.plot(subset_sizes, val_r2, 's-', color='#38a169', lw=2.5, label='Out-of-Sample Future Test R²')
    plt.xlabel('Historical Training Sample Size (Days)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Time-Series Learning Curve: Expanding Historical Window', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png', dpi=120)
    plt.close()
    
    # 7. TimeSeriesSplit Expanding Window Cross-Validation (10x7.5 at 120 DPI -> 1200x900)
    tscv = TimeSeriesSplit(n_splits=5)
    tscv_scores = []
    for train_idx, test_idx in tscv.split(X_train):
        m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        m.fit(X_train.iloc[train_idx], y_train[train_idx])
        tscv_scores.append(m.score(X_train.iloc[test_idx], y_train[test_idx]))
        
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), tscv_scores, color='#d69e2e', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        offset = 0.02 if yval >= 0 else -0.05
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.axhline(np.mean(tscv_scores), color='#e53e3e', linestyle='--', lw=2.5, label=f'Mean Walk-Forward R²: {np.mean(tscv_scores):.3f}')
    plt.axhline(0, color='black', lw=1)
    plt.xlabel('Temporal Walk-Forward Fold (Chronological)', fontsize=12)
    plt.ylabel('Out-of-Fold R² Score', fontsize=12)
    plt.title('TimeSeriesSplit Expanding Walk-Forward Cross-Validation', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation_tscv.png', dpi=120)
    plt.close()
    
    # 8. Feature Importances (10x7.5 at 120 DPI -> 1200x900)
    importances = model.feature_importances_
    features = X_train.columns
    sorted_idx = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.barh(range(len(sorted_idx)), importances[sorted_idx], color='#2b6cb0', edgecolor='black')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx], fontsize=11)
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval + 0.005, bar.get_y() + bar.get_height()/2.0, f'{xval:.3f}', va='center', ha='left', fontsize=10, fontweight='bold')
    plt.xlabel('Relative Feature Importance (Mean Decrease in Impurity)', fontsize=12)
    plt.title('Top 10 Technical Indicators Driving Stock Price Forecasts', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importances.png', dpi=120)
    plt.close()
    
    print("✓ All 8 Stock Prices visualizations saved in reports/figures/ at uniform (1200x900).")

if __name__ == '__main__':
    main()

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve, cross_val_score

def main():
    print("Generating Boston Housing visualizations (Standard Uniform Size: 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # 1. Exploratory Data Analysis
    df = pd.read_csv('data/raw/boston.csv')
    
    # Pairplot (10x7.5 at 120 DPI -> 1200x900)
    plot = sns.pairplot(df[['MEDV', 'RM', 'LSTAT', 'CRIM', 'AGE']], palette='husl')
    plot.fig.set_size_inches(10, 7.5)
    plt.suptitle('Pairplot of Key Boston Housing Features', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('reports/figures/pairplot.png', dpi=120)
    plt.close()
    
    # Heatmap (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.8, cbar=True)
    plt.title('Boston Housing Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/heatmap.png', dpi=120)
    plt.close()
    
    # Feature Distributions (10x7.5 at 120 DPI -> 1200x900)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.5), dpi=120)
    features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE']
    for i, feature in enumerate(features):
        row, col = i // 3, i % 3
        sns.histplot(data=df, x=feature, kde=True, ax=axes[row, col], color='#0d9488')
        axes[row, col].set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
    plt.suptitle('Urban & Environmental Feature Distributions', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png', dpi=120)
    plt.close()

    # 2. Model Evaluation
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/boston_model.joblib')
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return

    preds = model.predict(X_test)

    # Actual vs Predicted (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.scatter(y_test, preds, alpha=0.7, color='#2563eb', edgecolors='k', s=60, label='Test Tracts')
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], '#dc2626', lw=2.5, linestyle='--', label='Identity Line (y = y_hat)')
    plt.xlabel('Actual Median Home Value ($1000s)', fontsize=12)
    plt.ylabel('Predicted Median Home Value ($1000s)', fontsize=12)
    plt.title('Multiple Linear Regression (OLS): Actual vs. Predicted Prices', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png', dpi=120)
    plt.close()

    # Residual Plot (10x7.5 at 120 DPI -> 1200x900)
    residuals = y_test - preds
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.scatter(preds, residuals, alpha=0.7, color='#7c3aed', edgecolors='k', s=60)
    plt.axhline(y=0, color='#dc2626', linestyle='--', lw=2.5, label='Zero Residual Baseline')
    plt.xlabel('Predicted Home Values ($1000s)', fontsize=12)
    plt.ylabel('Residual Errors (Actual - Predicted, $1000s)', fontsize=12)
    plt.title('Residual Diagnostics: Homoscedasticity & Linearity Check', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/residual_plot.png', dpi=120)
    plt.close()

    # Learning Curve (10x7.5 at 120 DPI -> 1200x900)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='r2')
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(train_sizes, train_mean, 'o-', color="#dc2626", lw=2.5, label="Training R² Score")
    plt.plot(train_sizes, test_mean, 'o-', color="#16a34a", lw=2.5, label="Cross-Validation R² Score")
    plt.xlabel("Training Set Sample Size", fontsize=12)
    plt.ylabel("Coefficient of Determination (R²)", fontsize=12)
    plt.title("Multiple Linear Regression Learning Curve: Bias-Variance Diagnostics", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png', dpi=120)
    plt.close()

    # Cross Validation (10x7.5 at 120 DPI -> 1200x900)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), cv_scores, color='#0284c7', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.axhline(np.mean(cv_scores), color='#dc2626', linestyle='--', lw=2.5, label=f'Mean CV R²: {np.mean(cv_scores):.4f}')
    plt.title('5-Fold Cross-Validation R² Benchmark on Training Cohort', fontsize=14, fontweight='bold')
    plt.xlabel('Cross-Validation Fold', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.ylim(0.4, 0.9)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png', dpi=120)
    plt.close()

    # Feature Coefficients / Importances (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    if hasattr(model, 'coef_'):
        feature_names = [c for c in X_train.columns if c != 'MEDV']
        coef_series = pd.Series(model.coef_, index=feature_names).sort_values()
        colors = ['#dc2626' if c < 0 else '#2563eb' for c in coef_series.values]
        bars = plt.barh(coef_series.index, coef_series.values, color=colors, edgecolor='black')
        for bar in bars:
            xval = bar.get_width()
            offset = 0.1 if xval >= 0 else -0.3
            ha = 'left' if xval >= 0 else 'right'
            plt.text(xval + offset, bar.get_y() + bar.get_height()/2.0, f'{xval:+.2f}', va='center', ha=ha, fontsize=9, fontweight='bold')
        plt.title('Multiple Linear Regression Coefficients (Beta Weights)', fontsize=14, fontweight='bold')
        plt.xlabel('Coefficient Value (Impact on MEDV in $1000s)', fontsize=12)
    else:
        feature_names = [c for c in X_train.columns if c != 'MEDV']
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        importances.plot(kind='barh', color='#2563eb', edgecolor='black')
        plt.title('Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
    plt.axvline(x=0, color='black', lw=1)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importances.png', dpi=120)
    plt.close()

    print("✓ All 8 Boston Housing visualizations generated at uniform (1200x900) resolution.")

if __name__ == '__main__':
    main()

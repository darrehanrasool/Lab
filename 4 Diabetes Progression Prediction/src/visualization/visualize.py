import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve, cross_val_score

def main():
    print("Generating all visualizations for Diabetes Progression Prediction...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    df = pd.read_csv('data/raw/diabetes.csv')
    
    # 1. Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features_to_plot = ['age', 'bmi', 'bp', 's1', 's5', 'progression']
    for i, col in enumerate(features_to_plot):
        row, col_idx = i // 3, i % 3
        sns.histplot(data=df, x=col, kde=True, ax=axes[row, col_idx], color='teal')
        axes[row, col_idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png')
    plt.close()
    
    # 2. Pairplot
    plot = sns.pairplot(df[['bmi', 'bp', 's5', 's6', 'progression']], diag_kind='kde', palette='viridis')
    plot.fig.set_size_inches(10, 8)
    plt.suptitle('Pairwise Feature Relationships in Diabetes Dataset', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('reports/figures/pairplot.png')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Diabetes Features Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/heatmap.png')
    plt.close()
    
    # -------------------------------------------------------------
    # PART 2: MODEL EVALUATION DIAGNOSTICS
    # -------------------------------------------------------------
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/diabetes_model.joblib')
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return
        
    preds = model.predict(X_test)
    
    # 4. Actual vs Predicted Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, preds, alpha=0.7, color='darkblue', edgecolors='k', s=60, label='Test Samples')
    min_val, max_val = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction (y = x)')
    plt.xlabel('Actual Disease Progression', fontsize=12)
    plt.ylabel('Predicted Disease Progression', fontsize=12)
    plt.title('Actual vs Predicted Diabetes Progression', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()
    
    # 5. Residual Plot
    residuals = y_test - preds
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=preds, y=residuals, alpha=0.7, color='purple', s=60)
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Disease Progression', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot (Homoscedasticity Check)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/residual_plot.png')
    plt.close()
    
    # 6. Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2'
    )
    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='crimson', lw=2, label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='seagreen', lw=2, label='Cross-Validation Score')
    plt.xlabel('Training Sample Count', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Learning Curve (Bias vs Variance Analysis)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()
    
    # 7. Cross-Validation Scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(1, 6), cv_scores, color='cornflowerblue', edgecolor='black', width=0.55)
    plt.axhline(np.mean(cv_scores), color='red', linestyle='--', lw=2, label=f'Mean R²: {np.mean(cv_scores):.2f}')
    plt.xlabel('K-Fold Cross-Validation Fold', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('5-Fold Cross-Validation Performance Stability', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png')
    plt.close()
    
    # 8. Feature Importances
    importances = model.feature_importances_
    features = X_train.columns
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], color='teal', edgecolor='black', align='center')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.xlabel('Relative Feature Importance (Mean Decrease in Impurity)', fontsize=12)
    plt.title('Random Forest Feature Importance for Diabetes Progression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importances.png')
    plt.close()
    
    print("All visualizations saved to reports/figures/.")

if __name__ == '__main__':
    main()

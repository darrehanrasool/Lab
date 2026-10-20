import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, cross_val_score
import joblib
import os

def main():
    print("Generating all visualizations...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # ==========================================
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    print("1. Generating EDA plots...")
    df = pd.read_csv('data/raw/boston.csv')
    
    # Pairplot (Set to big size)
    plot = sns.pairplot(df[['MEDV', 'RM', 'LSTAT', 'CRIM', 'AGE']], palette='husl')
    plot.fig.set_size_inches(10, 8)
    plt.suptitle('Pairplot of Key Boston Housing Features', y=1.02)
    plt.savefig('reports/figures/pairplot.png')
    plt.close()
    
    # Heatmap (Set to big size)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/heatmap.png')
    plt.close()
    
    # Feature Distributions (Keep big, this one is already good)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE']
    for i, feature in enumerate(features):
        row, col = i // 3, i % 3
        sns.histplot(data=df, x=feature, kde=True, ax=axes[row, col], color='teal')
        axes[row, col].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png')
    plt.close()

    # ==========================================
    # PART 2: MODEL EVALUATION METRICS
    # ==========================================
    print("2. Generating Model Evaluation plots...")
    
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/boston_model.joblib')
    except FileNotFoundError:
        print("Error: Model or processed data not found. Run training scripts first!")
        return

    preds = model.predict(X_test)

    # Actual vs Predicted Plot (Set to big size)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, preds, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()

    # Residual Plot (Set to big size)
    residuals = y_test - preds
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=preds, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('reports/figures/residual_plot.png')
    plt.close()

    # Learning Curve (Set to big size)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2')
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("R2 Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()

    # Cross Validation (Set to big size)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    plt.figure(figsize=(10, 8))
    sns.boxplot(y=cv_scores, color='lightgreen')
    plt.title('Cross-Validation R2 Scores')
    plt.ylabel('R2 Score')
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png')
    plt.close()

    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        feature_names = [c for c in X_train.columns if c != 'MEDV']
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        plt.figure(figsize=(10, 8))
        importances.plot(kind='barh', color='steelblue')
        plt.title('Random Forest Feature Importances (MDI)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importances.png')
        plt.close()

    print("SUCCESS! All graphs generated in 'reports/figures/' folder.")

if __name__ == '__main__':
    main()
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve, cross_val_score

def main():
    print("Generating Titanic visualizations (Standard Uniform Size: 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2

    # 1. Load raw data for EDA
    df = pd.read_csv('data/raw/titanic.csv')
    df_viz = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].copy()
    df_viz['age'] = df_viz['age'].fillna(df_viz['age'].median())
    df_viz['sex'] = df_viz['sex'].map({'male': 0, 'female': 1})
    
    # Feature Distributions (10x7.5 at 120 DPI -> 1200x900)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.5), dpi=120)
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    for i, feature in enumerate(features):
        row, col = i // 3, i % 3
        sns.histplot(data=df_viz, x=feature, hue='survived', kde=True, ax=axes[row, col], palette='husl')
        axes[row, col].set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
    plt.suptitle('Titanic Feature Distributions by Survival', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png', dpi=120)
    plt.close()
    
    # Pairplot (10x7.5 at 120 DPI -> 1200x900)
    plot = sns.pairplot(df_viz, hue='survived', palette='husl')
    plot.fig.set_size_inches(10, 7.5)
    plt.suptitle('Pairplot of Key Titanic Demographic Features', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('reports/figures/pairplot.png', dpi=120)
    plt.close()
    
    # Heatmap (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(df_viz.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.8, cbar=True)
    plt.title('Titanic Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/heatmap.png', dpi=120)
    plt.close()

    # 2. Model Evaluation Diagnostics
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/titanic_model.joblib')
    except Exception as e:
        print(f"Error loading model or splits: {e}")
        return

    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix (10x7.5 at 120 DPI -> 1200x900)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'], cbar=True)
    plt.title('Random Forest Confusion Matrix (Holdout Test Set)', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=120)
    plt.close()

    # ROC Curve (10x7.5 at 120 DPI -> 1200x900)
    fpr, tpr, _ = roc_curve(y_test, preds_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(fpr, tpr, color='#dd6b20', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#2b6cb0', lw=2, linestyle='--', label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png', dpi=120)
    plt.close()

    # Precision-Recall Curve (10x7.5 at 120 DPI -> 1200x900)
    precision, recall, _ = precision_recall_curve(y_test, preds_proba)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(recall, precision, color='#3182ce', lw=2.5, label='Precision-Recall Curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Maritime Survival)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png', dpi=120)
    plt.close()

    # Learning Curve (10x7.5 at 120 DPI -> 1200x900)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(train_sizes, train_mean, 'o-', color="#e53e3e", lw=2.5, label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="#38a169", lw=2.5, label="Cross-Validation Score")
    plt.xlabel("Training Cohort Size", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.title("Random Forest Learning Curve: Bias-Variance Tradeoff", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png', dpi=120)
    plt.close()

    # Cross Validation (10x7.5 at 120 DPI -> 1200x900)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), cv_scores, color='#319795', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.axhline(np.mean(cv_scores), color='#e53e3e', linestyle='--', lw=2.5, label=f'Mean CV: {np.mean(cv_scores):.4f}')
    plt.title('5-Fold Cross-Validation Accuracy Benchmark', fontsize=14, fontweight='bold')
    plt.xlabel('Cross-Validation Fold', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.ylim(0.7, 0.95)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png', dpi=120)
    plt.close()

    print("✓ All 8 Titanic visualizations generated at uniform (1200x900) resolution.")

if __name__ == '__main__':
    main()

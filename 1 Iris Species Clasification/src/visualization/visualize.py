import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve, cross_val_score

def main():
    print("Generating Iris visualizations (Standard Uniform Size: 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global visual styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Load raw data
    df = pd.read_csv('data/raw/Iris.csv')
    feature_cols = [c for c in df.columns if c != 'target']
    
    # 1. Feature distributions (10x7.5 at 120 DPI -> 1200x900)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=120)
    for i, col in enumerate(feature_cols):
        r, c = i // 2, i % 2
        sns.histplot(data=df, x=col, hue='target', kde=True, ax=axes[r, c], palette='Set2')
        axes[r, c].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    plt.suptitle('Iris Feature Distributions by Species', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/01_feature_distributions.png', dpi=120)
    plt.close()
    
    # 2. Pairplot (10x7.5 at 120 DPI -> 1200x900)
    g = sns.pairplot(df, hue='target', palette='Set2')
    g.fig.set_size_inches(10, 7.5)
    plt.suptitle('Pairplot of Iris Morphological Features', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('reports/figures/02_pairplot.png', dpi=120)
    plt.close()
    
    # 3. Correlation Heatmap (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.8, cbar=True)
    plt.title('Iris Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/03_correlation_heatmap.png', dpi=120)
    plt.close()
    
    # Load processed data and model
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/logistic_regression_model.pkl')
    except Exception as e:
        print(f"Skipping model evaluation plots: {e}")
        return
        
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    # 4. Confusion Matrix (10x7.5 at 120 DPI -> 1200x900)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title('Multiclass Confusion Matrix (Softmax Logistic Regression)', fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth Species', fontsize=12)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/04_confusion_matrix.png', dpi=120)
    plt.close()
    
    # 5. ROC Curves (One-vs-Rest) (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    for i, name in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Multiclass ROC Curves (One-vs-Rest Analysis)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/05_roc_curves.png', dpi=120)
    plt.close()
    
    # 6. Precision-Recall Curves (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    for i, name in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        plt.plot(rec, prec, lw=2.5, label=f'{name}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multiclass Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/06_precision_recall_curves.png', dpi=120)
    plt.close()
    
    # 7. Learning Curve (10x7.5 at 120 DPI -> 1200x900)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy'
    )
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#e53e3e', lw=2.5, label='Training Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='#38a169', lw=2.5, label='Cross-Validation Score')
    plt.xlabel('Training Set Sample Size', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Iris Learning Curve: Bias-Variance Diagnostics', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/07_learning_curve.png', dpi=120)
    plt.close()
    
    # 8. Cross-Validation Scores (10x7.5 at 120 DPI -> 1200x900)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), cv_scores, color='#3182ce', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.axhline(np.mean(cv_scores), color='#e53e3e', linestyle='--', lw=2.5, label=f'Mean CV: {np.mean(cv_scores):.4f}')
    plt.xlabel('Stratified Fold Number', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0.7, 1.05)
    plt.title('5-Fold Stratified Cross-Validation Accuracy Scores', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/08_cross_validation.png', dpi=120)
    plt.close()
    
    print("✓ All 8 Iris visualizations successfully regenerated at uniform (1200x900) resolution.")

if __name__ == '__main__':
    main()

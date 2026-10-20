import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve, cross_val_score

def main():
    print("Generating Iris visualizations...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load raw data
    df = pd.read_csv('data/raw/Iris.csv')
    feature_cols = [c for c in df.columns if c != 'target']
    
    # 1. Feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, col in enumerate(feature_cols):
        r, c = i // 2, i % 2
        sns.histplot(data=df, x=col, hue='target', kde=True, ax=axes[r, c], palette='Set2')
        axes[r, c].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('reports/figures/01_feature_distributions.png')
    plt.close()
    
    # 2. Pairplot
    g = sns.pairplot(df, hue='target', palette='Set2')
    g.fig.set_size_inches(10, 8)
    plt.suptitle('Pairplot of Iris Features', y=1.02)
    plt.savefig('reports/figures/02_pairplot.png')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Iris Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/03_correlation_heatmap.png')
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
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Iris Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/figures/04_confusion_matrix.png')
    plt.close()
    
    # 5. ROC Curves (One-vs-Rest)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('reports/figures/05_roc_curves.png')
    plt.close()
    
    # 6. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        plt.plot(rec, prec, lw=2, label=f'{name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('reports/figures/06_precision_recall_curves.png')
    plt.close()
    
    # 7. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy'
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='g', label='Cross-Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Iris Learning Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('reports/figures/07_learning_curve.png')
    plt.close()
    
    # 8. Cross-Validation Scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    plt.figure(figsize=(7, 5))
    plt.bar(range(1, 6), cv_scores, color='skyblue', edgecolor='black')
    plt.axhline(np.mean(cv_scores), color='red', linestyle='--', label=f'Mean CV: {np.mean(cv_scores):.2f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('5-Fold Cross-Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/08_cross_validation.png')
    plt.close()
    print("Iris visualizations generated successfully.")

if __name__ == '__main__':
    main()

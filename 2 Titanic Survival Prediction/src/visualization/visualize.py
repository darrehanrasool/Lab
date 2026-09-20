import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
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
    df = pd.read_csv('data/raw/titanic.csv')
    df_viz = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].copy()
    df_viz['age'] = df_viz['age'].fillna(df_viz['age'].median())
    df_viz['sex'] = df_viz['sex'].map({'male': 0, 'female': 1})
    
    # Pairplot (Set to big size)
    plot = sns.pairplot(df_viz, hue='survived', palette='husl')
    plot.fig.set_size_inches(10, 8)
    plt.suptitle('Pairplot of Titanic Features', y=1.02)
    plt.savefig('reports/figures/pairplot.png')
    plt.close()
    
    # Heatmap (Set to big size)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_viz.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/heatmap.png')
    plt.close()
    
    # Feature Distributions (Keep big, this one is already good)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    for i, feature in enumerate(features):
        row, col = i // 3, i % 3
        sns.histplot(data=df_viz, x=feature, hue='survived', kde=True, ax=axes[row, col], palette='husl')
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
        model = joblib.load('models/titanic_model.joblib')
    except FileNotFoundError:
        print("Error: Model or processed data not found. Run training scripts first!")
        return

    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix (Set to big size)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()

    # ROC Curve (Set to big size)
    fpr, tpr, _ = roc_curve(y_test, preds_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png')
    plt.close()

    # Precision-Recall Curve (Set to big size)
    precision, recall, _ = precision_recall_curve(y_test, preds_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png')
    plt.close()

    # Learning Curve (Set to big size)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()

    # Cross Validation (Set to big size)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    plt.figure(figsize=(10, 8))
    sns.boxplot(y=cv_scores, color='lightblue')
    plt.title('Cross-Validation Accuracy Scores')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png')
    plt.close()

    print("SUCCESS! All graphs generated in 'reports/figures/' folder.")

if __name__ == '__main__':
    main()
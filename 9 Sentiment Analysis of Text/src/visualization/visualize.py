import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import learning_curve, cross_val_score

def main():
    print("Generating visualizations for Sentiment Analysis of Text...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    df = pd.read_csv('data/raw/sentiment.csv')
    df['word_count'] = df['review'].astype(str).apply(lambda s: len(s.split()))
    
    # 1. Sentiment Class Distribution
    plt.figure(figsize=(7, 5))
    counts = df['sentiment'].value_counts()
    bars = plt.bar(counts.index.str.capitalize(), counts.values, color=['#38a169', '#e53e3e'], edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval} ({yval/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')
    plt.title('IMDB Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/sentiment_distribution.png')
    plt.close()
    
    # 2. Review Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df[df['word_count'] <= 600], x='word_count', hue='sentiment', bins=40, kde=True,
                 palette={'positive': '#38a169', 'negative': '#e53e3e'}, alpha=0.6)
    plt.title('Review Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
    plt.xlabel('Word Count per Review', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/review_length_distribution.png')
    plt.close()
    
    # -------------------------------------------------------------
    # PART 2: MODEL EVALUATION DIAGNOSTICS
    # -------------------------------------------------------------
    try:
        X_train = joblib.load('data/processed/X_train_tfidf.joblib')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = joblib.load('data/processed/X_test_tfidf.joblib')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/sentiment_model.joblib')
        vectorizer = joblib.load('models/vectorizer.joblib')
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return
        
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Sentiment Analysis Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Sentiment', fontsize=12)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3182ce', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png')
    plt.close()
    
    # 5. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#805ad5', lw=2.5, label=f'PR Curve (AP = {ap:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png')
    plt.close()
    
    # 6. Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='crimson', lw=2, label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='seagreen', lw=2, label='Cross-Validation Score')
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Learning Curve (Logistic Regression)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()
    
    # 7. Cross-Validation Scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    plt.figure(figsize=(7, 5))
    plt.bar(range(1, 6), cv_scores, color='#4fd1c5', edgecolor='black', width=0.55)
    plt.axhline(np.mean(cv_scores), color='red', linestyle='--', lw=2, label=f'Mean Acc: {np.mean(cv_scores):.4f}')
    plt.xlabel('Cross-Validation Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('5-Fold Cross-Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png')
    plt.close()
    
    # 8. Top Sentiment Words (Model Coefficients)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    
    top_pos_idx = np.argsort(coefs)[-12:]
    top_neg_idx = np.argsort(coefs)[:12]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(range(12), coefs[top_neg_idx], color='#e53e3e', edgecolor='black')
    axes[0].set_yticks(range(12))
    axes[0].set_yticklabels(feature_names[top_neg_idx], fontsize=11)
    axes[0].set_title('Top Negative Sentiment Words (Negative Weights)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Coefficient Magnitude')
    
    axes[1].barh(range(12), coefs[top_pos_idx], color='#38a169', edgecolor='black')
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(feature_names[top_pos_idx], fontsize=11)
    axes[1].set_title('Top Positive Sentiment Words (Positive Weights)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Coefficient Magnitude')
    
    plt.tight_layout()
    plt.savefig('reports/figures/top_sentiment_words.png')
    plt.close()
    
    print("All Sentiment Analysis visualizations saved in reports/figures/.")

if __name__ == '__main__':
    main()

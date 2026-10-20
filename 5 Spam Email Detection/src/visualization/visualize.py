import os
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import learning_curve, cross_val_score

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Generating visualizations for Spam Email Detection...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    df = pd.read_csv('data/raw/spam.csv')
    df['char_length'] = df['message'].astype(str).apply(len)
    df['word_count'] = df['message'].astype(str).apply(lambda s: len(s.split()))
    
    # 1. Class Distribution
    plt.figure(figsize=(7, 5))
    counts = df['label'].value_counts()
    bars = plt.bar(counts.index, counts.values, color=['#2b6cb0', '#e53e3e'], edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval} ({yval/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')
    plt.title('Spam Email / SMS Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Message Category', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/class_distribution.png')
    plt.close()
    
    # 2. Message Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='char_length', hue='label', bins=50, kde=True,
                 palette={'ham': '#2b6cb0', 'spam': '#e53e3e'}, alpha=0.6)
    plt.xlim(0, 350)
    plt.title('Message Length Distribution: Ham vs. Spam', fontsize=14, fontweight='bold')
    plt.xlabel('Character Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/message_length_distribution.png')
    plt.close()
    
    # -------------------------------------------------------------
    # PART 2: MODEL EVALUATION DIAGNOSTICS
    # -------------------------------------------------------------
    try:
        X_train = joblib.load('data/processed/X_train_tfidf.joblib')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = joblib.load('data/processed/X_test_tfidf.joblib')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/spam_model.joblib')
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
                xticklabels=['Ham (Legit)', 'Spam'],
                yticklabels=['Ham (Legit)', 'Spam'])
    plt.title('Spam Detection Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
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
    plt.xlabel('Recall (Fraction of Spam Caught)', fontsize=12)
    plt.ylabel('Precision (Fraction of Flagged that are Truly Spam)', fontsize=12)
    plt.title('Precision-Recall Curve (Spam Imbalance Focus)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png')
    plt.close()
    
    # 6. Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1'
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='crimson', lw=2, label='Training F1')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='seagreen', lw=2, label='Cross-Validation F1')
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Learning Curve (F1 Metric)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()
    
    # 7. Cross-Validation Scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    plt.figure(figsize=(7, 5))
    plt.bar(range(1, 6), cv_scores, color='#319795', edgecolor='black', width=0.55)
    plt.axhline(np.mean(cv_scores), color='red', linestyle='--', lw=2, label=f'Mean F1: {np.mean(cv_scores):.4f}')
    plt.xlabel('Cross-Validation Fold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('5-Fold Cross-Validation Stability', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png')
    plt.close()
    
    # 8. Top Keywords Importance (Log-odds ratio in Naive Bayes)
    feature_names = np.array(vectorizer.get_feature_names_out())
    # log P(w|spam) - log P(w|ham)
    log_prob_spam = model.feature_log_prob_[1]
    log_prob_ham = model.feature_log_prob_[0]
    log_odds = log_prob_spam - log_prob_ham
    top_spam_indices = np.argsort(log_odds)[-15:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_spam_indices)), log_odds[top_spam_indices], color='#e53e3e', edgecolor='black')
    plt.yticks(range(len(top_spam_indices)), feature_names[top_spam_indices], fontsize=11)
    plt.xlabel('Log-Odds Ratio [log P(w|Spam) - log P(w|Ham)]', fontsize=12)
    plt.title('Top 15 Most Discriminative Spam Trigger Keywords', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/top_keywords_importance.png')
    plt.close()
    
    print("All visualizations saved successfully in reports/figures/.")

if __name__ == '__main__':
    main()

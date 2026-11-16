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
    print("Generating Spam Email visualizations (Standard Uniform Size: 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    df = pd.read_csv('data/raw/spam.csv')
    df['char_length'] = df['message'].astype(str).apply(len)
    df['word_count'] = df['message'].astype(str).apply(lambda s: len(s.split()))
    
    # 1. Class Distribution (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    counts = df['label'].value_counts()
    bars = plt.bar(counts.index, counts.values, color=['#2b6cb0', '#e53e3e'], edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 60, f'{yval:,} ({yval/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.title('Spam Email / SMS Class Distribution (Imbalance Analysis)', fontsize=14, fontweight='bold')
    plt.xlabel('Message Category', fontsize=12)
    plt.ylabel('Total Message Count', fontsize=12)
    plt.ylim(0, max(counts.values) * 1.15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/class_distribution.png', dpi=120)
    plt.close()
    
    # 2. Message Length Distribution (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.histplot(data=df, x='char_length', hue='label', bins=50, kde=True,
                 palette={'ham': '#2b6cb0', 'spam': '#e53e3e'}, alpha=0.6)
    plt.xlim(0, 350)
    plt.title('Message Character Length Distribution: Ham vs. Spam', fontsize=14, fontweight='bold')
    plt.xlabel('Character Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/message_length_distribution.png', dpi=120)
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
    
    # 3. Confusion Matrix (10x7.5 at 120 DPI -> 1200x900)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham (Legit)', 'Spam'],
                yticklabels=['Ham (Legit)', 'Spam'], cbar=True)
    plt.title('Multinomial Naive Bayes Confusion Matrix (Holdout Test Set)', fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=120)
    plt.close()
    
    # 4. ROC Curve (10x7.5 at 120 DPI -> 1200x900)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(fpr, tpr, color='#dd6b20', lw=2.5, label=f'Multinomial NB ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#2b6cb0', lw=2, linestyle='--', label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png', dpi=120)
    plt.close()
    
    # 5. Precision-Recall Curve (10x7.5 at 120 DPI -> 1200x900)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(recall, precision, color='#805ad5', lw=2.5, label=f'Precision-Recall Curve (AP = {ap:.4f})')
    plt.xlabel('Recall (Spam Detection Coverage)', fontsize=12)
    plt.ylabel('Precision (Spam Purity)', fontsize=12)
    plt.title('Precision-Recall Curve (Spam Minority Class Focus)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png', dpi=120)
    plt.close()
    
    # 6. Learning Curve (10x7.5 at 120 DPI -> 1200x900)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1', n_jobs=-1
    )
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#e53e3e', lw=2.5, label='Training F1-Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='#38a169', lw=2.5, label='Cross-Validation F1-Score')
    plt.xlabel('Training Corpus Size (Documents)', fontsize=12)
    plt.ylabel('Spam F1-Score', fontsize=12)
    plt.title('Naive Bayes Learning Curve: Asymptotic Generalization', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png', dpi=120)
    plt.close()
    
    # 7. Cross-Validation Scores (10x7.5 at 120 DPI -> 1200x900)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), cv_scores, color='#319795', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.axhline(np.mean(cv_scores), color='#e53e3e', linestyle='--', lw=2.5, label=f'Mean F1: {np.mean(cv_scores):.4f}')
    plt.xlabel('Cross-Validation Stratified Fold', fontsize=12)
    plt.ylabel('Spam F1-Score', fontsize=12)
    plt.ylim(0.85, 1.0)
    plt.title('5-Fold Cross-Validation Spam F1 Stability', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png', dpi=120)
    plt.close()
    
    # 8. Top Keywords Importance (10x7.5 at 120 DPI -> 1200x900)
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_prob_spam = model.feature_log_prob_[1]
    log_prob_ham = model.feature_log_prob_[0]
    llr = log_prob_spam - log_prob_ham
    top_spam_indices = np.argsort(llr)[-15:]
    
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.barh(range(len(top_spam_indices)), llr[top_spam_indices], color='#e53e3e', edgecolor='black')
    plt.yticks(range(len(top_spam_indices)), feature_names[top_spam_indices], fontsize=11)
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval + 0.1, bar.get_y() + bar.get_height()/2.0, f'{xval:.2f}', va='center', ha='left', fontsize=9, fontweight='bold')
    plt.title('Top 15 Informative Spam N-Grams (Log-Likelihood Ratio)', fontsize=14, fontweight='bold')
    plt.xlabel('Log-Likelihood Ratio [log P(w|Spam) - log P(w|Ham)]', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/top_keywords_importance.png', dpi=120)
    plt.close()
    
    print("✓ All 8 Spam Email visualizations generated at uniform (1200x900) resolution.")

if __name__ == '__main__':
    main()

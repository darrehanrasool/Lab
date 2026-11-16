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
    print("Generating visualizations for Sentiment Analysis of Text (Uniform 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    df = pd.read_csv('data/raw/sentiment.csv')
    df['word_count'] = df['review'].astype(str).apply(lambda s: len(s.split()))
    
    # 1. Sentiment Class Distribution (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    counts = df['sentiment'].value_counts()
    bars = plt.bar(counts.index.str.capitalize(), counts.values, color=['#38a169', '#e53e3e'], edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:,} ({yval/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.title('IMDB Movie Reviews Sentiment Class Balance', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment Polarity Category', fontsize=12)
    plt.ylabel('Review Count', fontsize=12)
    plt.ylim(0, max(counts.values) * 1.15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/sentiment_distribution.png', dpi=120)
    plt.close()
    
    # 2. Review Length Distribution (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.histplot(data=df[df['word_count'] <= 600], x='word_count', hue='sentiment', bins=40, kde=True,
                 palette={'positive': '#38a169', 'negative': '#e53e3e'}, alpha=0.6)
    plt.title('Review Word Count Distribution by Sentiment Polarity', fontsize=14, fontweight='bold')
    plt.xlabel('Word Count per Review', fontsize=12)
    plt.ylabel('Observation Frequency', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/review_length_distribution.png', dpi=120)
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
    
    # 3. Confusion Matrix (10x7.5 at 120 DPI -> 1200x900)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'], cbar=True)
    plt.title('Sentiment Classification Confusion Matrix (Holdout Test Set)', fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth Sentiment', fontsize=12)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=120)
    plt.close()
    
    # 4. ROC Curve (10x7.5 at 120 DPI -> 1200x900)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(fpr, tpr, color='#3182ce', lw=2.5, label=f'Logistic Regression ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#718096', lw=1.5, linestyle='--', label='Random Chance')
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
    plt.xlabel('Recall (Positive Sentiment Coverage)', fontsize=12)
    plt.ylabel('Precision (Positive Sentiment Purity)', fontsize=12)
    plt.title('Precision-Recall Curve (Sublinear TF-IDF)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png', dpi=120)
    plt.close()
    
    # 6. Learning Curve (10x7.5 at 120 DPI -> 1200x900)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#e53e3e', lw=2.5, label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='#38a169', lw=2.5, label='Cross-Validation Score')
    plt.xlabel('Training Corpus Sample Size', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Logistic Regression Learning Curve: Bias-Variance Diagnostics', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png', dpi=120)
    plt.close()
    
    # 7. Cross-Validation Scores (10x7.5 at 120 DPI -> 1200x900)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), cv_scores, color='#4fd1c5', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.axhline(np.mean(cv_scores), color='#e53e3e', linestyle='--', lw=2.5, label=f'Mean CV: {np.mean(cv_scores):.4f}')
    plt.xlabel('Cross-Validation Stratified Fold', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0.75, 0.95)
    plt.title('5-Fold Cross-Validation Accuracy Benchmark', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png', dpi=120)
    plt.close()
    
    # 8. Top Sentiment Words (Model Coefficients) (10x7.5 at 120 DPI -> 1200x900)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    
    top_pos_idx = np.argsort(coefs)[-12:]
    top_neg_idx = np.argsort(coefs)[:12]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 7.5), dpi=120)
    bars_neg = axes[0].barh(range(12), coefs[top_neg_idx], color='#e53e3e', edgecolor='black')
    axes[0].set_yticks(range(12))
    axes[0].set_yticklabels(feature_names[top_neg_idx], fontsize=10)
    axes[0].set_title('Top Negative N-Grams', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Coefficient Magnitude (Log-Odds)')
    axes[0].grid(axis='x', linestyle=':', alpha=0.6)
    
    bars_pos = axes[1].barh(range(12), coefs[top_pos_idx], color='#38a169', edgecolor='black')
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(feature_names[top_pos_idx], fontsize=10)
    axes[1].set_title('Top Positive N-Grams', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Coefficient Magnitude (Log-Odds)')
    axes[1].grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.suptitle('Logistic Regression Informative Feature Weights (Sublinear TF-IDF)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/top_sentiment_words.png', dpi=120)
    plt.close()
    
    print("✓ All 8 Sentiment Analysis visualizations saved in reports/figures/ at uniform (1200x900).")

if __name__ == '__main__':
    main()

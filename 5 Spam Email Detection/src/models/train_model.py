import os
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import cross_val_score

def main():
    print("Training model for Spam Email Detection...")
    os.makedirs('models', exist_ok=True)
    
    # Load processed matrices
    X_train = joblib.load('data/processed/X_train_tfidf.joblib')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val = joblib.load('data/processed/X_val_tfidf.joblib')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    X_test = joblib.load('data/processed/X_test_tfidf.joblib')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Train Multinomial Naive Bayes (Laplace smoothing alpha=0.1)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    
    # Validation evaluation
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    
    # Test evaluation
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    report = classification_report(y_test, test_preds, target_names=['Ham (Legitimate)', 'Spam'])
    
    # 5-fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    # Save model
    joblib.dump(model, 'models/spam_model.joblib')
    
    report_content = f"""Spam Email Detection Model Performance
========================================
Algorithm: Multinomial Naive Bayes (alpha=0.1) with TF-IDF Vectorizer (3,000 features, n-grams=(1,2))

Validation Set Metrics:
  - Accuracy: {val_acc:.4f}
  - F1 Score: {val_f1:.4f}

Test Set Evaluation:
  - Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)
  - Precision: {test_prec:.4f}
  - Recall: {test_rec:.4f}
  - F1-Score: {test_f1:.4f}
  - ROC-AUC: {test_auc:.4f}

Detailed Classification Report:
{report}

5-Fold Cross-Validation (F1-Score on Training Data):
  - Fold Scores: {[round(s, 4) for s in cv_scores]}
  - Mean F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print("Model saved to models/spam_model.joblib and accuracy.txt generated.")

if __name__ == '__main__':
    main()

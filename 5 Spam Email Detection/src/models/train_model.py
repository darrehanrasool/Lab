import os
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import cross_val_score

def main():
    print("Training models for Spam Email Detection (Syllabus: Naive Bayes, Logistic Regression, Linear SVM)...")
    os.makedirs('models', exist_ok=True)
    
    # Load processed matrices (70% Train, 20% Val, 10% Test)
    X_train = joblib.load('data/processed/X_train_tfidf.joblib')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val = joblib.load('data/processed/X_val_tfidf.joblib')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    X_test = joblib.load('data/processed/X_test_tfidf.joblib')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # 1. Primary Model: Multinomial Naive Bayes (Laplace smoothing alpha=0.1)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    report = classification_report(y_test, test_preds, target_names=['Ham (Legitimate)', 'Spam'])
    
    # 2. Discriminative Baseline: Logistic Regression (Sigmoid / Binary Cross-Entropy)
    log_reg = LogisticRegression(random_state=42, max_iter=200)
    log_reg.fit(X_train, y_train)
    log_preds = log_reg.predict(X_test)
    log_acc = accuracy_score(y_test, log_preds)
    log_f1 = f1_score(y_test, log_preds)
    log_prec = precision_score(y_test, log_preds)
    log_rec = recall_score(y_test, log_preds)
    
    # 3. Maximum Margin Classifier: Linear SVM (Linear SVC - No Kernels, C=1.0)
    linear_svm = LinearSVC(C=1.0, random_state=42)
    linear_svm.fit(X_train, y_train)
    svm_preds = linear_svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds)
    svm_prec = precision_score(y_test, svm_preds)
    svm_rec = recall_score(y_test, svm_preds)
    
    # 5-fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    # Save models
    joblib.dump(model, 'models/spam_model.joblib')
    joblib.dump(log_reg, 'models/logistic_model.joblib')
    joblib.dump(linear_svm, 'models/linear_svm_model.joblib')
    
    report_content = f"""Spam Email Detection Model Performance (Syllabus: Naive Bayes, Logistic Regression, Linear SVM)
=============================================================================================
Dataset Imbalance: 86.6% Ham (Legitimate) vs 13.4% Spam (70/20/10 Split)

1. Multinomial Naive Bayes (Laplace alpha=0.1):
   - Validation Accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}
   - Test Accuracy:       {test_acc:.4f} ({test_acc * 100:.2f}%)
   - Test Precision:      {test_prec:.4f}
   - Test Recall:         {test_rec:.4f}
   - Test F1-Score:       {test_f1:.4f}
   - ROC-AUC:             {test_auc:.4f}

2. Logistic Regression (Sigmoid Link, Log-Loss):
   - Test Accuracy:       {log_acc:.4f} ({log_acc * 100:.2f}%)
   - Test Precision:      {log_prec:.4f}
   - Test Recall:         {log_rec:.4f}
   - Test F1-Score:       {log_f1:.4f}

3. Linear Support Vector Machine (Linear SVC, C=1.0, No Kernels):
   - Test Accuracy:       {svm_acc:.4f} ({svm_acc * 100:.2f}%)
   - Test Precision:      {svm_prec:.4f}
   - Test Recall:         {svm_rec:.4f}
   - Test F1-Score:       {svm_f1:.4f}

Detailed Classification Report (Multinomial Naive Bayes):
{report}

5-Fold Cross-Validation (F1-Score on Training Data):
  - Fold Scores: {[round(s, 4) for s in cv_scores]}
  - Mean F1:     {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print("Model saved to models/spam_model.joblib and accuracy.txt generated.")

if __name__ == '__main__':
    main()

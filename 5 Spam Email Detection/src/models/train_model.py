"""
========================================================================================
PROJECT 5: SPAM EMAIL / SMS DETECTION
MODULE 3: MODEL TRAINING, MULTI-ALGORITHM BENCHMARKING (src/models/train_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: UCI / Almeida et al. SMS & Email Spam Collection (N = 5,572 Messages)

PURPOSE:
--------
1. Trains Multinomial Naive Bayes (primary generative model with Laplace smoothing alpha=0.1).
2. Trains Logistic Regression (discriminative probabilistic baseline with Sigmoid link).
3. Trains Linear Support Vector Machine (Linear SVC maximum-margin hyperplane, C=1.0).
4. Evaluates Accuracy, Precision, Recall, F1-Score, and ROC-AUC on the unseen holdout test set.
5. Computes 5-fold cross-validation on the training set.
6. Serializes model artifacts to `models/` and generates `accuracy.txt`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is the Naive Bayes assumption and how does it compute probabilities?
    -> It applies Bayes' Theorem with the 'naive' assumption of conditional feature independence given the class:
           P(Class = Spam | d) proportional to P(Spam) * product_{i=1}^V P(w_i | Spam)
       Because multiplying thousands of small probabilities causes arithmetic underflow, we compute in log-space:
           log P(Spam | d) = log P(Spam) + sum_{i=1}^V (TF-IDF_{i, d} * log P(w_i | Spam))

Q2: What is Laplace Smoothing (alpha = 0.1)?
    -> If a test email contains a legitimate word never seen in the spam training vocabulary,
       P(word | Spam) would be 0. Multiplying by 0 wipes out the entire probability!
       Laplace smoothing adds alpha to the numerator and alpha * |V| to the denominator:
           P(w_i | c) = (count(w_i, c) + alpha) / (total_words(c) + alpha * |V|)
       This guarantees non-zero probabilities for all vocabulary terms.

Q3: Why compare Naive Bayes to Logistic Regression and Linear SVM?
    -> Naive Bayes is a Generative classifier (models joint distribution P(X, Y)), which trains rapidly
       and performs exceptionally well on high-dimensional text.
       Logistic Regression and Linear SVM are Discriminative models (model decision boundary P(Y | X) directly).
       Comparing all three confirms whether linear hyperplanes or probabilistic word counts yield superior precision.
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def main() -> None:
    """Executes model training, multi-algorithm benchmarking, and evaluation metric logging."""
    print("==================================================================")
    print("PROJECT 5 - STEP 3: NAIVE BAYES, LOGISTIC REGRESSION & LINEAR SVM")
    print("==================================================================")
    os.makedirs("models", exist_ok=True)

    # 1. Load preprocessed TF-IDF matrices (70% Train, 20% Val, 10% Test)
    X_train = joblib.load("data/processed/X_train_tfidf.joblib")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    X_val = joblib.load("data/processed/X_val_tfidf.joblib")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    X_test = joblib.load("data/processed/X_test_tfidf.joblib")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # -------------------------------------------------------------------------
    # 2. PRIMARY MODEL: MULTINOMIAL NAIVE BAYES (alpha=0.1)
    # -------------------------------------------------------------------------
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    # Validation Set Evaluation (20%, N=1,114)
    val_preds = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1 = float(f1_score(y_val, val_preds))

    # Holdout Test Set Evaluation (10%, N=558)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_acc = float(accuracy_score(y_test, test_preds))
    test_prec = float(precision_score(y_test, test_preds))
    test_rec = float(recall_score(y_test, test_preds))
    test_f1 = float(f1_score(y_test, test_preds))
    test_auc = float(roc_auc_score(y_test, test_probs))
    report = classification_report(y_test, test_preds, target_names=["Ham (Legitimate)", "Spam"])

    # -------------------------------------------------------------------------
    # 3. DISCRIMINATIVE BASELINE: LOGISTIC REGRESSION
    # -------------------------------------------------------------------------
    log_reg = LogisticRegression(random_state=42, max_iter=200)
    log_reg.fit(X_train, y_train)
    log_preds = log_reg.predict(X_test)
    log_acc = float(accuracy_score(y_test, log_preds))
    log_prec = float(precision_score(y_test, log_preds))
    log_rec = float(recall_score(y_test, log_preds))
    log_f1 = float(f1_score(y_test, log_preds))

    # -------------------------------------------------------------------------
    # 4. MAXIMUM MARGIN CLASSIFIER: LINEAR SVM
    # -------------------------------------------------------------------------
    linear_svm = LinearSVC(C=1.0, random_state=42)
    linear_svm.fit(X_train, y_train)
    svm_preds = linear_svm.predict(X_test)
    svm_acc = float(accuracy_score(y_test, svm_preds))
    svm_prec = float(precision_score(y_test, svm_preds))
    svm_rec = float(recall_score(y_test, svm_preds))
    svm_f1 = float(f1_score(y_test, svm_preds))

    # 5-Fold Cross Validation on Training Split
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

    # -------------------------------------------------------------------------
    # 5. SERIALIZE ARTIFACTS
    # -------------------------------------------------------------------------
    joblib.dump(model, "models/spam_model.joblib")
    joblib.dump(log_reg, "models/logistic_model.joblib")
    joblib.dump(linear_svm, "models/linear_svm_model.joblib")

    # -------------------------------------------------------------------------
    # 6. GENERATE BENCHMARK REPORT (accuracy.txt)
    # -------------------------------------------------------------------------
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
"""
    with open("accuracy.txt", "w") as f:
        f.write(report_content)

    print(report_content)
    print("✓ Models saved to 'models/' and 'accuracy.txt' generated.")


if __name__ == "__main__":
    main()

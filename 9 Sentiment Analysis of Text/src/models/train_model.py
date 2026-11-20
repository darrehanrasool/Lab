"""
========================================================================================
PROJECT 9: SENTIMENT ANALYSIS OF TEXT
MODULE 3: LOGISTIC REGRESSION TRAINING & BENCHMARKING (src/models/train_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Maas et al. (2011) Stanford IMDB Large Movie Review Dataset (N = 6,000 Reviews)

PURPOSE:
--------
1. Fits a Logistic Regression classifier (C=1.0, L2 penalty) on 4,000-dimensional TF-IDF feature vectors.
2. Evaluates performance across Validation (15%) and Holdout Test (15%) sets (~86.9% accuracy, 0.9426 ROC-AUC).
3. Computes 5-fold cross-validation on the training cohort to verify stability.
4. Serializes the trained classifier to `models/sentiment_model.joblib` and generates `accuracy.txt`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why is Logistic Regression effective for text classification?
    -> Text feature matrices are extremely high-dimensional (4,000 TF-IDF features) but sparse (most entries are 0).
       Linear models like Logistic Regression scale efficiently with feature dimensionality, resist overfitting
       via L2 regularization (ridge penalty ||w||^2), and produce well-calibrated posterior probabilities:
           P(Positive | d) = 1 / (1 + exp(-(w^T * x + b)))

Q2: What does the parameter C = 1.0 represent?
    -> C is the inverse regularization strength (C = 1 / lambda). Smaller values enforce stronger regularization
       (shrinking weights towards zero to prevent overfitting), while larger values fit more closely to training data.

Q3: What does the ROC-AUC score of 0.9426 signify?
    -> Area Under the ROC Curve measures the probability that the model will rank a randomly chosen positive review
       higher than a randomly chosen negative review. A score of ~0.94 indicates outstanding ranking performance.
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


def main() -> None:
    """Executes model training, multi-split evaluation, cross-validation, and report writing."""
    print("==================================================================")
    print("PROJECT 9 - STEP 3: LOGISTIC REGRESSION SENTIMENT TRAINING")
    print("==================================================================")
    os.makedirs("models", exist_ok=True)

    # 1. Load preprocessed TF-IDF feature matrices (70% Train, 15% Val, 15% Test)
    X_train = joblib.load("data/processed/X_train_tfidf.joblib")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    X_val = joblib.load("data/processed/X_val_tfidf.joblib")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    X_test = joblib.load("data/processed/X_test_tfidf.joblib")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # -------------------------------------------------------------------------
    # 2. FIT LOGISTIC REGRESSION (C=1.0, L2 Regularization, Sigmoid Link)
    # -------------------------------------------------------------------------
    model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # 3. Validation Set Evaluation (15%, N=900)
    val_preds = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1 = float(f1_score(y_val, val_preds))

    # 4. Holdout Test Set Evaluation (15%, N=900)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_acc = float(accuracy_score(y_test, test_preds))
    test_prec = float(precision_score(y_test, test_preds))
    test_rec = float(recall_score(y_test, test_preds))
    test_f1 = float(f1_score(y_test, test_preds))
    test_auc = float(roc_auc_score(y_test, test_probs))
    report = classification_report(y_test, test_preds, target_names=["Negative", "Positive"])

    # 5. 5-Fold Cross Validation on Training Split
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    # 6. Serialize Trained Model
    joblib.dump(model, "models/sentiment_model.joblib")

    # 7. Generate Benchmark Report (accuracy.txt)
    report_content = f"""Sentiment Analysis of Text Model Performance
==============================================
Algorithm: Logistic Regression (C=1.0) with Sublinear TF-IDF (4,000 features, n-grams=(1,2))

Validation Set Metrics:
  - Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)
  - F1-Score: {val_f1:.4f}

Test Set Evaluation:
  - Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)
  - Precision: {test_prec:.4f}
  - Recall:    {test_rec:.4f}
  - F1-Score:  {test_f1:.4f}
  - ROC-AUC:   {test_auc:.4f}

Detailed Classification Report:
{report}

5-Fold Cross-Validation Accuracy on Training Data:
  - Fold Scores: {[round(float(s), 4) for s in cv_scores]}
  - Mean Accuracy: {float(np.mean(cv_scores)):.4f} (+/- {float(np.std(cv_scores)):.4f})
"""
    with open("accuracy.txt", "w") as f:
        f.write(report_content)

    print(report_content)
    print("✓ Model saved to 'models/sentiment_model.joblib' and 'accuracy.txt' generated.")


if __name__ == "__main__":
    main()

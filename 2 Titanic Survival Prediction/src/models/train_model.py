"""
========================================================================================
PROJECT 2: TITANIC SURVIVAL PREDICTION
MODULE 3: MODEL TRAINING, EVALUATION & DIAGNOSTICS (src/models/train_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: RMS Titanic Passenger Manifest (N = 891 Samples, Binary Classification)

PURPOSE:
--------
1. Trains an ensemble Random Forest Classifier (100 decision trees) on standardized features.
2. Evaluates performance on the unseen Holdout Test set (~77.6% accuracy).
3. Evaluates 5-fold cross-validation stability on the training cohort.
4. Generates publication-grade diagnostic plots (Confusion Matrix, ROC Curve, PR Curve, Learning Curve).
5. Persists the trained model artifact to `models/titanic_model.joblib` and writes `accuracy.txt`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is a Random Forest and why is it preferred over a single Decision Tree?
    -> A single decision tree has low bias but high variance (it easily overfits noisy training data).
       A Random Forest is an ensemble of B = 100 decorrelated decision trees built via:
         a) Bootstrap Aggregation (Bagging): Each tree trains on a random sample drawn with replacement.
         b) Random Feature Subspace: At each node split, only a random subset of m = sqrt(p) features
            is considered.
       Averaging the predictions across 100 decorrelated trees drastically reduces variance without increasing bias.

Q2: What is Gini Impurity?
    -> The criterion used to evaluate candidate node splits:
           Gini = 1 - sum_{i=1}^C (p_i)^2
       A pure node (all survived or all perished) has Gini = 0. The tree selects the feature and threshold
       that maximizes Gini Information Gain.

Q3: What does the ROC Curve and AUC measure?
    -> Receiver Operating Characteristic (ROC) plots True Positive Rate (Sensitivity) vs False Positive Rate
       across all classification probability thresholds [0.0 to 1.0]. An AUC of ~0.83 indicates strong
       discriminative capacity compared to a random guess (AUC = 0.50).
========================================================================================
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, learning_curve


def main() -> None:
    """Executes model training, multi-metric evaluation, diagnostic plotting, and artifact persistence."""
    print("==================================================================")
    print("PROJECT 2 - STEP 3: RANDOM FOREST ENSEMBLE TRAINING & DIAGNOSTICS")
    print("==================================================================")

    # 1. Load standardized training and test partitions
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # 2. Fit Random Forest Classifier
    # n_estimators=100: Ensemble of 100 decision trees
    # random_state=42: Deterministic bootstrap sampling
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Model Inference on Unseen Holdout Test Cohort
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]  # Posterior survival probability P(Survived=1|X)

    # 4. Quantitative Metrics Calculation
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, target_names=["Not Survived", "Survived"])

    # 5-Fold Cross Validation on Training Split
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    print(f"✓ Model Trained Successfully:")
    print(f"  - Holdout Test Accuracy: {acc * 100:.2f}%")
    print(f"  - 5-Fold CV Mean Score:  {float(np.mean(cv_scores)) * 100:.2f}% (±{float(np.std(cv_scores)) * 100:.2f}%)")

    # 5. Serialize Trained Model Artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/titanic_model.joblib")

    # Persist benchmark report
    with open("accuracy.txt", "w") as f:
        f.write(
            f"Accuracy: {acc:.4f}\n"
            f"5-Fold CV Mean Accuracy: {float(np.mean(cv_scores)):.4f} (±{float(np.std(cv_scores)):.4f})\n\n"
            f"Classification Report:\n{report}"
        )

    # 6. Generate Diagnostic Figures
    os.makedirs("reports/figures", exist_ok=True)

    # Figure 1: Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )
    plt.title("Confusion Matrix – Holdout Test Set")
    plt.ylabel("Actual Survival")
    plt.xlabel("Predicted Survival")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.close()

    # Figure 2: Receiver Operating Characteristic (ROC) Curve
    fpr, tpr, _ = roc_curve(y_test, preds_proba)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curve.png")
    plt.close()

    # Figure 3: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, preds_proba)
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(recall, precision, color="blue", lw=2, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("reports/figures/precision_recall_curve.png")
    plt.close()

    # Figure 4: Learning Curve (Diagnosing Bias vs Variance)
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
    )
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="r", label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), "o-", color="g", label="Cross-validation score")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.title("Random Forest Learning Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("reports/figures/learning_curve.png")
    plt.close()

    print("✓ Generated and saved all diagnostic plots to 'reports/figures/'.")
    print("✓ Model saved to 'models/titanic_model.joblib' and report saved to 'accuracy.txt'.")


if __name__ == "__main__":
    main()

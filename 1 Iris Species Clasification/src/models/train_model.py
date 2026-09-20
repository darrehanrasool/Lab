# =============================================================================
# train_model.py
# Iris Flower Classification – Full Training Pipeline
# =============================================================================

# ---------- 1. IMPORTS ----------
import os                          # For file system operations
import joblib                      # For saving the trained model
import numpy as np                 # For numerical operations
import pandas as pd                # For data manipulation
import matplotlib.pyplot as plt    # For plotting graphs
import seaborn as sns              # For statistical visualizations

from sklearn.datasets import load_iris            # Built‑in Iris dataset
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold

# ---------- 2. CONFIGURATION ----------
RANDOM_STATE = 42                  # Ensures reproducible results
TEST_SIZE = 0.2                    # 20% of data for testing
FIGURES_DIR = "reports/figures"    # Where to save all graphs
MODEL_DIR = "models"               # Where to save the trained model
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- 3. LOAD DATASET ----------
iris = load_iris()                 # Load the Iris dataset
X = iris.data                      # Features: sepal length, sepal width, petal length, petal width
y = iris.target                    # Target: 0=Setosa, 1=Versicolor, 2=Virginica
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")          # (150, 4)
print(f"Classes: {target_names}")

# ---------- 4. TRAIN / VALIDATION / TEST SPLIT ----------
# First split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
# Second split: 75% train, 25% validation from the train+val set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

# ---------- 5. FEATURE SCALING ----------
scaler = StandardScaler()                          # Standardize features (mean=0, std=1)
X_train_scaled = scaler.fit_transform(X_train)     # Fit on train, transform train
X_val_scaled = scaler.transform(X_val)             # Transform validation using train statistics
X_test_scaled = scaler.transform(X_test)           # Transform test using train statistics

# ---------- 6. MODEL TRAINING ----------
model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)                 # Train the model

# ---------- 7. PREDICTIONS ----------
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# Probabilities for ROC curves
y_test_proba = model.predict_proba(X_test_scaled)

# ---------- 8. METRICS ----------
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Save accuracy to file
with open("accuracy.txt", "w") as f:
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_test_pred, target_names=target_names)}")

# =============================================================================
# GRAPH 1: FEATURE DISTRIBUTIONS (Histograms)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
for i, feature in enumerate(feature_names):
    axes[i].hist(X[:, i], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel("Value (cm)")
    axes[i].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/01_feature_distributions.png", dpi=150)
plt.close()
print("Saved: 01_feature_distributions.png")

# =============================================================================
# GRAPH 2: PAIRPLOT (Seaborn)
# =============================================================================
df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]
sns.pairplot(df, hue="species", palette="viridis", diag_kind="kde")
plt.savefig(f"{FIGURES_DIR}/02_pairplot.png", dpi=150)
plt.close()
print("Saved: 02_pairplot.png")

# =============================================================================
# GRAPH 3: CORRELATION HEATMAP
# =============================================================================
plt.figure(figsize=(8, 6))
corr = df[feature_names].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{FIGURES_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: 03_correlation_heatmap.png")

# =============================================================================
# GRAPH 4: CONFUSION MATRIX
# =============================================================================
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix – Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{FIGURES_DIR}/04_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: 04_confusion_matrix.png")

# =============================================================================
# GRAPH 5: ROC CURVES (One-vs-Rest for 3 classes)
# =============================================================================
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(target_names):
    y_binary = (y_test == i).astype(int)          # Binary target for this class
    fpr, tpr, _ = roc_curve(y_binary, y_test_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – One-vs-Rest")
plt.legend(loc="lower right")
plt.savefig(f"{FIGURES_DIR}/05_roc_curves.png", dpi=150)
plt.close()
print("Saved: 05_roc_curves.png")

# =============================================================================
# GRAPH 6: PRECISION‑RECALL CURVES
# =============================================================================
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(target_names):
    y_binary = (y_test == i).astype(int)
    precision, recall, _ = precision_recall_curve(y_binary, y_test_proba[:, i])
    plt.plot(recall, precision, label=f"{class_name}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision‑Recall Curves")
plt.legend(loc="lower left")
plt.savefig(f"{FIGURES_DIR}/06_precision_recall_curves.png", dpi=150)
plt.close()
print("Saved: 06_precision_recall_curves.png")

# =============================================================================
# GRAPH 7: LEARNING CURVE (Training vs Validation Accuracy)
# =============================================================================
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy", random_state=RANDOM_STATE
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, "o-", label="Training Accuracy")
plt.plot(train_sizes, val_mean, "s-", label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.grid(True)
plt.savefig(f"{FIGURES_DIR}/07_learning_curve.png", dpi=150)
plt.close()
print("Saved: 07_learning_curve.png")

# =============================================================================
# GRAPH 8: CROSS‑VALIDATION SCORES
# =============================================================================
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), cv_scores, color="teal", edgecolor="black")
plt.axhline(y=cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.3f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("5‑Fold Cross‑Validation Accuracy")
plt.legend()
plt.savefig(f"{FIGURES_DIR}/08_cross_validation.png", dpi=150)
plt.close()
print("Saved: 08_cross_validation.png")

# =============================================================================
# 9. SAVE MODEL AND SCALER
# =============================================================================
joblib.dump(model, f"{MODEL_DIR}/logistic_regression_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
print(f"Model and scaler saved to {MODEL_DIR}/")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"All graphs saved in: {FIGURES_DIR}/")
print(f"Accuracy report saved in: accuracy.txt")
print("="*50)
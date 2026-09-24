# VIVA PREP GUIDE — Project 1: Iris Species Classification

## Algorithm: Logistic Regression | Task: Multiclass Classification

This document is built directly from your repo (`darrehanrasool/Lab`), reading the project's README, `train_model.py`, and related files. Nothing here is guessed — it's explained from your actual code.

**How to use this for the viva:** Read Section A first — this is the vocabulary examiners test regardless of what they point at. Then go through the project walkthrough and line-by-line code explanation. The "Likely Viva Questions" section at the end is the format examiners actually use — rehearse those out loud.

---

## SECTION A — Core ML Concepts You Must Know Cold

These concepts repeat across all three projects. If you understand this section, 80% of viva questions become easy regardless of which project is asked about.

## 0.1 Supervised Learning — the type of ML all 3 projects use

All three projects are **supervised learning**: you give the model input features `X` and the correct answer `y` for many examples, and it learns a function that maps `X → y`. There are two flavors used here:

| Type               | What it predicts            | Used in                                                           |
| ------------------ | --------------------------- | ----------------------------------------------------------------- |
| **Classification** | A category / discrete label | Project 1 (3 flower species), Project 2 (survived / not survived) |
| **Regression**     | A continuous number         | Project 3 (house price in $1000s)                                 |

## 0.2 Train / Test Split

You never train and test on the same data — that would let the model "cheat" by memorizing answers instead of learning patterns (this is called **overfitting** and gives a falsely high accuracy). So the dataset is split:

- **Training set** — the model learns from this (typically 70–80%)
- **Test set** — held back, never seen during training, used only to check how well the model generalizes to new/unseen data (typically 20–25%)

`random_state=42` is used everywhere in your code — this is a **seed** that makes the "random" split reproducible. Every time you re-run the code, you get the exact same split, so results are consistent and comparable.

`stratify=y` (used in Project 1) means the split preserves the same class proportions in both train and test sets — e.g., if 33% of the full dataset is Setosa, both the train and test sets will also be ~33% Setosa. This matters for classification so no split ends up starved of a class.

## 0.3 Validation Set (Project 1 only)

Project 1 splits data three ways: **Train / Validation / Test**.

- Train: fit the model's parameters
- Validation: tune decisions / check performance _during development_ without touching the test set
- Test: the final, untouched judge of real-world performance

This is more rigorous than Projects 2 & 3, which only do Train/Test (no separate validation split).

## 0.4 Feature Scaling — `StandardScaler`

Different features can live on wildly different numeric scales (e.g., Titanic's `fare` can be 0–500, while `pclass` is 1–3). Some algorithms (like Logistic Regression, which uses gradient-based optimization) are sensitive to this — large-scale features can dominate the learning purely because of their size, not because they're more important.

`StandardScaler` transforms every feature so it has:

- **Mean = 0**
- **Standard deviation = 1**

Formula: `z = (x − mean) / std`

**Critical rule (used correctly in all 3 projects):** `scaler.fit_transform()` is called ONLY on the training data. The scaler learns the mean/std from training data only. Then `scaler.transform()` (no `fit`) is applied to validation/test data using those _same_ training statistics. This prevents **data leakage** — if you let the scaler see test data statistics, information from the "unseen" test set would leak into training, giving an unrealistically optimistic result.

> Note: Random Forest (Projects 2 & 3) doesn't actually _need_ scaling — it splits on raw thresholds, not distances/gradients — but scaling doesn't hurt it either, so the same pipeline pattern is reused for consistency.

## 0.5 The Algorithms Used

**Logistic Regression (Project 1):** Despite the name, it's a _classification_ algorithm. It computes a weighted sum of the input features, then squashes the result through a **sigmoid/softmax function** into a probability between 0 and 1 (or, for 3 classes, a probability distribution over all 3 classes). It's a _linear_ model — it finds (in multi-dimensional space) boundaries that separate the classes.

**Random Forest (Projects 2 & 3):** An **ensemble** method — instead of one decision tree, it builds many decision trees (`n_estimators=100` → 100 trees) on random subsets of data and features, then combines their outputs:

- Classification (Project 2): each tree "votes" for a class, majority vote wins
- Regression (Project 3): each tree predicts a number, the forest averages all 100 predictions

Why Random Forest is popular: it reduces **overfitting** compared to a single decision tree (because errors of individual trees tend to cancel out when averaged/voted), handles non-linear relationships well, and needs little preprocessing.

## 0.6 Cross-Validation (`cv=5`, K-Fold)

Instead of a single train/test split, K-Fold cross-validation splits the training data into **K equal parts (folds)** — here K=5. The model is trained 5 times, each time using 4 folds to train and 1 different fold to validate. This gives 5 accuracy scores instead of 1, so you can see both the **average performance** and **how much it varies** (its stability) — a single split could get lucky or unlucky.

`StratifiedKFold` (Project 1) keeps class proportions balanced in every fold — same idea as `stratify` in the train/test split, applied per-fold.

## 0.7 Classification Metrics

**Accuracy** = (correct predictions) / (total predictions). Simple, but misleading on imbalanced data (e.g., if 90% of patients don't have a disease, always predicting "no disease" gives 90% accuracy while being useless).

**Confusion Matrix** — a table of actual vs. predicted classes:

|                     | Predicted Negative  | Predicted Positive  |
| ------------------- | ------------------- | ------------------- |
| **Actual Negative** | True Negative (TN)  | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP)  |

- **Precision** = TP / (TP + FP) → "Of everyone I predicted positive, how many actually were?" (matters when false positives are costly)
- **Recall (Sensitivity)** = TP / (TP + FN) → "Of everyone who was actually positive, how many did I catch?" (matters when false negatives are costly)
- **F1-score** = harmonic mean of precision and recall — a single number balancing both, useful when you care about both errors.

## 0.8 ROC Curve & AUC — the concept examiners love to probe

**ROC = Receiver Operating Characteristic curve.** It plots:

- **X-axis: False Positive Rate (FPR)** = FP / (FP + TN) — how often you wrongly flag a negative as positive
- **Y-axis: True Positive Rate (TPR)**, same as Recall = TP / (TP + FN)

A classifier doesn't just output a class — it outputs a **probability** (e.g., 0.73 chance of survival). To turn that into a yes/no decision, you need a **threshold** (default 0.5). The ROC curve is generated by sweeping that threshold from 0 to 1 and plotting the resulting (FPR, TPR) pair at every threshold. That's why it's a _curve_, not a point.

- The **diagonal dashed line** (`plt.plot([0,1],[0,1],'k--')`) represents a random guess (no skill) — a coin flip.
- A curve that bulges up toward the top-left corner (0 FPR, 1 TPR) is a better classifier.
- **AUC (Area Under the Curve)** condenses the whole curve into one number between 0 and 1:
  - AUC = 1.0 → perfect classifier
  - AUC = 0.5 → random guessing (no better than chance)
  - AUC < 0.5 → worse than random (something's wrong)
  - In practice: 0.9+ excellent, 0.8–0.9 good, 0.7–0.8 fair

**Why ROC and not just accuracy?** ROC/AUC is **threshold-independent** and robust to class imbalance — it tells you how well the model separates classes across _all_ possible decision thresholds, not just the default 0.5 one.

**Multiclass ROC (Project 1, Iris has 3 classes):** ROC is naturally a binary (2-class) tool. For 3 classes, your code uses **One-vs-Rest (OvR)**: for each class in turn, treat it as "positive" and all other classes as "negative," and draw a separate ROC curve. That's why Project 1's ROC graph has 3 curves (one per species), each with its own AUC.

## 0.9 Precision-Recall Curve

Similar idea to ROC, but plots **Precision (y-axis) vs. Recall (x-axis)** as the threshold sweeps. This is especially useful when classes are **imbalanced** (e.g., rare disease detection) because accuracy and even ROC can look deceptively good on imbalanced data, while PR curves expose it — a good PR curve stays high (near precision=1) across a wide range of recall.

## 0.10 Learning Curve

Plots model performance (accuracy or R²) on the **y-axis** against **training set size** on the **x-axis**, showing two lines: training score and validation/cross-validation score. It's generated by training the model repeatedly on increasing subsets of the training data (10%, 20%, ... 100%) and scoring each time. It diagnoses:

- **High gap between train and validation lines, both converging slowly** → **overfitting** (model memorizes training data, doesn't generalize)
- **Both lines low and close together** → **underfitting** (model too simple to capture the pattern)
- **Both lines converge to a high score** → good fit
- If validation score is still climbing as training size increases → more data would likely help.

## 0.11 Cross-Validation Score Bar/Box Plot

A simple visualization of the 5 accuracy (or R²) scores from 5-fold cross-validation — either as a bar chart per fold (Project 1) or a box plot showing their spread (Projects 2 & 3). It tells you the model's performance is **stable** if the scores are close together, or **unstable/sensitive to which data it sees** if they vary a lot.

## 0.12 Regression Metrics (Project 3 only — no classes, so accuracy doesn't apply)

**R² Score (Coefficient of Determination)** — how much of the variance in the target variable is explained by the model, from 0 to 1 (can go negative for a very bad model):

- R² = 1 → model predicts perfectly
- R² = 0 → model is no better than just predicting the mean every time
- Your Boston model: **R² ≈ 0.889** → the model explains ~89% of the variance in house prices.

**MSE (Mean Squared Error)** = average of (actual − predicted)² across all test samples. Squaring penalizes large errors more than small ones, and keeps everything positive. Lower is better. Your model: **MSE ≈ 8.13** (in units of price² — since MEDV is in $1000s, this means typical errors are around √8.13 ≈ $2,850).

## 0.13 Correlation Heatmap

Shows the **Pearson correlation coefficient** (−1 to +1) between every pair of numeric features (and the target). +1 = perfectly positively related, −1 = perfectly negatively related, 0 = no linear relationship. Used for **EDA (Exploratory Data Analysis)** — to spot which features are strongly related to the target (useful predictors) and which features are strongly correlated with _each other_ (redundant/multicollinear).

## 0.14 Pairplot

A grid of scatter plots, one for every pair of features, with histograms/KDE curves on the diagonal. Lets you visually spot which pairs of features separate the classes well (classification) or show a clear linear trend with the target (regression), and reveals feature distributions and outliers all in one figure.

---

## SECTION B — Project Walkthrough: Iris Species Classification

## 1.1 Overview

|                              |                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------- |
| **Task type**                | Multiclass classification (3 classes)                                         |
| **Algorithm**                | Logistic Regression                                                           |
| **Target**                   | Species: Setosa, Versicolor, Virginica                                        |
| **Test Accuracy**            | ~93.3% (see `accuracy.txt`)                                                   |
| **Data source used in code** | `sklearn.datasets.load_iris()` — a built-in dataset bundled with scikit-learn |

**⚠️ Important detail an examiner might probe:** your repo also has a `data/raw/Iris.csv` file with columns `Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species`. But if you look at `train_model.py`, it does **not** read this CSV — it loads data directly via `sklearn.datasets.load_iris()`, scikit-learn's built-in copy of the same dataset. The CSV in `data/raw/` appears to be there for reference/the `make_dataset.py` script but the actual training pipeline uses the built-in loader. Know this so you don't get caught saying "I loaded it from the CSV."

## 1.2 Dataset Structure

The classic **Fisher's Iris dataset** — 150 total samples, perfectly balanced: 50 of each species.

**4 input features** (all continuous, measured in cm):

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

**Target (`y`)**: one of 3 species, encoded as integers 0, 1, 2:

- 0 = Setosa
- 1 = Versicolor
- 2 = Virginica

Shape: `X.shape = (150, 4)`, `y.shape = (150,)`.

## 1.3 The Pipeline (What Happens, In Order)

1. Load the 150×4 dataset
2. Split: 80% train+val / 20% test (stratified so each split keeps ~33/33/33% class balance)
3. Split train+val again: 75% train / 25% validation (so final ratio ≈ 60% train / 20% val / 20% test)
4. Scale features with `StandardScaler` (fit on train only)
5. Train a `LogisticRegression` model on the scaled training data
6. Predict on train, validation, and test sets
7. Compute accuracy on all three, save to `accuracy.txt`
8. Generate 8 evaluation graphs
9. Save the trained model (`logistic_regression_model.pkl`) and the scaler (`scaler.pkl`) with `joblib` — so the web app can load them later without retraining

## 1.4 Why Logistic Regression here?

The Iris dataset is small, clean, and the 3 classes are known to be almost linearly separable (Setosa especially is trivially separable; Versicolor/Virginica overlap slightly). Logistic Regression is a simple, fast, interpretable linear model — a good fit for a dataset like this, and a good "first algorithm" to demonstrate the full ML pipeline. `max_iter=200` just gives the optimizer enough iterations to converge (default 100 sometimes isn't enough).

## 1.5 The 8 Graphs — Purpose of Each

| #   | File                             | What it shows                                                       | Why it's useful                                                                                                              |
| --- | -------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1   | `01_feature_distributions.png`   | Histogram of each of the 4 features across all 150 samples          | EDA — see the spread/shape of each feature before modeling                                                                   |
| 2   | `02_pairplot.png`                | Scatter plots of every feature pair, colored by species             | Shows visually how separable the 3 classes are — Setosa clusters distinctly; Versicolor/Virginica overlap a bit              |
| 3   | `03_correlation_heatmap.png`     | Correlation coefficients between the 4 features                     | Shows petal length & petal width are highly correlated with each other (and strong predictors)                               |
| 4   | `04_confusion_matrix.png`        | 3×3 grid of actual vs predicted species on the **test set**         | Shows exactly which species get confused with which (likely Versicolor ↔ Virginica, since Setosa is trivial)                 |
| 5   | `05_roc_curves.png`              | 3 ROC curves (One-vs-Rest), one per species, each with its AUC      | Shows how well the model separates each species from the other two, across all thresholds                                    |
| 6   | `06_precision_recall_curves.png` | 3 precision-recall curves, one per species                          | Complementary view to ROC — useful since even though classes are balanced here, it's still good practice                     |
| 7   | `07_learning_curve.png`          | Training accuracy vs validation accuracy as training set size grows | Diagnoses overfitting/underfitting — both curves should converge near a high accuracy for a well-fit small dataset like this |
| 8   | `08_cross_validation.png`        | Bar chart of accuracy across 5 CV folds + mean line                 | Shows stability of the model's performance across different data splits                                                      |

## 1.6 `train_model.py` — Line by Line

```python
# ---------- 1. IMPORTS ----------
import os                          # For file system operations
```

Standard library module for creating directories (`os.makedirs`).

```python
import joblib                      # For saving the trained model
```

`joblib` is used instead of Python's built-in `pickle` because it's more efficient for objects containing large NumPy arrays (like trained sklearn models) — common practice in ML pipelines.

```python
import numpy as np                 # For numerical operations
import pandas as pd                # For data manipulation
import matplotlib.pyplot as plt    # For plotting graphs
import seaborn as sns              # For statistical visualizations
```

`matplotlib` = low-level plotting; `seaborn` = built on top of matplotlib, gives nicer statistical plots (pairplot, heatmap) with less code.

```python
from sklearn.datasets import load_iris            # Built‑in Iris dataset
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
```

Every tool needed for the pipeline: data loading, splitting utilities, scaling, the model itself, and all the metric/curve functions. (Note: `StratifiedKFold` is imported here but not actually explicitly used later in the visible code — `cross_val_score(cv=5)` internally uses regular K-Fold by default for classifiers, not necessarily stratified unless specified; worth knowing this is an unused/leftover import if asked.)

```python
# ---------- 2. CONFIGURATION ----------
RANDOM_STATE = 42                  # Ensures reproducible results
TEST_SIZE = 0.2                    # 20% of data for testing
FIGURES_DIR = "reports/figures"    # Where to save all graphs
MODEL_DIR = "models"               # Where to save the trained model
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
```

Central config constants (good practice — one place to change settings). `exist_ok=True` means "don't error if the folder already exists."

```python
# ---------- 3. LOAD DATASET ----------
iris = load_iris()                 # Load the Iris dataset
X = iris.data                      # Features: sepal length, sepal width, petal length, petal width
y = iris.target                    # Target: 0=Setosa, 1=Versicolor, 2=Virginica
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")          # (150, 4)
print(f"Classes: {target_names}")
```

Loads the dataset object; `.data` is the 150×4 feature matrix, `.target` is the 150-length label array, `.feature_names`/`.target_names` are human-readable labels sklearn bundles with it.

```python
# ---------- 4. TRAIN / VALIDATION / TEST SPLIT ----------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
```

Two-stage split. First carve off 20% as test (never touched until final evaluation). Then take the remaining 80% and split again: 25% of that 80% (=20% of the total) becomes validation, leaving 60% of the total as pure training. `stratify=` on both splits keeps class ratios balanced everywhere.

```python
# ---------- 5. FEATURE SCALING ----------
scaler = StandardScaler()                          # Standardize features (mean=0, std=1)
X_train_scaled = scaler.fit_transform(X_train)     # Fit on train, transform train
X_val_scaled = scaler.transform(X_val)             # Transform validation using train statistics
X_test_scaled = scaler.transform(X_test)           # Transform test using train statistics
```

`fit_transform` on train computes mean/std from training data AND applies the transformation in one step. `transform` (no fit) on val/test reuses those same train statistics — this is the data-leakage-prevention rule from Part 0.4.

```python
# ---------- 6. MODEL TRAINING ----------
model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)                 # Train the model
```

Instantiates the model (`max_iter=200` = max optimization iterations for convergence) and fits it — this is where the model actually "learns" the weights that separate the 3 classes.

```python
# ---------- 7. PREDICTIONS ----------
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# Probabilities for ROC curves
y_test_proba = model.predict_proba(X_test_scaled)
```

`.predict()` returns hard class labels (0/1/2). `.predict_proba()` returns a probability for each of the 3 classes per sample (needed to draw ROC/PR curves, which sweep across probability thresholds rather than just looking at final labels).

```python
# ---------- 8. METRICS ----------
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

with open("accuracy.txt", "w") as f:
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_test_pred, target_names=target_names)}")
```

Computes accuracy on all three sets, prints them, and writes a readable report to `accuracy.txt` including a full `classification_report` (precision/recall/F1 per class, plus overall accuracy/macro-avg/weighted-avg) on the test set.

```python
# GRAPH 1: FEATURE DISTRIBUTIONS
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
```

Creates a 2×2 grid of subplots (4 features). `.ravel()` flattens the 2D grid of axes into a 1D array so it can be looped with a single index. Each subplot is a histogram (20 bins) of one feature across all 150 samples. `dpi=150` = image resolution. `plt.close()` frees memory / prevents overlapping plots.

```python
# GRAPH 2: PAIRPLOT
df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]
sns.pairplot(df, hue="species", palette="viridis", diag_kind="kde")
plt.savefig(f"{FIGURES_DIR}/02_pairplot.png", dpi=150)
plt.close()
```

Converts the NumPy array into a pandas DataFrame (seaborn needs a DataFrame), adds a readable species column. `hue="species"` colors points by class. `diag_kind="kde"` uses smoothed density curves (instead of histograms) on the diagonal.

```python
# GRAPH 3: CORRELATION HEATMAP
plt.figure(figsize=(8, 6))
corr = df[feature_names].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{FIGURES_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()
```

`.corr()` computes the pairwise Pearson correlation matrix. `annot=True` prints the numeric value in each cell; `fmt=".2f"` = 2 decimal places; `cmap="coolwarm"` = red (positive) to blue (negative) color scale.

```python
# GRAPH 4: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix – Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{FIGURES_DIR}/04_confusion_matrix.png", dpi=150)
plt.close()
```

`confusion_matrix()` builds the 3×3 table of actual (rows) vs predicted (columns) counts on the **test set only**. `fmt="d"` = display as integers (counts), not decimals.

```python
# GRAPH 5: ROC CURVES (One-vs-Rest for 3 classes)
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
```

Loops over each of the 3 species. `y_binary = (y_test == i)` converts the multiclass problem into "is it this species (1) or not (0)" — the One-vs-Rest trick from Part 0.8. `y_test_proba[:, i]` grabs the predicted probability of _this specific class_ for every test sample. `roc_curve()` returns arrays of FPR and TPR at many thresholds; `auc()` integrates the area under that curve. The `_` throwaway variable is the array of threshold values (not needed for plotting).

```python
# GRAPH 6: PRECISION‑RECALL CURVES
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
```

Same One-vs-Rest binarization trick, but plots precision against recall instead of TPR against FPR.

```python
# GRAPH 7: LEARNING CURVE
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
```

`learning_curve()` internally retrains the model at 10 increasing training-set sizes (10%, 20%, ... 100%, via `np.linspace(0.1, 1.0, 10)`), each time using 5-fold CV (`cv=5`), and returns a score matrix (rows=sizes, columns=folds). `np.mean(..., axis=1)` averages across the 5 folds for each size, leaving one mean score per training size.

```python
# GRAPH 8: CROSS‑VALIDATION SCORES
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
```

`cross_val_score()` runs the full 5-fold CV process and directly returns the 5 accuracy numbers (one per fold). `plt.bar(range(1,6), ...)` makes a bar per fold (folds 1–5); the red dashed line marks the mean.

```python
# SAVE MODEL AND SCALER
joblib.dump(model, f"{MODEL_DIR}/logistic_regression_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
```

Serializes the trained model and the fitted scaler to disk so the Flask web app (`web/app.py`) can load them later and make predictions on new user input without retraining. **Both** must be saved — the scaler is needed because new input at prediction time must be scaled the exact same way the training data was.

## 1.7 Likely Viva Questions — Project 1

- **Q: Why 3-way split instead of just train/test?** → To have a validation set for tuning/sanity-checking decisions without contaminating the final test evaluation.
- **Q: Why scale the features?** → Logistic Regression's optimizer is sensitive to feature scale; scaling puts sepal/petal measurements on equal footing.
- **Q: Why stratify?** → To keep the 3-class balance (50/50/50) consistent across train/val/test, avoiding a split that's accidentally missing/skewed on one class.
- **Q: What does One-vs-Rest mean and why is it needed here?** → ROC curves are inherently binary; with 3 classes you binarize by treating one class as "positive" and the rest as "negative," repeated once per class.
- **Q: What would an AUC of 0.5 mean?** → The model is no better than random guessing for that class.
- **Q: Why is Setosa's AUC likely near 1.0 but Versicolor/Virginica lower?** → Setosa is linearly/visibly distinct in petal measurements (seen in the pairplot); Versicolor and Virginica have overlapping feature ranges, making them harder to separate.
- **Q: Why Logistic Regression and not something more complex?** → Small, clean, near-linearly-separable dataset — a simple interpretable model is appropriate and avoids unnecessary complexity/overfitting risk.

---

## -e

## SECTION C — Rapid-Fire Definitions (memorize these one-liners)

- **Feature**: an input variable used to make a prediction.
- **Label/Target**: the value you're trying to predict.
- **Overfitting**: model learns training data too specifically (including its noise), performs worse on new data.
- **Underfitting**: model is too simple to capture the underlying pattern, performs poorly everywhere.
- **Hyperparameter**: a setting you choose before training (e.g., `n_estimators=100`, `max_iter=200`) — not learned from data.
- **Ensemble method**: combines multiple models to get a better/more stable result than any single model (Random Forest = ensemble of decision trees).
- **Bootstrap sampling**: sampling with replacement — each tree in a Random Forest is trained on a random sample of the data (some rows repeated, some left out).
- **Data leakage**: information from outside the training set (often future/test info) accidentally influencing training, giving unrealistically good results.
- **Joblib/Pickle**: serialize (save) a trained Python object to disk so it can be reloaded later without retraining.
- **EDA (Exploratory Data Analysis)**: examining and visualizing data before/independent of modeling, to understand its structure, distributions, and relationships.

---

_Good luck with your viva. Read Section A twice, then walk through Section B's Likely Viva Questions out loud._

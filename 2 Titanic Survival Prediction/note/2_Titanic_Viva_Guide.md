# VIVA PREP GUIDE — Project 2: Titanic Survival Prediction
### Algorithm: Random Forest Classifier | Task: Binary Classification

This document is built directly from your repo (`darrehanrasool/Lab`), reading the project's README, `build_features.py`, `train_model.py`, and related files. Nothing here is guessed — it's explained from your actual code.

**How to use this for the viva:** Read Section A first — this is the vocabulary examiners test regardless of what they point at. Then go through the project walkthrough and line-by-line code explanation. The "Likely Viva Questions" section at the end is the format examiners actually use — rehearse those out loud.

---

# SECTION A — Core ML Concepts You Must Know Cold

These concepts repeat across all three projects. If you understand this section, 80% of viva questions become easy regardless of which project is asked about.

## 0.1 Supervised Learning — the type of ML all 3 projects use
All three projects are **supervised learning**: you give the model input features `X` and the correct answer `y` for many examples, and it learns a function that maps `X → y`. There are two flavors used here:

| Type | What it predicts | Used in |
|---|---|---|
| **Classification** | A category / discrete label | Project 1 (3 flower species), Project 2 (survived / not survived) |
| **Regression** | A continuous number | Project 3 (house price in $1000s) |

## 0.2 Train / Test Split
You never train and test on the same data — that would let the model "cheat" by memorizing answers instead of learning patterns (this is called **overfitting** and gives a falsely high accuracy). So the dataset is split:
- **Training set** — the model learns from this (typically 70–80%)
- **Test set** — held back, never seen during training, used only to check how well the model generalizes to new/unseen data (typically 20–25%)

`random_state=42` is used everywhere in your code — this is a **seed** that makes the "random" split reproducible. Every time you re-run the code, you get the exact same split, so results are consistent and comparable.

`stratify=y` (used in Project 1) means the split preserves the same class proportions in both train and test sets — e.g., if 33% of the full dataset is Setosa, both the train and test sets will also be ~33% Setosa. This matters for classification so no split ends up starved of a class.

## 0.3 Validation Set (Project 1 only)
Project 1 splits data three ways: **Train / Validation / Test**.
- Train: fit the model's parameters
- Validation: tune decisions / check performance *during development* without touching the test set
- Test: the final, untouched judge of real-world performance

This is more rigorous than Projects 2 & 3, which only do Train/Test (no separate validation split).

## 0.4 Feature Scaling — `StandardScaler`
Different features can live on wildly different numeric scales (e.g., Titanic's `fare` can be 0–500, while `pclass` is 1–3). Some algorithms (like Logistic Regression, which uses gradient-based optimization) are sensitive to this — large-scale features can dominate the learning purely because of their size, not because they're more important.

`StandardScaler` transforms every feature so it has:
- **Mean = 0**
- **Standard deviation = 1**

Formula: `z = (x − mean) / std`

**Critical rule (used correctly in all 3 projects):** `scaler.fit_transform()` is called ONLY on the training data. The scaler learns the mean/std from training data only. Then `scaler.transform()` (no `fit`) is applied to validation/test data using those *same* training statistics. This prevents **data leakage** — if you let the scaler see test data statistics, information from the "unseen" test set would leak into training, giving an unrealistically optimistic result.

> Note: Random Forest (Projects 2 & 3) doesn't actually *need* scaling — it splits on raw thresholds, not distances/gradients — but scaling doesn't hurt it either, so the same pipeline pattern is reused for consistency.

## 0.5 The Algorithms Used

**Logistic Regression (Project 1):** Despite the name, it's a *classification* algorithm. It computes a weighted sum of the input features, then squashes the result through a **sigmoid/softmax function** into a probability between 0 and 1 (or, for 3 classes, a probability distribution over all 3 classes). It's a *linear* model — it finds (in multi-dimensional space) boundaries that separate the classes.

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

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

- **Precision** = TP / (TP + FP) → "Of everyone I predicted positive, how many actually were?" (matters when false positives are costly)
- **Recall (Sensitivity)** = TP / (TP + FN) → "Of everyone who was actually positive, how many did I catch?" (matters when false negatives are costly)
- **F1-score** = harmonic mean of precision and recall — a single number balancing both, useful when you care about both errors.

## 0.8 ROC Curve & AUC — the concept examiners love to probe
**ROC = Receiver Operating Characteristic curve.** It plots:
- **X-axis: False Positive Rate (FPR)** = FP / (FP + TN) — how often you wrongly flag a negative as positive
- **Y-axis: True Positive Rate (TPR)**, same as Recall = TP / (TP + FN)

A classifier doesn't just output a class — it outputs a **probability** (e.g., 0.73 chance of survival). To turn that into a yes/no decision, you need a **threshold** (default 0.5). The ROC curve is generated by sweeping that threshold from 0 to 1 and plotting the resulting (FPR, TPR) pair at every threshold. That's why it's a *curve*, not a point.

- The **diagonal dashed line** (`plt.plot([0,1],[0,1],'k--')`) represents a random guess (no skill) — a coin flip.
- A curve that bulges up toward the top-left corner (0 FPR, 1 TPR) is a better classifier.
- **AUC (Area Under the Curve)** condenses the whole curve into one number between 0 and 1:
  - AUC = 1.0 → perfect classifier
  - AUC = 0.5 → random guessing (no better than chance)
  - AUC < 0.5 → worse than random (something's wrong)
  - In practice: 0.9+ excellent, 0.8–0.9 good, 0.7–0.8 fair

**Why ROC and not just accuracy?** ROC/AUC is **threshold-independent** and robust to class imbalance — it tells you how well the model separates classes across *all* possible decision thresholds, not just the default 0.5 one.

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
Shows the **Pearson correlation coefficient** (−1 to +1) between every pair of numeric features (and the target). +1 = perfectly positively related, −1 = perfectly negatively related, 0 = no linear relationship. Used for **EDA (Exploratory Data Analysis)** — to spot which features are strongly related to the target (useful predictors) and which features are strongly correlated with *each other* (redundant/multicollinear).

## 0.14 Pairplot
A grid of scatter plots, one for every pair of features, with histograms/KDE curves on the diagonal. Lets you visually spot which pairs of features separate the classes well (classification) or show a clear linear trend with the target (regression), and reveals feature distributions and outliers all in one figure.

---

# SECTION B — Project Walkthrough: Titanic Survival Prediction

## 2.1 Overview
| | |
|---|---|
| **Task type** | Binary classification |
| **Algorithm** | Random Forest Classifier (`n_estimators=100`) |
| **Target** | `survived` — 0 = Not Survived, 1 = Survived |
| **Test Accuracy** | ~80.4% (see `accuracy.txt`) |
| **Data source** | Loaded via Seaborn's built-in Titanic dataset (`sns.load_dataset('titanic')`) |

Unlike Project 1, this project's pipeline is split across **three separate scripts** that must be run in order:
1. `src/data/make_dataset.py` — fetches raw data
2. `src/features/build_features.py` — cleans + preprocesses + splits + scales + saves to disk
3. `src/models/train_model.py` — loads the already-processed data, trains, evaluates
(and separately, `src/visualization/visualize.py` regenerates all graphs, including redoing some EDA plots from scratch)

This is a more "production-style" structure than Project 1 (which does everything in one script) — each stage's output is written to disk so later stages don't need to repeat earlier work.

## 2.2 Dataset Structure
Seaborn's Titanic dataset has 15 raw columns (`survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, alone`), but `build_features.py` keeps only 7 for modeling:

| Column | Meaning | Type |
|---|---|---|
| `survived` | Target: 0/1 | binary |
| `pclass` | Ticket class: 1st/2nd/3rd | ordinal (1,2,3) |
| `sex` | Passenger sex | categorical → encoded to 0/1 |
| `age` | Age in years | numeric (has missing values) |
| `sibsp` | # siblings/spouses aboard | numeric |
| `parch` | # parents/children aboard | numeric |
| `fare` | Ticket fare | numeric |

~891 total passengers (the classic Titanic training dataset size).

## 2.3 `build_features.py` — Preprocessing (Line by Line)
```python
df = pd.read_csv('data/raw/titanic.csv')

df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
```
Loads the raw CSV, then keeps only the 7 relevant columns — drops columns like `deck` (too many missing values), `embark_town`/`embarked` (redundant with `pclass` roughly, and adds complexity), `who`/`adult_male`/`alive`/`alone`/`class` (these are largely derived from — or duplicates of — columns already kept, e.g. `alive` is just `survived` as text, which would be **target leakage** if kept as a feature).

```python
df['age'] = df['age'].fillna(df['age'].median())
```
`age` has missing values (some passengers' ages weren't recorded). **Median imputation**: replaces missing values with the median age of all passengers. Median is preferred over mean here because age distributions can be skewed, and median is robust to outliers.

```python
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
```
**Label encoding**: converts the text category `sex` into numbers, since ML models need numeric input. Male → 0, Female → 1.

```python
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Separates features from target, then does an 80/20 train/test split. (Note: unlike Project 1, there's **no `stratify=y` here** — worth knowing, since survival classes aren't perfectly balanced ~62%/38%, an examiner might ask why stratification wasn't used; the honest answer is it wasn't applied in this script, which is a minor gap compared to Project 1's more careful approach.)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('data/processed', exist_ok=True)
joblib.dump(scaler, 'data/processed/scaler.joblib')

pd.DataFrame(X_train_scaled).to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
```
Same fit-on-train-only scaling rule as Project 1. Everything is written to `data/processed/` as CSVs (plus the scaler as a `.joblib` file) so `train_model.py` can just read these directly rather than repeating preprocessing.

## 2.4 Why Random Forest here (not Logistic Regression)?
Titanic survival depends on **interactions** between features (e.g., "1st class AND female" survives much more than either factor alone) and non-linear effects (e.g., very young children had different survival odds than the general age trend). Random Forest naturally captures these feature interactions and non-linearities without needing you to manually engineer interaction terms, unlike plain Logistic Regression.

## 2.5 The 8 Graphs — Purpose of Each

| File | What it shows | Why it's useful |
|---|---|---|
| `pairplot.png` | Scatter plots of the 6 features, colored by survival | Visual EDA — spot which features separate survivors from non-survivors |
| `heatmap.png` | Correlation matrix of all 7 columns (incl. `survived`) | Shows `sex` and `pclass`/`fare` are the strongest correlates of survival |
| `feature_distributions.png` | Histogram+KDE of each of the 6 features, split by survival (hue) | Shows e.g. that survival rate skews toward lower `pclass` number (1st class) and higher `fare` |
| `confusion_matrix.png` | Actual vs predicted survival on test set | Shows counts of correctly/incorrectly predicted survivors and non-survivors |
| `roc_curve.png` | Single ROC curve (binary problem, no OvR needed) with AUC | Shows the model's ability to rank survivors above non-survivors across thresholds |
| `precision_recall_curve.png` | Precision vs recall trade-off curve | Useful given survival classes aren't perfectly balanced |
| `learning_curve.png` | Train vs CV accuracy as training size grows | Diagnoses over/underfitting |
| `cross_validation.png` | Box plot of 5-fold CV accuracy scores | Shows spread/stability of performance (box plot instead of bar chart, unlike Project 1) |

Note: `visualize.py` regenerates *all* of these (both EDA and evaluation graphs) at larger figure sizes (10×8) than `train_model.py`'s versions — `train_model.py` also makes 4 of these itself (confusion matrix, ROC, PR curve, learning curve) during training as a quick sanity check, but the "official" full set of 8 comes from running `visualize.py` afterward, per the README's Quick Start steps.

## 2.6 `train_model.py` — Line by Line

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, classification_report)
from sklearn.model_selection import learning_curve, cross_val_score
import joblib
import os
```
Imports the classifier version of Random Forest, all metric functions, and utilities. Notice `train_test_split`/`StandardScaler` are NOT imported here — because splitting/scaling already happened in `build_features.py`; this script just consumes the already-processed data.

```python
def main():
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
```
Loads the already-scaled/split data saved by `build_features.py`. `.values.ravel()` converts the DataFrame column into a flat 1D NumPy array — `y_train.csv` when read by pandas becomes a DataFrame with 1 column, but sklearn models expect a 1D array for labels, so `.values` gets the underlying NumPy array and `.ravel()` flattens it from shape `(n,1)` to `(n,)`.

```python
    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
```
`n_estimators=100` = build 100 decision trees. `random_state=42` = reproducible randomness (which samples/features each tree sees). `.fit()` trains all 100 trees on bootstrap samples of the training data.

```python
    # Predictions
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1] # Probability for ROC/PR curves
```
`.predict()` gives hard 0/1 labels (majority vote across the 100 trees). `.predict_proba(X_test)` returns a 2-column array (probability of class 0, probability of class 1) for each sample; `[:, 1]` grabs just the "probability of survival" column, which is what's needed to draw ROC/PR curves.

```python
    # 1. Accuracy & Report
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=['Not Survived', 'Survived'])
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/titanic_model.joblib')
    
    with open('accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {acc}\n\nClassification Report:\n{report}")
```
Computes overall accuracy and the detailed per-class report, saves the trained model to disk, writes results to `accuracy.txt`.

```python
    os.makedirs('reports/figures', exist_ok=True)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
```
Same confusion matrix logic as Project 1, but binary (2×2) instead of 3×3. `plt.tight_layout()` auto-adjusts spacing so labels don't get cut off.

```python
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, preds_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png')
    plt.close()
```
No One-vs-Rest needed here since this is a genuinely binary problem — `roc_curve()` is called directly with the survival probabilities. `lw=2` = line width 2.

```python
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, preds_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png')
    plt.close()
```
Standard binary PR curve, same logic as Part 0.9.

```python
    # 5. Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()
```
Same idea as Project 1's learning curve, but only 5 training-size steps (`np.linspace(0.1,1.0,5)` = 10%,32.5%,55%,77.5%,100%) instead of 10, and `n_jobs=-1` uses all CPU cores in parallel to speed up the repeated retraining (Random Forest is slower to train than Logistic Regression, so parallelizing matters more here).

```python
    # 6. Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Model trained. Accuracy: {acc}")
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
```
Runs 5-fold CV and prints (not plots — the boxplot version is in `visualize.py`) the mean accuracy and its variability. `cv_scores.std() * 2` is roughly a 95% confidence interval width (2 standard deviations), a common shorthand for "the uncertainty range."

## 2.7 Likely Viva Questions — Project 2
- **Q: Why is preprocessing in a separate file (`build_features.py`) instead of inside `train_model.py`?** → Separation of concerns / modular pipeline design — so preprocessing doesn't need to be repeated every time you retrain, and other scripts (like `visualize.py`) can reuse the same processed data.
- **Q: Why drop columns like `deck`, `embarked`, `alive`?** → `deck` has too many missing values to be reliable; `alive` is literally the same information as the target (`survived`) in text form, so including it would be target leakage (the model would "cheat").
- **Q: Why median instead of mean to fill missing ages?** → Median is robust to outliers/skew; a few very old or very young outliers won't distort it the way they'd distort a mean.
- **Q: Why Random Forest over Logistic Regression here?** → Survival depends on interactions between features (class × sex × age), which tree-based ensembles capture naturally, whereas Logistic Regression models a single linear combination.
- **Q: What does `n_estimators=100` control?** → The number of individual decision trees in the forest; more trees generally means more stable predictions (via averaging/voting) up to a point of diminishing returns.
- **Q: What's the difference between this project's ROC curve and Project 1's?** → This one is a true binary ROC curve — no One-vs-Rest transformation needed, since there are only 2 classes.

---

-e 
---

# SECTION C — Rapid-Fire Definitions (memorize these one-liners)

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

*Good luck with your viva. Read Section A twice, then walk through Section B's Likely Viva Questions out loud.*

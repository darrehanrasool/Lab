# VIVA PREP GUIDE — Project 3: Boston Housing Price Prediction
### Algorithm: Random Forest Regressor | Task: Regression

This document is built directly from your repo (`darrehanrasool/Lab`), reading the project's README, `make_dataset.py`, `build_features.py`, `train_model.py`, and related files. Nothing here is guessed — it's explained from your actual code.

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

# SECTION B — Project Walkthrough: Boston Housing Price Prediction

## 3.1 Overview
| | |
|---|---|
| **Task type** | Regression (predicting a continuous number, not a class) |
| **Algorithm** | Random Forest Regressor (`n_estimators=100`) |
| **Target** | `MEDV` — median value of owner-occupied homes, in $1000s |
| **Test R² Score** | ~0.889 |
| **Test MSE** | ~8.13 |
| **Data source** | Fetched live from `http://lib.stat.cmu.edu/datasets/boston` (the classic Boston Housing dataset — note: this dataset has been deprecated/removed from scikit-learn itself in recent versions due to ethical concerns about one feature, `B`, but is still available as a raw CSV/text source, which is what this script fetches) |

This is the only project of the three that is **regression, not classification** — so accuracy/confusion-matrix/ROC concepts don't apply here at all. Know this distinction cold; it's an easy trap question ("what's the accuracy of your Boston model?" — trick question, regression models don't have "accuracy," they have R²/MSE).

## 3.2 Dataset Structure
506 rows (neighborhoods/towns around Boston), 13 input features + 1 target:

| Column | Meaning |
|---|---|
| `CRIM` | Per-capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable (1 if tract bounds the river, else 0) |
| `NOX` | Nitric oxide concentration (air pollution proxy) |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to Boston employment centers |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Property tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | A formula involving proportion of Black residents by town (this feature is the reason the dataset is considered ethically problematic and was removed from scikit-learn; worth mentioning briefly if the examiner asks about the dataset's known controversy) |
| `LSTAT` | % lower status of the population |
| `MEDV` | **Target**: Median home value in $1000s |

## 3.3 `make_dataset.py` — Line by Line (unique to this project — a quirky data format)
```python
url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
```
Downloads the raw text file directly from the source. `sep="\s+"` means "split on any amount of whitespace" (the file uses variable spacing, not commas). `skiprows=22` skips the file's text description/header at the top. `header=None` because the raw file has no column-name row.

```python
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
```
This is the unusual/tricky part: the raw file stores each data record across **two physical lines** (a historical quirk of this dataset's original format). `raw_df.values[::2, :]` takes every even-indexed row (the first line of each record, containing the first 11 features), `raw_df.values[1::2, :2]` takes every odd-indexed row but only its first 2 columns (the last 2 features, `B` and `LSTAT`, continued on the second line). `np.hstack` glues them side by side into full 13-feature rows. `target = raw_df.values[1::2, 2]` grabs the 3rd column of every odd row — that's where `MEDV` (the price) is stored in this format.

```python
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(data, columns=columns)
df['MEDV'] = target
```
Manually assigns the 13 known column names (since the raw file had none) and appends the target column.

## 3.4 `build_features.py` — What's Different Here
Same overall pattern as Titanic's (split → scale → save), but notably **simpler** — no missing-value handling and no categorical encoding needed, because the Boston dataset is already fully numeric with no missing values. This is worth noting if asked to compare preprocessing complexity across your 3 projects: Iris (none needed, uses built-in clean data) < Boston (numeric, no missing values, just split+scale) < Titanic (missing values + categorical encoding + column dropping).

```python
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
Same fit-on-train-only scaling rule. No stratification here — stratification is a classification concept (balancing discrete classes) and doesn't apply to continuous regression targets.

## 3.5 Why Random Forest Regressor here?
Housing prices are influenced by complex, non-linear combinations of features (e.g., the effect of `RM` — rooms per dwelling — on price is not simply linear across the whole range; crime rate's effect may plateau; interactions like "low crime AND high rooms" compound). Random Forest Regressor captures these non-linear relationships and interactions automatically by averaging many decision trees, without needing manual feature engineering (like polynomial terms) that a plain Linear Regression would require.

## 3.6 The 7 Graphs — Purpose of Each
(README lists 7 for this project — one fewer than Projects 2/3's classifiers, because there's no confusion matrix, ROC, or PR curve for regression — those two extra graphs are replaced by regression-specific ones: Actual vs Predicted, and Residual Plot.)

| File | What it shows | Why it's useful |
|---|---|---|
| `pairplot.png` | Scatter plots among `MEDV` and 4 key features (`RM`, `LSTAT`, `CRIM`, `AGE`) | Shows visually which features have a linear-ish trend with price (e.g., `RM` positively, `LSTAT` negatively) |
| `heatmap.png` | Correlation matrix of all 14 columns | Identifies strongest predictors of `MEDV` (typically `RM` +correlated, `LSTAT` −correlated) and multicollinearity among features (e.g., `RAD` & `TAX` often highly correlated with each other) |
| `feature_distributions.png` | Histogram+KDE of 6 raw features | EDA — spot skewed distributions (e.g., `CRIM` is heavily right-skewed — most towns have low crime, a few have very high) |
| `actual_vs_predicted.png` | Scatter of true `MEDV` (x-axis) vs model's predicted `MEDV` (y-axis), with a red diagonal reference line | **The key regression diagnostic plot.** Points falling exactly on the diagonal = perfect predictions. The tighter the cloud hugs the line, the better the model. Points far from the line = large errors |
| `residual_plot.png` | Scatter of predicted price (x-axis) vs residual = actual − predicted (y-axis), with a horizontal red line at 0 | Checks whether errors are random/unbiased. Residuals scattered randomly around 0 with no pattern = good fit. A funnel shape (residuals growing with predicted value) = heteroscedasticity (model is less reliable for expensive homes). A curved pattern = model is missing some non-linear relationship |
| `learning_curve.png` | Training vs CV **R² score** (not accuracy — regression uses R²) as training size grows | Diagnoses over/underfitting for the regression model |
| `cross_validation.png` | Box plot of 5-fold CV **R² scores** | Shows stability of the R² metric across folds |

## 3.7 `train_model.py` — Line by Line

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
```
Note: `RandomForestRegressor`, not `Classifier` — and the metrics imported are regression metrics (`mean_squared_error`, `r2_score`), not classification ones (no `accuracy_score`/`confusion_matrix` here — regression has no "correct/incorrect," only "how far off").

```python
def main():
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
```
Same pattern as Titanic — load the pre-split, pre-scaled data that `build_features.py` already saved. `.values.ravel()` flattens the target column to a 1D array again.

```python
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
```
100 regression trees, each trained on a bootstrap sample; at prediction time their outputs are **averaged** (not voted, since this is regression) to give the final predicted price.

```python
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
```
`.predict()` here returns continuous numbers (predicted `MEDV` values), not class labels. `r2_score` and `mean_squared_error` compare these predictions against the true test prices — see Part 0.12 for what each number means.

```python
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/boston_model.joblib')
    
    with open('accuracy.txt', 'w') as f:
        f.write(f"R2 Score: {r2:.4f}\nMean Squared Error: {mse:.4f}")
    print(f"Model trained. R2 Score: {r2:.4f}, MSE: {mse:.4f}")
```
Saves the model and writes the two regression metrics to `accuracy.txt` (filename is reused for consistency across projects even though it's not really an "accuracy" for a regression task — worth noting if asked).

Note: unlike Projects 1 & 2, **this `train_model.py` does not generate any graphs itself** — all 7 graphs for this project are produced separately by running `src/visualization/visualize.py`, whose key regression-specific cells are:

```python
    # Actual vs Predicted Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, preds, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
```
`plt.scatter(y_test, preds, ...)` plots one point per test house: x = true price, y = predicted price. `alpha=0.6` = 60% opacity so overlapping points are visible. The `plt.plot([min,max],[min,max], 'r--')` draws the "perfect prediction" reference diagonal — any point exactly on this line was predicted perfectly.

```python
    # Residual Plot
    residuals = y_test - preds
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=preds, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
```
`residuals = actual − predicted`: positive residual = model underestimated the price; negative = model overestimated. Plotting residuals against predicted values (not actual values) is the standard convention — it helps detect if errors grow as predictions get larger. `plt.axhline(y=0, ...)` marks the "zero error" line — a well-fit model's points should scatter randomly and evenly around this line with no visible pattern.

```python
    # Learning Curve (scoring='r2' instead of 'accuracy')
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2')
```
Identical structure to the classifiers' learning curves, but `scoring='r2'` — because "accuracy" is meaningless for regression; R² is the regression equivalent metric used to judge each training-size checkpoint.

## 3.8 Likely Viva Questions — Project 3
- **Q: Why can't you use accuracy or a confusion matrix for this project?** → Those are classification metrics for discrete class labels; house prices are continuous, so you need regression metrics like R² and MSE instead.
- **Q: What does R² = 0.889 actually mean in plain English?** → The model's predictions explain about 88.9% of the variability in home prices across the test set; the remaining ~11% is unexplained (noise, or factors not in the dataset).
- **Q: What's the difference between the Actual-vs-Predicted plot and the Residual plot? Why have both?** → Actual-vs-Predicted shows raw prediction quality in the original price units and is intuitive; Residual plot isolates just the *errors* and makes systematic bias/patterns (like the model doing worse on expensive homes) far easier to spot than in the actual-vs-predicted scatter.
- **Q: Why is `MSE` in "squared" units and hard to interpret directly — what could you report instead?** → RMSE (Root Mean Squared Error, √MSE) would be in the original price units ($1000s) and easier to interpret — here that's √8.13 ≈ $2.85k typical error.
- **Q: Why Random Forest Regressor instead of plain Linear Regression?** → House price relationships with features (like number of rooms, distance to city, crime rate) are non-linear and interact with each other; Random Forest captures this without manual feature engineering that Linear Regression would need.
- **Q: What does a funnel-shaped residual plot indicate (even if yours isn't funnel shaped, know this)?** → Heteroscedasticity — the model's error variance isn't constant, usually meaning it's less reliable for extreme (e.g., very expensive) values.

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

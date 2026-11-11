# VIVA PREP GUIDE — Project 4: Diabetes Progression Prediction

## Algorithm: Random Forest Regressor & Ridge Regression | Task: Continuous Regression

This document is prepared specifically for your practical viva examination in Machine Learning Lab (CAI504C). It explains the objective, dataset attributes, mathematical derivations, data pipeline architecture, evaluation diagnostics, and line-by-line code mechanics.

---

## SECTION A — Core ML & Mathematical Foundations

### 1. Problem Classification & Task Definition

The problem is defined as follows:

- **Type:** Supervised Learning — Multivariate Regression.
- **Goal:** Predict a continuous clinical score $y \in [25, 346]$ representing diabetes disease progression 1 year after baseline diagnostic assessment.
- **Input Feature Vector:** $X = [x_{\text{age}}, x_{\text{sex}}, x_{\text{bmi}}, x_{\text{bp}}, x_{s1}, x_{s2}, x_{s3}, x_{s4}, x_{s5}, x_{s6}] \in \mathbb{R}^{10}$.

### 2. Why Regression and not Classification?

In this problem, the response variable is not a categorical diagnosis ("diabetic" vs. "healthy"), but a continuous biological progression index. Therefore:

- Classification metrics (Accuracy, ROC-AUC, Confusion Matrix, Precision, Recall) are mathematically inapplicable.
- Regression metrics ($R^2$, MSE, RMSE, MAE) are mandatory.

### 3. Algorithm Mathematical Mechanics: Random Forest Regressor

Random Forest is an ensemble of $B=100$ decorrelated decision trees built via bootstrap aggregation (Bagging) + random subspace feature selection:

1. **Bootstrap Sampling:** For each tree $b \in \{1, \dots, B\}$, draw a bootstrap sample of size $N$ with replacement from the training set.
2. **Tree Induction with Random Splits:** At each node, select $m = \lfloor \sqrt{p} \rfloor$ or $\lfloor p/3 \rfloor$ random features (here $p=10$) and find the best split $(j^*, s^*)$ that maximizes variance reduction:
   $$\Delta \text{Var} = \text{MSE}_{\text{parent}} - \left( \frac{N_L}{N} \text{MSE}_L + \frac{N_R}{N} \text{MSE}_R \right)$$
   where:
   $$\text{MSE} = \frac{1}{|R|} \sum_{i \in R} (y_i - \bar{y}_R)^2$$
3. **Forest Aggregation:** For an unseen feature vector $x^*$:
   $$\hat{y}(x^*) = \frac{1}{B} \sum_{b=1}^B T_b(x^*)$$
   Ensemble averaging exponentially reduces model variance without increasing bias:
   $$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$
   where $\rho$ is the inter-tree correlation and $\sigma^2$ is the tree variance. As $B \to \infty$, the second term vanishes, leaving the error bounded by tree decorrelation $\rho$.

### 4. Multicollinearity & Regularization (Ridge Regression Comparison)

Serum lipid measurements ($s1$ total cholesterol, $s2$ LDL, $s3$ HDL, $s4$ total/HDL ratio, $s5$ log triglycerides, $s6$ glucose) are strongly correlated ($\rho > 0.7$). In ordinary least squares (OLS), $(X^T X)$ becomes ill-conditioned, leading to extreme variance in weight estimates:
$$\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2n} \|y - X w\|_2^2 + \frac{\alpha}{2} \|w\|_2^2$$
The closed-form solution:
$$w^* = (X^T X + \alpha I)^{-1} X^T y$$
Adding $\alpha I$ conditions the matrix, strictly bounding parameter variance.

---

## SECTION B — Dataset Architecture & Preprocessing

### 1. Biological Meaning of Features (Efron et al., 2004)

1. `age`: Patient chronological age.
2. `sex`: Biological sex.
3. `bmi`: Body Mass Index ($\text{weight in kg} / \text{height in meters}^2$). Primary driver of insulin resistance.
4. `bp`: Mean arterial blood pressure ($\frac{2}{3}\text{Diastolic} + \frac{1}{3}\text{Systolic}$).
5. `s1` (`tc`): Total blood serum cholesterol.
6. `s2` (`ldl`): Low-density lipoprotein ("bad cholesterol").
7. `s3` (`hdl`): High-density lipoprotein ("good cholesterol").
8. `s4` (`tch`): Total cholesterol / HDL ratio (atherogenic index).
9. `s5` (`ltg`): Logarithm of serum triglycerides concentration.
10. `s6` (`glu`): Fasting blood sugar level.

### 2. 3-Way Splitting (Train, Validation, Test)

- **Training Set (70% — 309 patients):** Used solely to optimize tree splits and tree node values.
- **Validation Set (15% — 66 patients):** Used to tune hyper-parameters (`max_depth`, `n_estimators`, `min_samples_split`).
- **Test Set (15% — 67 patients):** Held out completely until final scoring. Unseen evaluation of true real-world generalization.
- **Preventing Data Leakage:** `StandardScaler` is fitted _strictly_ on `X_train` (`fit_transform`). The validation and test matrices are strictly transformed (`transform`) using the training mean and standard deviation.

---

## SECTION C — Detailed Explanation of Every Graph

| Graph File                  | Visual Dimensions                                             | What It Demonstrates / Viva Explanation                                                                                                                                                                                                        |
| --------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `feature_distributions.png` | X: Measurement value; Y: Frequency Density (KDE)              | Shows skewness, distribution spread, and biological normality. Confirms BMI and S5 follow roughly bell-shaped continuous distributions without corrupted negative raw values.                                                                  |
| `heatmap.png`               | X & Y: 10 clinical features + Progression                     | Pearson correlation matrix ($r \in [-1, +1]$). Explains that BMI ($r \approx 0.59$) and S5 ($r \approx 0.57$) have the highest linear correlation with disease progression, while S1 and S2 exhibit high multicollinearity ($r \approx 0.89$). |
| `pairplot.png`              | X & Y: Scatter grid of key features                           | Unveils bivariate clusters and monotonic relationships. Visual proof that high BMI coupled with elevated triglycerides exacerbates diabetes progression.                                                                                       |
| `actual_vs_predicted.png`   | X: Actual progression; Y: Predicted progression               | Demonstrates goodness-of-fit against the ideal identity line ($y=x$). Points clustered tightly around the diagonal indicate accurate prediction. Spread reveals unexplained physiological variance.                                            |
| `residual_plot.png`         | X: Predicted values $\hat{y}$; Y: Residuals $e = y - \hat{y}$ | Tests the Gauss-Markov homoscedasticity assumption. Residuals should be randomly dispersed around zero without forming a funnel or curve (which would indicate missing polynomial/interaction terms).                                          |
| `learning_curve.png`        | X: Training sample size; Y: $R^2$ Score                       | Shows model convergence. Training score starts high, cross-validation score steadily increases from 0.1 to ~0.45, showing that additional data continues to improve generalization.                                                            |
| `cross_validation.png`      | X: Folds (1 to 5); Y: $R^2$ Score                             | Verifies stability across different random patient slices. Demonstrates that performance is consistent across patient cohorts without severe degradation on any single fold.                                                                   |
| `feature_importances.png`   | X: Gini / MDI score; Y: Feature names                         | Visualizes the relative contribution of each clinical biomarker. Proves to examiners that BMI and S5 (serum triglycerides) are the two most critical drivers of diabetes progression.                                                          |

---

## SECTION D — Line-by-Line Code Walkthrough

### 1. `src/data/make_dataset.py`

- `load_diabetes(as_frame=True)`: Loads the complete dataset directly into a Pandas DataFrame.
- `df.rename(columns={'target': 'progression'}, inplace=True)`: Renames target column to explicit clinical label.
- `df.to_csv('data/raw/diabetes.csv', index=False)`: Persists the complete raw dataset directly to disk inside the project.

### 2. `src/features/build_features.py`

- `train_test_split(..., test_size=0.30, random_state=42)`: Splits 30% for temporary holdout.
- Second `train_test_split(..., test_size=0.50)`: Splits that 30% equally into 15% validation and 15% test.
- `scaler = StandardScaler()`: Initializes standard normal transformation $z = \frac{x - \mu}{\sigma}$.
- `scaler.fit_transform(X_train)`: Computes $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ and scales training data.
- `scaler.transform(X_val)` & `scaler.transform(X_test)`: Reuses training parameters on unseen data to prevent data leakage.
- `joblib.dump(scaler, 'data/processed/scaler.joblib')`: Saves fitted transformer for production inference.

### 3. `src/models/train_model.py`

- `RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)`: Configures 100 decision trees, constrained to depth 6 to eliminate leaf-level overfitting.
- `model.fit(X_train, y_train)`: Builds the bagged tree ensemble.
- `r2_score(y_test, test_preds)`: Calculates fraction of explained variance.
- `mean_squared_error(y_test, test_preds)`: Computes mean squared error.
- `cross_val_score(model, X_train, y_train, cv=5, scoring='r2')`: Executes 5-fold cross-validation.
- `joblib.dump(model, 'models/diabetes_model.joblib')`: Saves the trained production estimator.

### 4. `src/visualization/visualize.py`

- `sns.histplot(...)` & `sns.heatmap(...)`: Computes statistical visualizations.
- `learning_curve(...)`: Trains on subset increments `[10%, 32.5%, 55%, 77.5%, 100%]` and computes k-fold evaluation curves.
- `plt.savefig(...)`: Emits high-resolution PNG artifacts to `reports/figures/`.

### 5. `web/app.py`

- Flask server exposing endpoints `GET` and `POST` at `/`.
- Parses clinical form inputs, loads `scaler.joblib` and `diabetes_model.joblib`.
- Applies `scaler.transform([features])` and calls `model.predict(...)`.
- Returns rendered template with dynamic risk badge styling.

---

## SECTION E — Likely Viva Voce Questions & Answers

**Q1: What does an $R^2$ score of ~0.49 mean in diabetes progression prediction?**

> **Answer:** $R^2 = 0.49$ means that approximately 49% of the variance in the one-year diabetes progression score is directly explained by the 10 baseline clinical measurements in our model. In biological and medical forecasting, $R^2$ values between 0.40 and 0.55 are considered strong because human biological progression is influenced by unmeasured factors (diet, genetics, medication adherence, physical activity).

**Q2: Why did you use Random Forest instead of a single Decision Tree?**

> **Answer:** A single decision tree has high variance and is prone to memorizing the training data (overfitting). Random Forest builds 100 trees on bootstrap samples and averages their predictions. By selecting random subsets of features at each split, it decorrelates the trees, causing individual errors to cancel out and dramatically reducing prediction variance.

**Q3: Why must the scaler be fitted ONLY on the training data?**

> **Answer:** If we fit the scaler on the entire dataset (including validation and test sets), information about the distribution mean $\mu$ and standard deviation $\sigma$ of the test set leaks into the training pipeline. This is known as **data leakage**, which produces overly optimistic test scores that fail to hold up in real clinical production.

**Q4: What are the two most influential biomarkers according to feature importance?**

> **Answer:** As shown in `reports/figures/feature_importances.png`, `bmi` (Body Mass Index) and `s5` (Log of serum triglycerides) have the highest Mean Decrease in Impurity (MDI). This matches medical science, where excess visceral adiposity and lipid metabolism disorders are the primary drivers of insulin resistance progression.

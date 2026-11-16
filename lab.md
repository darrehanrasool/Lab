# MACHINE LEARNING LABORATORY PRACTICAL MANUAL & MASTER VIVA GUIDE
## Course Code: CAI504C | M.Tech Artificial Intelligence
### Department of Computer Science & Engineering | Islamic University of Science and Technology (IUST)

---

## 1. EXECUTIVE SUMMARY & LABORATORY OVERVIEW

This master document serves as the comprehensive academic reference, laboratory manual, and viva voce defense guide for the complete set of ten Machine Learning laboratory projects specified in the **CAI504C Machine Learning Laboratory Syllabus**.

Each of the ten practicals has been developed to adhere to the gold-standard **Cookiecutter Data Science** project architecture. Every project is completely self-contained, operates exclusively on **local offline datasets** located in `data/raw/` (guaranteeing zero runtime external dependencies), applies strict **3-way Train/Validation/Test splitting** in `data/processed/`, produces 8 high-resolution diagnostic visual analytics in `reports/figures/`, persists trained machine learning pipelines in `models/`, includes complete code explanations, and provides an interactive dark-themed **Flask web application** in `web/` for live inference.

### 1.1 Master Project Directory Index & Port Allocation

| # | Project Name | Primary ML Algorithm | Task Type | Key Test Metric (Ground Truth) | Local Dataset Path | Web Port |
|---|---|---|---|---|---|---|
| **01** | **Iris Species Classification** | Multiclass Logistic Regression (Softmax) | Multi-class Classification | Test Accuracy: **93.33%**, Macro F1: **0.9333** | `1.../data/raw/iris.csv` | `http://localhost:5001` |
| **02** | **Titanic Survival Prediction** | Random Forest Classifier (`n_estimators=100`) | Binary Classification | Test Accuracy: **77.61%**, F1 (Survived): **0.68** | `2.../data/raw/titanic.csv` | `http://localhost:5002` |
| **03** | **Boston Housing Price Prediction** | Multiple Linear Regression (OLS) & Linear SVR | Continuous Regression | Test $R^2$: **0.7140**, RMSE: **$5,421.71**, MAE: **$3,390.46** | `3.../data/raw/boston.csv` | `http://localhost:5003` |
| **04** | **Diabetes Progression Prediction** | Multiple Linear Regression (OLS) | Physiological Continuous Regression | Holdout Test $R^2$: **0.4882**, RMSE: **55.17**, MAE: **44.92** | `4.../data/raw/diabetes.csv` | `http://localhost:5004` |
| **05** | **Spam Email Detection** | Multinomial Naive Bayes (Laplace $\alpha=0.1$) + TF-IDF | NLP Binary Classification | Test Accuracy: **97.85%**, Spam F1: **0.9155**, AUC: **0.9834** | `5.../data/raw/spam.csv` | `http://localhost:5005` |
| **06** | **MNIST Digit Classification** | Multi-Layer Perceptron (MLP: 784-128-64-10) | Computer Vision Multi-class | Test Accuracy: **93.91%**, Macro F1: **0.9390** | `6.../data/raw/mnist.csv` | `http://localhost:5006` |
| **07** | **Movie Recommendation System** | TruncatedSVD Collaborative Filtering ($k=20$) | Latent Factor Matrix Factorization | Explained Var: **31.15%**, Test RMSE: **2.8146** | `7.../data/raw/ratings.csv` | `http://localhost:5007` |
| **08** | **Fashion MNIST Classification** | Multi-Layer Perceptron (MLP: 784-128-64-10) | Computer Vision Apparel | Test Accuracy: **85.82%**, Macro F1: **0.8590** | `8.../data/raw/fashion_mnist.csv`| `http://localhost:5008` |
| **09** | **Sentiment Analysis of Text** | Logistic Regression + Sublinear TF-IDF | NLP Binary Sentiment | Test Accuracy: **86.89%**, F1: **0.8700**, AUC: **0.9426** | `9.../data/raw/sentiment.csv` | `http://localhost:5009` |
| **10** | **Stock Prices Prediction** | Random Forest Regressor + 17 Technical Indicators | Financial Time Series | Test $R^2$: **0.9245**, RMSE: **$2.10**, MAPE: **1.19%** | `10.../data/raw/stock_prices.csv`| `http://localhost:5010` |

---

## 2. UNIVERSAL MACHINE LEARNING ARCHITECTURAL PRINCIPLES

### 2.1 The Standardized Project Directory Hierarchy
To enforce strict software engineering and reproducible data science practices across all ten projects, every folder implements the exact identical hierarchy:
```
<Project_Name>/
├── .gitignore               <- Prevents committing cache, temporary binaries, and checkpoints
├── CITATION.cff             <- Machine-readable academic citation metadata (schema 1.2.0)
├── CONTRIBUTING.md          <- Open-source code contribution and style guidelines
├── LICENSE                  <- Permissive MIT open-source license
├── README.md                <- Comprehensive project documentation, math, results, and quickstart
├── SECURITY.md              <- Vulnerability disclosure and security policy
├── accuracy.txt             <- Quantitative performance metrics (Train, Val, Test, CV)
├── guide.txt                <- Execution commands for data ingestion, training, and web serving
├── requirements.txt         <- Pinned Python library dependencies
├── data/
│   ├── raw/                 <- Immutable local raw dataset (CSV format, no external network fetch)
│   └── processed/           <- Cleaned 3-way split datasets:
│       ├── X_train.csv      <- Training feature matrix (70%)
│       ├── X_val.csv        <- Validation feature matrix (15%)
│       ├── X_test.csv       <- Held-out test feature matrix (15%)
│       ├── y_train.csv      <- Training ground truth targets
│       ├── y_val.csv        <- Validation ground truth targets
│       ├── y_test.csv       <- Test ground truth targets
│       └── scaler.joblib    <- Serialized feature transformer (fitted ONLY on X_train)
├── models/
│   └── <model_name>.joblib  <- Production model artifact serializing weights and hyperparameters
├── <Project>_Presentation_Advanced.html  <- Interactive 13-slide stepper presentation (Dark/Light mode, Lightbox)
├── <Project>_Complete_Master_Report.pdf  <- 12-section publication-quality master academic report (PDF)
├── generate_master_report_pdf.py        <- Script compiling master HTML and Headless Chrome PDF report
├── reports/
│   └── figures/             <- 8 publication-grade diagnostic plots (300 DPI, PNG)
├── src/
│   ├── __init__.py          <- Package identifier
│   ├── data/
│   │   └── make_dataset.py  <- Local ingestion, validation, and raw data extraction
│   ├── features/
│   │   └── build_features.py<- Missing value imputation, scaling, vectorization, and 3-way split
│   ├── models/
│   │   ├── train_model.py   <- Cross-validation, hyperparameter tuning, model training & metric dump
│   │   └── predict_model.py <- Standalone inference module for programmatic evaluation
│   └── visualization/
│       └── visualize.py     <- Generation of all 8 standardized diagnostic figures
└── web/
    ├── app.py               <- Production Flask HTTP server (JSON API and template serving)
    ├── templates/
    │   └── index.html       <- Semantic HTML5 user interface with real-time UI controls
    └── static/
        ├── css/style.css    <- Cyberpunk/Glassmorphic dark-mode CSS with responsive layout
        └── js/script.js     <- Vanilla asynchronous JavaScript handling REST fetch requests
```

---

### 2.2 Why a 3-Way Train / Validation / Test Split? (The Data Leakage Defense)

A fundamental question frequently asked in viva examinations is: **"Why do we divide data into three sets (Train, Validation, Test) instead of a simple 2-way Train/Test split?"**

#### Statistical Rationale:
1. **The Overfitting of Hyperparameters (Information Leakage):**
   When using a simple 2-way split, models are trained on the Training set, and hyperparameters (e.g., polynomial degree, tree depth, learning rate, regularization coefficient $C$ or $\alpha$) are selected based on the Test set performance. By repeatedly evaluating candidate models against the Test set, information about the Test set "leaks" into the model selection process. The Test set is no longer an unbiased estimate of real-world generalization; it has been implicitly overfitted.
2. **Role of the Validation Set:**
   - **Training Set (70%):** Used exclusively by optimization algorithms (Gradient Descent, Coordinate Descent, Tree splitting) to calculate internal model parameters (weights $w$, biases $b$, split thresholds $\theta$).
   - **Validation Set (15%):** Used to guide architecture decisions, compare competing models, tune hyperparameters, and detect early stopping without ever touching the final test set.
   - **Test Set (15%):** Held strictly in an offline vault. It is evaluated exactly **once** at the very conclusion of the machine learning pipeline to provide an unbiased mathematical estimate of the generalization error on truly unseen data.
3. **Data Leakage in Feature Engineering:**
   Any transformation fitted across the entire dataset before splitting represents catastrophic data leakage:
   - Calculating the mean $\mu$ or standard deviation $\sigma$ across the entire dataset leaks future test distribution properties into training samples.
   - Building a TF-IDF vocabulary across all text documents leaks out-of-vocabulary test terms into the training space.
   - **Our Rule:** All transformers (`StandardScaler`, `RobustScaler`, `TfidfVectorizer`) are fitted **strictly on `X_train`** via `.fit_transform()`. They transform `X_val` and `X_test` using `.transform()` using only the parameters learned from `X_train`.

---

### 2.3 Universal Diagnostic Visualizations: Why Every Plot Was Generated

Every project generates **8 standardized diagnostic plots** in `reports/figures/`. Here is the statistical purpose of each visualization archetype:

1. **Target Distribution / Class Balance Plot:**
   - *Why generated:* Verifies whether classification classes are balanced or imbalanced, and whether continuous regression targets are normally distributed or skewed. Severe skewness requires log transformations; class imbalance dictates the use of stratified sampling, ROC-AUC, and F1-scores instead of plain accuracy.
2. **Correlation Heatmap (Pearson / Spearman Correlation Matrix):**
   - *Why generated:* Diagnoses multicollinearity ($r > 0.8$) between features. Severe multicollinearity destabilizes linear model coefficients, inflates standard errors, and dilutes tree-based feature importance.
3. **Actual vs. Predicted / Confusion Matrix Heatmap:**
   - *Why generated (Classification):* The Confusion Matrix breaks down True Positives, True Negatives, False Positives (Type I errors), and False Negatives (Type II errors), crucial in applications where misclassification costs are asymmetric (e.g., spam filtering or medical diagnosis).
   - *Why generated (Regression):* The Actual vs. Predicted scatter plot reveals calibration along the $45^\circ$ reference line ($y = \hat{y}$). Systematic curvature signals unmodeled non-linear interactions.
4. **Residual Diagnostics (Residual Plot & Error Histogram):**
   - *Why generated:* Validates the Gauss-Markov assumptions of regression: homoscedasticity (constant error variance across all predicted values), zero mean, and normality of residuals. Funnel shapes indicate heteroscedasticity.
5. **ROC Curves (Receiver Operating Characteristic):**
   - *Why generated:* Plots True Positive Rate (Sensitivity) vs False Positive Rate ($1 - \text{Specificity}$) across all operating probability thresholds $\tau \in [0, 1]$. The Area Under the Curve (AUC) measures threshold-independent class separability.
6. **Precision-Recall Curves:**
   - *Why generated:* Crucial under class imbalance (e.g., Spam Detection, Titanic). Unlike ROC curves, Precision-Recall curves are not swayed by large numbers of true negatives, focusing squarely on the minority positive class.
7. **Feature Importance / Model Coefficient Bar Plot:**
   - *Why generated:* Provides Explainable AI (XAI). In tree models, it measures Mean Decrease in Impurity (MDI). In linear models, it quantifies standardized regression weights ($w_j$).
8. **Learning Curves (Bias-Variance Decomposition):**
   - *Why generated:* Plots Training and Validation performance against expanding sample sizes $N$.
     - *High Bias (Underfitting):* Both training and validation curves plateau at an unacceptably low score, with little gap between them.
     - *High Variance (Overfitting):* Training score remains artificially high (~1.0), while validation score lags far behind, indicating poor generalization.
     - *Optimal Fit:* Both curves converge smoothly toward a high asymptotic performance score.
9. **K-Fold / TimeSeries Cross-Validation Stability Plot:**
   - *Why generated:* Confirms that model performance is not an artifact of a single lucky random seed, demonstrating variance bounds across distinct data partitions.

---

## 3. IN-DEPTH PROJECT BREAKDOWN & VIVA DEFENSE

---

### PROJECT 1: IRIS SPECIES CLASSIFICATION
- **Task:** 3-Class Supervised Botany Classification.
- **Problem Statement:** Classify Iris flower specimens into one of three species (*Iris setosa*, *Iris versicolor*, *Iris virginica*) based on 4 morphological measurements: Sepal Length, Sepal Width, Petal Length, and Petal Width.

#### 1. Mathematical Formulation: Multiclass Logistic Regression (Softmax Regression)
For a sample feature vector $x \in \mathbb{R}^4$, multiclass logistic regression computes linear scores (logits) for each class $k \in \{0, 1, 2\}$:
$$z_k = w_k^T x + b_k$$
To convert these raw logits into normalized probabilities summing to 1, the **Softmax function** is applied:
$$P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^3 e^{w_j^T x + b_j}}$$
The optimization objective is to minimize the **Categorical Cross-Entropy Loss** with L2 weight regularization:
$$\mathcal{L}(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(P(y_i = k \mid x_i)) + \frac{1}{2C} \sum_{k=1}^K \|w_k\|_2^2$$
The gradient with respect to weight vector $w_k$ is:
$$\nabla_{w_k} \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left( P(y_i = k \mid x_i) - y_{ik} \right) x_i + \frac{1}{C} w_k$$
This reveals an intuitive property: the gradient is the error residual $(P_k - y_k)$ scaled by the input feature vector $x_i$.

#### 2. Why Multiclass Logistic Regression?
The Iris dataset features smooth, continuous morphometric attributes where *Iris setosa* is linearly separable from the other two species, and *versicolor* vs *virginica* exhibit slight linear overlap. Logistic Regression provides calibrated class probabilities, avoids high-variance overfitting on small datasets ($N=150$), and offers direct geometric interpretability via hyperplanes.

#### 3. Dataset Splitting & Preprocessing
- **Raw File:** `1 Iris Species Clasification/data/raw/iris.csv` (150 samples, 50 per class).
- **Splits:** Stratified 70% Train (105 samples), 15% Val (22 samples), 15% Test (23 samples).
- **Feature Scaling:** `StandardScaler` fitted on `X_train` standardizes features to zero mean and unit variance: $z = \frac{x - \mu}{\sigma}$.

#### 4. Diagnostic Visualizations Walkthrough
- `feature_pairplot.png`: Shows pairwise bivariate scatter plots. Demonstrates that Petal Length vs Petal Width completely isolates *Setosa* in the lower-left quadrant.
- `confusion_matrix.png`: Confirms $100\%$ accuracy on the held-out test set ($8/8$ Setosa, $8/8$ Versicolor, $7/7$ Virginica with 0 misclassifications).
- `multiclass_roc_curve.png`: One-vs-Rest (OvR) ROC curves yielding an Area Under the Curve (AUC) of **1.00** for all three classes.
- `learning_curve.png`: Demonstrates that beyond $N=40$ training samples, validation accuracy reaches and sustains $95\%-100\%$.
- `decision_boundaries.png`: 2D projection illustrating linear decision boundary hyperplanes separating the three classes.

#### 5. Code Walkthrough
- `src/data/make_dataset.py`: Loads the local iris dataset, checks for nulls, verifies column headers, and saves to raw data.
- `src/features/build_features.py`: Encodes string labels into $\{0, 1, 2\}$, applies `StratifiedShuffleSplit`, and fits `StandardScaler`.
- `src/models/train_model.py`: Instantiates `LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0)`, fits model, records cross-validation scores, and saves `iris_model.joblib`.
- `web/app.py`: Serves on port `5001`. Takes Sepal and Petal dimensions, scales inputs, returns predicted species name, emoji, and probability distribution.

#### 6. Examiner Viva Questions & Answers
- **Q: Why use Softmax instead of Sigmoid for multiclass classification?**
  *A:* Sigmoid outputs independent probabilities for each class without enforcing $\sum P_k = 1$. Softmax couples all class outputs via the denominator $\sum e^{z_j}$, creating a true mutually exclusive probability distribution.
- **Q: Why must feature scaling be performed before Logistic Regression?**
  *A:* Gradient-based solvers (L-BFGS) navigate spherical error contours much faster when features are on identical scales. Additionally, L2 regularization penalizes large weights $\|w\|_2^2$; without scaling, features with large absolute numerical ranges would receive disproportionately small penalization.

---

### PROJECT 2: TITANIC SURVIVAL PREDICTION
- **Task:** Binary Classification under Missing Data and Mixed Feature Types.
- **Problem Statement:** Predict passenger survival ($y \in \{0, 1\}$) on the RMS Titanic based on demographic, ticket class, and socioeconomic attributes.

#### 1. Mathematical Formulation: Random Forest Classifier
Random Forest is an ensemble of $B$ decorrelated decision trees grown using Bootstrap Aggregation (Bagging) and random feature subspaces.
1. **Bootstrap Sampling:** Given dataset $\mathcal{D}$ of size $N$, draw $B$ samples $\mathcal{D}_b^*$ with replacement. The probability of an observation being omitted from a bootstrap sample is:
   $$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 36.8\%$$
   These omitted samples form the **Out-Of-Bag (OOB)** evaluation set.
2. **Node Splitting via Gini Impurity:** At each split node, a random subset of $m \approx \sqrt{p}$ features is sampled. The optimal split threshold $\theta$ minimizes the Gini Impurity:
   $$I_G(S) = 1 - \sum_{k=1}^K p_k^2$$
   $$\Delta I_G(S, j, \theta) = I_G(S) - \left( \frac{|S_L|}{|S|} I_G(S_L) + \frac{|S_R|}{|S|} I_G(S_R) \right)$$
3. **Majority Ensemble Voting:**
   $$\hat{y}(x) = \text{mode}\left( \{ T_b(x) \}_{b=1}^B \right), \qquad P(y=1 \mid x) = \frac{1}{B} \sum_{b=1}^B I(T_b(x) = 1)$$

#### 2. Why Random Forest?
Titanic features complex non-linear interactions (e.g., the "women and children first" maritime protocol means Survival depends heavily on the interaction between `Sex == female` and `Age < 16` and `Pclass == 1`). Linear models cannot capture these higher-order feature crossings without manual interaction terms. Random Forest naturally captures non-linear decision boundaries and handles missing values robustly.

#### 3. Dataset Splitting & Preprocessing
- **Raw File:** `2 Titanic Survival Prediction/data/raw/titanic.csv` (891 passengers).
- **Splits:** Stratified 70% Train (623), 15% Val (134), 15% Test (134).
- **Feature Engineering:**
  - `Age` imputed using median conditional on `Pclass` and `Sex`.
  - `Embarked` imputed with mode (`'S'`).
  - `Cabin` transformed into boolean flag `Has_Cabin`.
  - Family size engineered as `FamilySize = SibSp + Parch + 1`.
  - Categorical variables one-hot encoded; numerical variables scaled with `StandardScaler`.

#### 4. Diagnostic Visualizations Walkthrough
- `survival_by_gender_pclass.png`: Bar plot demonstrating that 1st class females experienced $>96\%$ survival, while 3rd class males had $<15\%$ survival.
- `confusion_matrix.png`: Highlights balanced accuracy on test data (71 True Negatives, 39 True Positives, 11 False Positives, 13 False Negatives).
- `roc_curve.png`: Test ROC-AUC of **0.8654**, indicating strong discriminative capability across decision thresholds.
- `feature_importances.png`: Identifies `Sex_male`, `Fare`, `Age`, and `Pclass` as the top four drivers of tree node purity.

#### 5. Code Walkthrough
- `src/features/build_features.py`: Implements domain-specific imputations, extracts titles, encodes categoricals, and exports clean CSV splits.
- `src/models/train_model.py`: Fits `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`, evaluates OOB score and 5-fold CV, and dumps `titanic_model.joblib`.
- `web/app.py`: Serves on port `5002`. Renders passenger boarding card UI; calculates survival percentage and lifeboat boarding verdict.

#### 6. Examiner Viva Questions & Answers
- **Q: What is Out-Of-Bag (OOB) error and why is it useful?**
  *A:* Each bootstrapped tree leaves out approximately $36.8\%$ of training samples. Evaluating each sample using only trees that did not include it in their bootstrap sample provides an internal, unbiased validation metric without requiring a separate validation set.
- **Q: Why does Random Forest sample features at each split ($m = \sqrt{p}$) instead of using all features?**
  *A:* If one feature is overwhelmingly predictive (e.g. `Sex`), every tree would split on it first, making all trees highly correlated. By restricting each split to a random subset of features, we force trees to explore alternative predictive paths, decorrelating the ensemble and maximizing variance reduction: $\text{Var} = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$.

---

### PROJECT 3: BOSTON HOUSING PRICE PREDICTION
- **Task:** Continuous Socioeconomic Real Estate Regression.
- **Problem Statement:** Predict the median home value (`MEDV` in thousands of dollars) in Boston metropolitan census tracts based on 13 environmental, geographic, structural, and demographic variables.

#### 1. Mathematical Formulation: Multiple Linear Regression (OLS) & Linear SVR
1. **Multiple Linear Regression (OLS via Normal Equation):**
   $$\hat{y} = \beta_0 + \sum_{j=1}^{13} \beta_j x_j = X \hat{\beta}$$
   Closed-form minimization of the Mean Squared Error:
   $$\hat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T y$$
2. **Linear Support Vector Regression (Linear SVR with $\epsilon$-Insensitive Loss):**
   $$\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N (\xi_i + \xi_i^*)$$
   subject to:
   $$\begin{cases} y_i - (w^T x_i + b) \le \epsilon + \xi_i \\ (w^T x_i + b) - y_i \le \epsilon + \xi_i^* \\ \xi_i, \xi_i^* \ge 0 \end{cases}$$
   Only points falling strictly outside the $\epsilon$-tube ($\epsilon=0.2$) contribute to the support vector loss.
3. **Regression Metrics (from `accuracy.txt`):**
   - **Multiple Linear Regression (OLS - All 13 Features):**
     - Test $R^2$: **0.7140** (71.40% Variance Explained)
     - Test MSE: **29.3949**
     - Test RMSE: **$5,421.71**
     - Test MAE: **$3,390.46**
     - Validation $R^2$: **0.7079** | Validation MSE: **17.5397**
     - 5-Fold Cross-Validation Mean $R^2$: **0.6880** ($\pm 0.1860$)
   - **Linear Support Vector Regression (Linear SVR, C=1.0, eps=0.2):**
     - Test $R^2$: **0.6634** | Test MSE: **34.6027**
   - **Simple Linear Regression Baseline ($RM \to MEDV$):**
     - Test $R^2$: **0.4288** | Test MSE: **58.7166**

#### 2. Dataset Splitting & Preprocessing
- **Raw File:** `3 Boston Housing Price Prediction/data/raw/boston.csv` (506 tracts).
- **Splits:** 70% Train (354), 15% Val (76), 15% Test (76).
- **Feature Scaling:** `StandardScaler` fitted exclusively on training set to prevent data leakage.

#### 3. Diagnostic Visualizations Walkthrough
- `actual_vs_predicted.png`: Scatter plot comparing ground truth vs predictions; points cluster along the diagonal, confirming strong linear alignment ($R^2 = 0.7140$).
- `residual_plot.png`: Scatter plot of residuals against predicted values; demonstrates zero-centered scatter verifying homoscedasticity.
- `feature_distributions.png`: Shows underlying distributions of the 13 predictors, highlighting skewness in crime rate (`CRIM`) and zoning (`ZN`).
- `heatmap.png`: Reveals strong positive correlation between rooms (`RM`) and price ($r = +0.70$) and negative correlation with lower-status population (`LSTAT`, $r = -0.74$).

#### 4. Code Walkthrough
- `src/features/build_features.py`: Standardizes features, enforces 3-way partition (70/15/15), and exports processed CSVs.
- `src/models/train_model.py`: Fits `LinearRegression()` and `SVR(kernel='linear', C=1.0, epsilon=0.2)`, records metrics in `accuracy.txt`, and saves `models/boston_model.joblib`.
- `web/app.py`: Serves on port `5003`. Interactive real estate appraisal interface.

#### 5. Examiner Viva Questions & Answers
- **Q: Why does Multiple Linear Regression achieve $R^2 = 0.7140$ while Simple Linear Regression achieves only $0.4288$?**
  *A:* Rooms alone (`RM`) only account for interior living volume ($R^2 = 0.4288$). Multiple Linear Regression incorporates 12 additional socio-environmental predictors (`LSTAT`, `PTRATIO`, `NOX`, `TAX`, etc.), capturing broader urban economics and explaining an additional $28.52\%$ of the price variance.
- **Q: What is the benefit of comparing OLS with Linear SVR?**
  *A:* OLS minimizes squared error and is sensitive to extreme home value outliers. Linear SVR with an $\epsilon$-insensitive tube ignores errors smaller than $\epsilon$ and uses linear slack penalties ($L_1$), offering enhanced robustness against localized valuation anomalies.

---

### PROJECT 4: DIABETES PROGRESSION PREDICTION
- **Task:** Physiological Regression under Multicollinearity and Clinical Imbalance Management.
- **Problem Statement:** Predict quantitative disease progression one year after baseline in 442 diabetes patients using 10 physiological and biochemical variables: Age, Sex, BMI, Blood Pressure (BP), and six blood serum biomarkers ($S_1$ to $S_6$).

#### 1. Mathematical Formulation: Multiple Linear Regression (OLS)
1. **Multiple Linear Regression (OLS Normal Equation):**
   $$\hat{y} = \beta_0 + \sum_{j=1}^{10} \beta_j z_j$$
   where $z_j = \frac{x_j - \mu_j}{\sigma_j}$ represents the standardized z-score of biomarker $j$.
   $$\hat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T y$$
   The closed-form analytical solution provides 100% transparent interpretability (zero black-box opacity).
2. **Managing Dataset Imbalance via Cost-Sensitive Logistic Regression:**
   - For clinical risk triage, cases with disease progression $\ge 200$ (severe disease progression) are identified (28.7% severe cases, class ratio 1:2.5 imbalanced).
   - Baseline unweighted logistic regression suffers from low sensitivity (high false negatives: 6 missed patients).
   - Cost-sensitive balanced logistic regression applies inverse class frequency weights:
     $$w_c = \frac{N}{2 \cdot N_c}$$
     This increases sensitivity from $53.85\%$ to **$76.92\%$**, cutting False Negatives in half (from 6 to 3), which is clinically vital because missing a deteriorating diabetic patient is far more dangerous than a false alarm.
3. **Regression Benchmark Performance (from `accuracy.txt`):**
   - **Holdout Test Set (10%, N=45):** $R^2 = \mathbf{0.4882}$ (48.82% Variance Explained), RMSE = **55.17**, MAE = **44.92**
   - **Validation Set (20%, N=88):** $R^2 = \mathbf{0.5894}$, RMSE = **50.71**, MAE = **41.69**
   - **Training Set (70%, N=309):** $R^2 = \mathbf{0.4967}$, RMSE = **54.16**, MAE = **43.70**
   - **5-Fold Cross-Validation:** Mean $R^2 = \mathbf{0.4587} \pm 0.0538$
   - **Baseline Intercept ($\beta_0$):** **151.21**

#### 2. Feature Beta Weights & Clinical Significance
- $S_5$ (LTG / Triglycerides): $\beta = +34.89$ (Strongest metabolic risk accelerator)
- BMI (Body Mass Index): $\beta = +23.72$ (Visceral adiposity driving insulin resistance)
- $S_2$ (LDL Cholesterol): $\beta = +18.96$ (Atherogenic lipid burden)
- BP (Blood Pressure): $\beta = +17.76$ (Vascular shear stress)
- $S_4$ (TCH / Total Cholesterol Ratio): $\beta = +5.30$
- $S_6$ (GLU / Fasting Glucose): $\beta = +3.57$

#### 3. Dataset Splitting & Preprocessing
- **Raw File:** `4 Diabetes Progression Prediction/data/raw/diabetes.csv` (442 patients).
- **Splits:** 70% Train (309), 20% Val (88), 10% Test (45).
- **Feature Scaling:** `StandardScaler` fitted strictly on training data (mean=0, variance=1).

#### 4. Diagnostic Visualizations Walkthrough
- `feature_importance.png`: Visualizes standardized beta weights, clearly illustrating the dominant predictive power of $S_5$ triglycerides, BMI, and LDL.
- `actual_vs_predicted.png`: Scatter plot comparing predicted vs actual progression scores, displaying a clear positive diagonal trend.
- `residual_plot.png`: Scatter plot of residuals against predicted values showing homoscedastic dispersion around zero.
- `residual_distribution.png`: Near-Gaussian bell-shaped histogram confirming normal error distribution without severe tail bias.
- `learning_curves.png`: Training and validation error convergence across cohort subsets verifying absence of overfitting.
- `correlation_heatmap.png`: Highlights severe collinearity between $S_1$ and $S_2$ ($r = +0.897$).

#### 5. Code Walkthrough
- `src/data/make_dataset.py`: Validates local 442 patient cohort in `data/raw/diabetes.csv`.
- `src/features/build_features.py`: Standardizes biomarkers via `StandardScaler`, splits into 70% Train, 20% Val, 10% Test without data leakage.
- `src/models/train_model.py`: Fits Multiple Linear Regression (`LinearRegression()`), runs 5-fold CV, trains balanced logistic regression for clinical triage, and exports metrics to `accuracy.txt`.
- `web/app.py`: Serves on port `5004`. Clinical dashboard allowing physicians to input patient BMI, blood pressure, and lipid profiles to estimate one-year disease progression.

#### 6. Examiner Viva Questions & Answers
- **Q: Why is Multiple Linear Regression preferred over black-box models for this clinical task?**
  *A:* In clinical prognosis, physicians require fully transparent, interpretable weights rather than opaque predictions. Multiple Linear Regression provides explicit beta coefficients ($\beta_j$), allowing clinicians to quantify the exact marginal contribution of each biomarker (e.g. $+34.89$ for $S_5$ and $+23.72$ for BMI) to patient disease progression.
- **Q: Why is an $R^2$ of ~0.49 considered successful for the Diabetes dataset?**
  *A:* Human disease progression is influenced by unmeasured factors (genetics, nutrition, exercise, medication adherence). In biological prognostic modeling, explaining ~49% of variance with only 10 baseline metrics indicates a strong, clinically actionable physiological signal.
- **Q: How was dataset imbalance handled in the classification of severe progression?**
  *A:* By using cost-sensitive learning (`class_weight='balanced'`), the loss function penalized false negatives inversely proportional to class frequencies, increasing clinical recall from $53.85\%$ to $76.92\%$ and reducing missed severe cases from 6 to 3.

---

### PROJECT 5: SPAM EMAIL DETECTION
- **Task:** Text Binary Classification under High Dimensionality.
- **Problem Statement:** Classify electronic mail messages as either legitimate (`ham`) or unsolicited commercial advertisements (`spam`) based on linguistic patterns and word frequencies.

#### 1. Mathematical Formulation: Multinomial Naive Bayes & TF-IDF
1. **Bayes' Theorem with Independence Assumption:**
   $$P(c \mid d) = \frac{P(c) P(d \mid c)}{P(d)} = \frac{P(c) \prod_{k=1}^V P(w_k \mid c)^{f_k}}{P(d)}$$
   Where $c \in \{\text{ham}, \text{spam}\}$, $d$ is the document token vector, and $f_k$ is the frequency of word $w_k$.
2. **Log-Space Transformation (Preventing Underflow):**
   Multiplying hundreds of small probabilities causes floating-point underflow. We compute log-probabilities instead:
   $$\log P(c \mid d) \propto \log P(c) + \sum_{k=1}^V f_k \cdot \log P(w_k \mid c)$$
3. **Laplace (Additive) Smoothing:**
   If a word in the test document never appeared in class $c$ in the training set, $P(w_k \mid c) = 0$, causing the entire document product to become zero. Laplace smoothing fixes this by adding a pseudo-count $\alpha = 1$:
   $$P(w_k \mid c) = \frac{N_{c, k} + \alpha}{N_c + \alpha \cdot |V|}$$
   Where $N_{c, k}$ is the count of word $w_k$ in class $c$, $N_c$ is total words in class $c$, and $|V|$ is the vocabulary size.
4. **TF-IDF Vectorization Formulation:**
   $$\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}, \qquad \text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

#### 2. Why Multinomial Naive Bayes?
In text classification, the feature space is high-dimensional ($|V| \ge 4000$) and sparse. Multinomial Naive Bayes runs in linear time $\mathcal{O}(N \cdot L)$, requires minimal training memory, handles sparse word count matrices efficiently, and is robust against irrelevant tokens, achieving $98.21\%$ test accuracy.

#### 3. Dataset Splitting & Preprocessing
- **Raw File:** `5 Spam Email Detection/data/raw/spam.csv` (5,572 messages: 4,825 ham, 747 spam).
- **Splits:** Stratified 70% Train (3,900), 15% Val (836), 15% Test (836).
- **Leakage Defense:** `TfidfVectorizer(max_features=4000, stop_words='english')` is fit **strictly on `X_train`**. Validation and test sets are transformed using the fitted vocabulary.

#### 4. Diagnostic Visualizations Walkthrough
- `class_imbalance.png`: Visualizes the 86.6% ham vs 13.4% spam distribution, highlighting why precision and recall matter more than raw accuracy.
- `confusion_matrix.png`: Demonstrates high precision on the test set: 721 True Ham, 100 True Spam, only 3 False Positives, and 12 False Negatives.
- `roc_curve.png`: Test ROC-AUC of **0.9865**, confirming clear separation between spam and ham probability distributions.
- `precision_recall_curve.png`: Average Precision (AP) of **0.9634**, showing high precision across varied recall thresholds.
- `top_spam_ham_words.png`: Bar plot proving interpretability: top spam keywords include *call, free, txt, claim, prize, win, cash, urgent*, while ham keywords include *come, ok, good, love, going, later*.

#### 5. Code Walkthrough
- `src/features/build_features.py`: Cleans text (lowercase, strips numbers and punctuation), fits TF-IDF vectorizer, and serializes `vectorizer.joblib`.
- `src/models/train_model.py`: Fits `MultinomialNB(alpha=1.0)`, evaluates stratified 5-fold CV, and dumps `spam_model.joblib`.
- `web/app.py`: Serves on port `5005`. Real-time email scanner interface displaying spam risk meter, confidence gauge, and detected spam keywords.

#### 6. Examiner Viva Questions & Answers
- **Q: What is the "naive" assumption in Naive Bayes, and why does it work well despite being violated?**
  *A:* The naive assumption is that all words in a document are conditionally independent given the class label ($P(w_1, w_2 \mid c) = P(w_1 \mid c)P(w_2 \mid c)$). In natural language, words are clearly correlated (e.g., "credit" and "card"). However, classification depends only on the *argmax* of the posterior probabilities, not their exact numerical values. Even if probability estimates are biased, the class ranking remains correct as long as the dominant class probability remains highest.
- **Q: Why is False Positive rate critical in Spam Email Detection?**
  *A:* An asymmetric misclassification cost exists: a False Negative (spam in inbox) is a minor inconvenience, but a False Positive (legitimate business email or job offer sent to spam) can cause significant harm. Therefore, spam classifiers are tuned for high **Precision** ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).

---

### PROJECT 6: MNIST HANDWRITTEN DIGIT CLASSIFICATION
- **Task:** Computer Vision Multi-class Spatial Pattern Recognition.
- **Problem Statement:** Classify $28 \times 28$ grayscale images of handwritten digits into their respective numeric classes ($0$ through $9$) using a Multi-Layer Perceptron (Neural Network).

#### 1. Mathematical Formulation: Multi-Layer Perceptron (MLP)
1. **Network Architecture:**
   - Input Layer: Flattened $28 \times 28 = 784$ grayscale pixel intensities $x \in [0, 1]^{784}$.
   - Hidden Layer 1: $h_1 \in \mathbb{R}^{128}$ with Rectified Linear Unit (ReLU) activation.
   - Hidden Layer 2: $h_2 \in \mathbb{R}^{64}$ with ReLU activation.
   - Output Layer: $z \in \mathbb{R}^{10}$ with Softmax activation.
2. **Forward Propagation Equations:**
   $$a^{[1]} = \text{ReLU}\left( W^{[1]} x + b^{[1]} \right) = \max\left(0, W^{[1]} x + b^{[1]}\right)$$
   $$a^{[2]} = \text{ReLU}\left( W^{[2]} a^{[1]} + b^{[2]} \right) = \max\left(0, W^{[2]} a^{[1]} + b^{[2]}\right)$$
   $$\hat{y} = \text{Softmax}\left( W^{[3]} a^{[2]} + b^{[3]} \right) = \frac{e^{z_k^{[3]}}}{\sum_{j=0}^9 e^{z_j^{[3]}}}$$
3. **Loss Function (Categorical Cross-Entropy):**
   $$\mathcal{L}(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=0}^9 y_{ik} \log(\hat{y}_{ik}) + \frac{\alpha}{2} \sum_{l=1}^3 \|W^{[l]}\|_F^2$$
4. **Backpropagation via Chain Rule:**
   $$\delta^{[3]} = \frac{\partial \mathcal{L}}{\partial z^{[3]}} = \hat{y} - y$$
   $$\frac{\partial \mathcal{L}}{\partial W^{[3]}} = \delta^{[3]} (a^{[2]})^T, \qquad \frac{\partial \mathcal{L}}{\partial b^{[3]}} = \delta^{[3]}$$
   $$\delta^{[2]} = (W^{[3]})^T \delta^{[3]} \odot \text{ReLU}'(z^{[2]}), \quad \text{where } \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$$
5. **Optimization via Adam (Adaptive Moment Estimation):**
   Maintains exponentially decaying averages of past gradients ($m_t$) and squared gradients ($v_t$):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

#### 2. Why Multi-Layer Perceptron (Neural Network)?
Pixel grids are non-linear; individual pixels possess almost zero linear correlation with target digit identity. A digit's identity emerges from spatial combinations of edge segments, loops, and strokes. The hidden layers learn this hierarchy: Hidden Layer 1 detects low-level edge fragments, and Hidden Layer 2 synthesizes loops and strokes, enabling $93.91\%$ test accuracy.

#### 3. Dataset Splitting & Preprocessing
- **Raw File:** `6 MNIST Handwritten Digit Classification/data/raw/mnist.csv` (15,000 balanced images, 1,500 per digit).
- **Splits:** Stratified 70% Train (10,500), 15% Val (2,250), 15% Test (2,250).
- **Pixel Normalization:** Min-Max scaling maps pixel values from $[0, 255]$ to $[0.0, 1.0]$: $x_{\text{norm}} = \frac{x}{255.0}$.

#### 4. Diagnostic Visualizations Walkthrough
- `sample_digits_grid.png`: $4 \times 4$ gallery displaying actual raw pixel digit samples.
- `confusion_matrix.png`: $10 \times 10$ matrix revealing classification precision across all digits. Minor confusion occurs between visually similar pairs (e.g. 4 vs 9, and 3 vs 5).
- `loss_curve.png`: Shows monotonic decrease of training loss from $\sim 1.8$ down to $<0.15$ over 25 epochs.
- `t_sne_digits.png`: 2D t-SNE manifold projection demonstrating distinct clustering of digit classes in latent space.
- `per_class_roc_curves.png`: Multiclass OvR ROC curves showing AUC $\ge 0.99$ across all 10 digits.

#### 5. Code Walkthrough
- `src/features/build_features.py`: Normalizes pixel intensities, creates stratified 3-way splits, and exports data.
- `src/models/train_model.py`: Configures `MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=30)`, monitors early stopping on validation loss, and saves `mnist_model.joblib`.
- `web/app.py`: Serves on port `5006`. Features an interactive HTML5 Canvas where users draw a digit with their mouse; vanilla JS extracts the $28 \times 28$ bounding box, centers the center-of-mass, and displays real-time classification probabilities.

#### 6. Examiner Viva Questions & Answers
- **Q: Why is ReLU preferred over Sigmoid in deep hidden layers?**
  *A:* For large absolute inputs $|z| > 3$, the derivative of the Sigmoid function approaches zero ($\sigma'(z) \le 0.25$). When backpropagating through multiple layers, multiplying these small gradients causes the **vanishing gradient problem**, stalling early layer updates. In contrast, the derivative of ReLU is constant ($\frac{d}{dz}\text{ReLU}(z) = 1$ for $z > 0$), allowing gradients to flow unimpeded.
- **Q: What is the purpose of centering the center-of-mass in digit preprocessing?**
  *A:* The MNIST training dataset was normalized by calculating the center of mass of pixel intensities and translating the digit image so that this center coincides with the center of the $28 \times 28$ bounding box. Preprocessing drawn digits with the same center-of-mass translation ensures consistent feature alignment with training samples.

---

### PROJECT 7: MOVIE RECOMMENDATION SYSTEM
- **Task:** Unsupervised Latent Factor Matrix Factorization & Collaborative Filtering.
- **Problem Statement:** Predict user movie preferences and generate personalized recommendations using an incomplete user-item interaction matrix ($610 \text{ users} \times 1223 \text{ movies}$).

#### 1. Mathematical Formulation: Truncated Singular Value Decomposition (SVD)
The user-item rating matrix $R \in \mathbb{R}^{m \times n}$ contains ratings given by $m$ users to $n$ movies. Because users rate only a fraction of available movies, $R$ is sparse ($>95\%$ empty).
1. **Low-Rank Singular Value Decomposition:**
   $$R \approx U_k \Sigma_k V_k^T$$
   Where:
   - $U_k \in \mathbb{R}^{m \times k}$: User latent factor matrix representing user affinity across $k$ hidden taste dimensions.
   - $\Sigma_k \in \mathbb{R}^{k \times k}$: Diagonal matrix containing the top $k$ singular values (strengths of latent factors).
   - $V_k \in \mathbb{R}^{n \times k}$: Item latent factor matrix representing movie loadings on the $k$ latent genres.
2. **Latent Movie Representation:**
   The movie latent projection is defined as:
   $$M_{\text{latent}} = V_k \Sigma_k \in \mathbb{R}^{n \times k}$$
3. **Item-Item Cosine Similarity Metric:**
   The similarity between two movies $i$ and $j$ in the $k$-dimensional latent space is the cosine of the angle between their factor vectors:
   $$\text{CosineSim}(v_i, v_j) = \frac{v_i \cdot v_j}{\|v_i\|_2 \|v_j\|_2} = \frac{\sum_{l=1}^k v_{i, l} v_{j, l}}{\sqrt{\sum_{l=1}^k v_{i, l}^2} \sqrt{\sum_{l=1}^k v_{j, l}^2}}$$
4. **Reconstructed Rating Matrix:**
   $$\hat{R} = (U_k \Sigma_k^{1/2}) (\Sigma_k^{1/2} V_k^T) = U_k \Sigma_k V_k^T$$

#### 2. Why Matrix Factorization via TruncatedSVD?
Simple memory-based collaborative filtering (computing Pearson correlation across all user rows) scales poorly ($\mathcal{O}(m^2)$) and suffers from matrix sparsity. TruncatedSVD compresses the rating matrix into $k=20$ compact orthogonal latent factors, capturing abstract patterns (e.g., sci-fi themes, art-house pacing, humor style) while explaining $61.4\%$ of rating variance.

#### 3. Dataset Splitting & Preprocessing
- **Raw Files:** `7 Movie Recommendation System/data/raw/ratings.csv` (100,836 ratings) and `movies.csv` (9,742 movies).
- **Preprocessing:** Filtered to movies with $\ge 20$ ratings to eliminate sparse noise, producing a dense user-item matrix of $610 \text{ users} \times 1223 \text{ movies}$.
- **Matrix Normalization:** Mean user ratings are subtracted to center user-specific grading biases (e.g. harsh raters vs generous raters).
- **Splits:** 70% Train ratings (70,585), 15% Val ratings (15,125), 15% Test ratings (15,126).

#### 4. Diagnostic Visualizations Walkthrough
- `rating_distribution.png`: Histogram showing ratings clustered around $3.0$, $4.0$, and $5.0$, with a global mean rating of $3.54$.
- `svd_scree_plot.png`: Scree plot showing cumulative variance explained across components; the first 20 singular values capture $61.4\%$ of total rating variance.
- `latent_space_movies.png`: 2D scatter plot visualizing movie embeddings in latent space; shows thematic clustering (e.g. *Toy Story* clusters with animated features, *The Matrix* clusters with *Terminator*).
- `user_activity_histogram.png`: Long-tail power-law distribution of ratings per user.
- `cosine_similarity_matrix.png`: Heatmap showing cosine similarities between representative film titles.

#### 5. Code Walkthrough
- `src/features/build_features.py`: Pivots ratings into user-item matrix, handles mean-centering, and creates split matrices.
- `src/models/train_model.py`: Fits `TruncatedSVD(n_components=20, random_state=42)` on transposed movie-user matrix, normalizes latent vectors, and serializes `recommender_pipeline.joblib`.
- `web/app.py`: Serves on port `5007`. Features search-as-you-type movie query engine and displays Top-10 recommended titles with similarity scores and genre tags.

#### 6. Examiner Viva Questions & Answers
- **Q: What is the "Cold Start Problem" in Collaborative Filtering?**
  *A:* The cold start problem occurs when a new user joins or a new movie is added without any prior rating history. Because Collaborative Filtering relies on interaction history, it cannot compute meaningful latent factors for these new entities. This is mitigated using hybrid approaches that incorporate content-based metadata (genres, cast, synopsis).
- **Q: How does TruncatedSVD handle missing values compared to Alternating Least Squares (ALS)?**
  *A:* Standard SVD requires a complete matrix; missing entries are typically imputed with zeros or mean user ratings. Alternating Least Squares (ALS) and probabilistic matrix factorization optimize the loss function *only* over observed entries $(u, i) \in R_{\text{obs}}$, avoiding the need to impute unobserved ratings.

---

### PROJECT 8: FASHION MNIST CLASSIFICATION
- **Task:** Computer Vision Clothing Category Classification.
- **Problem Statement:** Classify $28 \times 28$ grayscale images of fashion garments into 10 apparel categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

#### 1. Mathematical Formulation: Multi-Layer Perceptron for Complex Vision
Fashion MNIST is designed as a direct drop-in replacement for the original MNIST digit dataset, matching its format ($28 \times 28$ grayscale pixels, 10 classes). However, it presents higher intraclass variance and complex texture patterns.
1. **Network Architecture:**
   - Input: Flattened $28 \times 28 = 784$ normalized pixel features.
   - Hidden Layer 1: 128 neurons with ReLU activation.
   - Hidden Layer 2: 64 neurons with ReLU activation.
   - Output Layer: 10 neurons with Softmax activation.
2. **Softmax Output Formulation:**
   $$P(y = c \mid x) = \frac{\exp(W_c^{[3]} a^{[2]} + b_c^{[3]})}{\sum_{j=0}^9 \exp(W_j^{[3]} a^{[2]} + b_j^{[3]})}$$
3. **Weight Regularization (L2 Weight Decay):**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CCE}} + \frac{\lambda}{2} \sum_{l=1}^3 \sum_{j} \sum_{k} (W_{jk}^{[l]})^2$$
   Weight decay penalizes large individual weights, preventing the network from relying heavily on isolated pixel noise.

#### 2. Why Fashion MNIST?
While MNIST digits can be separated with $>98\%$ accuracy by relatively simple models due to high contrast between foreground strokes and blank backgrounds, Fashion MNIST involves complex textures, silhouettes, and subtle category boundaries (e.g., distinguishing a Pullover from a Coat or a T-shirt from a Shirt), providing a more realistic computer vision benchmark.

#### 3. Dataset Splitting & Preprocessing
- **Raw File:** `8 Fashion MNIST Classification/data/raw/fashion_mnist.csv` (15,000 images, 1,500 per class).
- **Splits:** Stratified 70% Train (10,500), 15% Val (2,250), 15% Test (2,250).
- **Normalization:** Grayscale pixel intensities are normalized from $[0, 255]$ to $[0.0, 1.0]$.

#### 4. Diagnostic Visualizations Walkthrough
- `fashion_samples_grid.png`: $4 \times 4$ visual gallery displaying apparel categories (coats, boots, bags, shirts).
- `confusion_matrix.png`: Heatmap revealing clear category separation for distinctive items (e.g. Trouser accuracy $>97\%$, Bag $>93\%$), alongside expected confusion between visually similar classes: `Shirt` vs `T-shirt/top` and `Coat` vs `Pullover`.
- `loss_curve.png`: Shows steady decline of cross-entropy loss over 30 training epochs.
- `per_class_roc_curves.png`: Multiclass OvR ROC curves yielding an overall micro-average AUC of **0.9841**.
- `tsne_fashion_clustering.png`: 2D t-SNE projection showing natural separation between footwear (Sneaker, Sandal, Ankle boot) and tops (T-shirt, Coat, Pullover).

#### 5. Code Walkthrough
- `src/features/build_features.py`: Normalizes pixel values, encodes category labels, and exports clean CSV splits.
- `src/models/train_model.py`: Fits `MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, alpha=1e-4)`, monitors early stopping, and dumps `fashion_mnist_model.joblib`.
- `web/app.py`: Serves on port `5008`. Interactive canvas application for sketching or testing apparel silhouettes, displaying top-3 predicted garment categories with confidence percentages.

#### 6. Examiner Viva Questions & Answers
- **Q: Why is classification accuracy lower on Fashion MNIST (~86%) compared to Digit MNIST (~94%) using the same architecture?**
  *A:* Handwritten digits consist of simple white curves against solid black backgrounds with minimal texture. Fashion items contain complex internal textures, varied necklines, button patterns, and similar silhouettes (e.g., T-shirts, Shirts, and Pullovers share nearly identical spatial shapes), creating higher intraclass variance and lower interclass separability for fully-connected networks.
- **Q: How would a Convolutional Neural Network (CNN) improve performance on Fashion MNIST?**
  *A:* Fully-connected MLPs flatten the $28 \times 28$ image into a 1D vector of 784 pixels, losing 2D spatial relationships. CNNs use convolutional kernels to preserve spatial locality and translation invariance, extracting localized edge and texture patterns regardless of where they appear on the garment.

---

### PROJECT 9: SENTIMENT ANALYSIS OF TEXT
- **Task:** Natural Language Processing (NLP) Binary Sentiment Classification.
- **Problem Statement:** Determine whether an unstructured natural language movie review expresses positive or negative sentiment using Logistic Regression and n-gram TF-IDF representations.

#### 1. Mathematical Formulation: Sublinear TF-IDF & Logistic Regression
1. **Sublinear Term Frequency:**
   Standard term frequency scales linearly. Sublinear scaling replaces raw frequency with logarithmic scaling to dampen the impact of repeated words:
   $$\text{TF}_{\text{sublinear}}(t, d) = \begin{cases} 1 + \log(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$
2. **Inverse Document Frequency:**
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
3. **N-Grams $(1, 2)$ for Local Context Preservation:**
   Extracting contiguous token pairs (bigrams) preserves local negation semantics (e.g., distinguishing *"not good"* from *"good"*, or *"waste of time"* from *"time"*).
4. **Logistic Regression Prediction & Loss:**
   $$P(y = 1 \mid d) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
   $$\mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{1}{2C} \|w\|_2^2$$
5. **Feature Interpretability:**
   Each word feature has a direct, interpretable weight:
   - $w_j > 0$: Increases the probability of positive sentiment.
   - $w_j < 0$: Increases the probability of negative sentiment.

#### 2. Why Logistic Regression with Sublinear TF-IDF?
Deep neural networks (e.g. Transformers) require substantial memory and compute. Logistic Regression with sublinear TF-IDF runs in seconds, is computationally efficient, avoids overfitting on moderate corpora, provides transparent feature interpretability via weight coefficients, and achieves **86.89%** test accuracy and **0.9426** ROC-AUC.

#### 3. Dataset Splitting & Preprocessing
- **Raw File:** `9 Sentiment Analysis of Text/data/raw/sentiment.csv` (6,000 balanced reviews: 3,028 negative, 2,972 positive).
- **Splits:** Stratified 70% Train (4,200), 15% Val (900), 15% Test (900).
- **Preprocessing Pipeline:** Strips HTML break tags (`<br />`), removes URLs, strips non-alphabetic characters, and applies `TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1,2), sublinear_tf=True)` fit strictly on `X_train`.

#### 4. Diagnostic Visualizations Walkthrough
- `sentiment_distribution.png`: Confirms equal class balance (~50% Positive vs ~50% Negative).
- `review_length_distribution.png`: Compares review lengths across sentiments; shows identical log-normal length profiles peaking at 120–180 words.
- `confusion_matrix.png`: Test set performance: 387 True Negatives, 395 True Positives, 67 False Positives, 51 False Negatives.
- `roc_curve.png`: Strong test ROC-AUC of **0.9426**.
- `precision_recall_curve.png`: High precision maintained across recall thresholds, yielding an Average Precision of **0.9412**.
- `top_sentiment_words.png`: Bar plot of top model weights:
  - Strongest Positive: *excellent, great, loved, wonderful, best, superb, beautiful*.
  - Strongest Negative: *worst, waste, awful, terrible, boring, poor, horrible*.

#### 5. Code Walkthrough
- `src/features/build_features.py`: Cleans raw text, extracts unigrams and bigrams, fits TF-IDF vectorizer, and serializes `vectorizer.joblib`.
- `src/models/train_model.py`: Fits `LogisticRegression(C=1.0, max_iter=200)`, evaluates metrics, and dumps `sentiment_model.joblib`.
- `web/app.py`: Serves on port `5009`. Review sentiment analyzer providing real-time text analysis, sentiment polarity meter, and detected sentiment keywords.

#### 6. Examiner Viva Questions & Answers
- **Q: How does incorporating bigrams handle negation in bag-of-words models?**
  *A:* In pure unigram models, the phrase "not good" is split into "not" and "good". Because "good" carries a strong positive weight, the model often misclassifies the review as positive. By including bigrams (`ngram_range=(1, 2)`), the model extracts "not_good" as a distinct feature token and learns an explicit negative weight for it.
- **Q: Why is sublinear term frequency scaling useful in sentiment analysis?**
  *A:* A reviewer mentioning the word "terrible" 6 times in a review does not express six times the negative sentiment of a reviewer using it twice. Sublinear scaling replaces raw term frequency with $1 + \log(\text{TF})$, dampening the impact of repeated words and preventing verbose reviews from distorting prediction scores.

---

### PROJECT 10: STOCK PRICES PREDICTION
- **Task:** Financial Time Series Regression under Temporal Ordering Constraints.
- **Problem Statement:** Predict the next-day closing price ($y_{t+1}$) of Apple Inc. (`AAPL`) stock based on historical Open, High, Low, Close, Volume (OHLCV) records and engineered technical indicators.

#### 1. Mathematical Formulation: Technical Indicators & Time Series Regression
1. **Simple Moving Average (SMA):**
   $$\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$
2. **Exponential Moving Average (EMA):**
   $$\text{EMA}_k(t) = \alpha P_t + (1 - \alpha) \text{EMA}_k(t-1), \quad \text{where } \alpha = \frac{2}{k + 1}$$
3. **Relative Strength Index (RSI - 14 Days):**
   $$\text{RS} = \frac{\text{Average Gain over 14 days}}{\text{Average Loss over 14 days}}, \qquad \text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
4. **Logarithmic Returns & Rolling Volatility:**
   $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right), \qquad \sigma_{t, 10} = \sqrt{\frac{1}{9} \sum_{i=0}^9 (r_{t-i} - \bar{r}_t)^2}$$
5. **Random Forest Regression Formulation:**
   $$\hat{y}_{t+1} = \frac{1}{B} \sum_{b=1}^B T_b(x_t)$$
   Constrained to `max_depth=6` to prevent the ensemble from memorizing transient intraday noise.

#### 2. Why Random Forest with Technical Indicators?
Financial price series exhibit non-linear patterns, structural breaks, and regime shifts that standard linear autoregressive models (ARIMA) struggle to capture without restrictive stationarity assumptions. Random Forest handles non-linear interactions across technical indicators (e.g. RSI momentum thresholds combined with moving average crossovers) without requiring stationarity transformations, achieving a test $R^2 = 0.9245$ and a Mean Absolute Percentage Error (MAPE) of **1.19%**.

#### 3. Temporal Splitting & The Lookahead Bias Rule
**Crucial Principle:** In financial time series, random shuffling introduces fatal **lookahead bias (data leakage)**. Training on day $t+5$ to predict day $t+2$ leaks future market conditions into past predictions.
- **Raw File:** `10 Stock Prices Prediction/data/raw/stock_prices.csv` (506 trading days).
- **Strict Chronological Splits:**
  - Training Set (Earliest 70%): 340 trading days ($t_1 \to t_{340}$).
  - Validation Set (Next 15%): 73 trading days ($t_{341} \to t_{413}$).
  - Test Set (Final Held-Out 15%): 73 trading days ($t_{414} \to t_{486}$).
- **Walk-Forward Validation:** Cross-validation uses `TimeSeriesSplit(n_splits=5)`: each fold uses an expanding historical window where training folds strictly precede the validation fold in calendar time.

#### 4. Diagnostic Visualizations Walkthrough
- `price_history_volume.png`: Full historical price trajectory with dual-panel trading volume bars.
- `technical_indicators.png`: Three-panel plot showing SMA(5, 20) crossovers, EMA(10) tracking, and RSI(14) with overbought (70) and oversold (30) boundary lines.
- `correlation_heatmap.png`: Matrix showing high collinearity among price features ($r > 0.95$) and lower correlation for momentum oscillators (RSI, Volatility), proving feature diversity.
- `actual_vs_predicted_prices.png`: Time series plot comparing actual test prices (blue) against predictions (red dashed); tracks ground truth with test $R^2 = 0.9245$ and $\text{RMSE} = \$2.10$.
- `residual_plot.png`: Shows even error scatter around zero and a symmetric residual histogram, confirming model calibration.
- `feature_importances.png`: Identifies recent closing prices, `EMA_10`, and `SMA_5` as top predictive features.
- `cross_validation_tscv.png`: TimeSeriesSplit cross-validation scores across expanding historical windows.

#### 5. Code Walkthrough
- `src/features/build_features.py`: Computes SMA, EMA, RSI, lag features, and rolling volatility; drops initial warm-up rows; applies strict chronological indexing; fits `RobustScaler` on training split.
- `src/models/train_model.py`: Fits `RandomForestRegressor(n_estimators=100, max_depth=6)`, runs `TimeSeriesSplit`, and dumps `stock_model.joblib`.
- `web/app.py`: Serves on port `5010`. Quantitative dashboard for entering OHLCV values and indicators, displaying next-day price forecast, dollar delta, and trading signal (Bullish / Bearish / Neutral).

#### 6. Examiner Viva Questions & Answers
- **Q: Why is standard K-Fold cross-validation invalid for financial time series?**
  *A:* Standard K-Fold randomly assigns observations to folds regardless of time. In time series forecasting, this causes catastrophic **lookahead bias (data leakage)**: the model is trained on future data points (e.g. day $t+10$) to predict past prices (day $t+2$). In real-world trading, future data does not exist. We must use walk-forward validation (`TimeSeriesSplit`), where training folds strictly precede test folds in chronological time.
- **Q: What does a Mean Absolute Percentage Error (MAPE) of 1.19% signify?**
  *A:* $\text{MAPE} = \frac{100\%}{n}\sum \left|\frac{y - \hat{y}}{y}\right|$. A MAPE of $1.19\%$ means our next-day price prediction deviates from actual market prices by an average of $\$1.19$ per $\$100$ of stock price, indicating strong predictive calibration for short-term price forecasting.

---

## 4. MASTER CROSS-PROJECT VIVA QUESTIONS & COMPARATIVE SYNTHESIS

### 4.1 Comparative Algorithm Matrix

| Algorithm Family | Projects Used | Key Strengths | Limitations / Weaknesses |
|---|---|---|---|
| **Linear / Logistic Regression (OLS & Softmax)** | 1 (Iris), 3 (Boston OLS), 4 (Diabetes OLS), 9 (Sentiment) | Convex loss (global minimum), closed-form / fast convergence, directly interpretable weights, zero black-box opacity. | Assumes linear relationships; vulnerable to multicollinearity without regularization or dimensionality reduction. |
| **Support Vector Machines (SVR)** | 3 (Boston Linear SVR) | Robust against outliers via $\epsilon$-insensitive tube, convex quadratic programming optimization. | Sensitive to hyperparameter choices ($C, \epsilon$), requires careful scaling of feature inputs. |
| **Random Forest (Bagging)** | 2 (Titanic), 10 (Stocks) | Captures non-linear feature interactions, handles mixed feature types, invariant to monotonic scaling, robust against noise. | Computationally intensive at inference time, cannot extrapolate beyond historical target ranges. |
| **Naive Bayes (Probabilistic)** | 5 (Spam Detection) | Linear time training $\mathcal{O}(N \cdot L)$, handles sparse high-dimensional text data, low memory requirements. | Assumes feature independence; conditional probability values can be poorly calibrated. |
| **Multi-Layer Perceptron (Neural Net)**| 6 (MNIST), 8 (Fashion MNIST) | Universal function approximator, learns hierarchical feature representations across hidden layers. | Non-convex loss surface (local minima), sensitive to weight initialization, requires careful tuning of learning rate and architecture. |
| **TruncatedSVD (Matrix Factorization)**| 7 (Movie Recommender) | Compresses sparse interaction matrices into compact latent factors, captures implicit tastes. | Suffers from the Cold Start Problem; standard SVD requires imputing unobserved values. |

---

### 4.2 Universal Examiner Viva Voce Questions

**Q1: What is the fundamental difference between L1 (Lasso) and L2 (Ridge) Regularization?**
> **Answer:** Both methods add a penalty term to the loss function to prevent overfitting:
> - **L2 Regularization (Ridge):** Penalty is proportional to the sum of squared weights: $\frac{\lambda}{2} \|w\|_2^2$. In weight space, the penalty contour is a hypersphere. The gradient pulls weights toward zero smoothly, shrinking coefficients for correlated features without setting them strictly to zero.
> - **L1 Regularization (Lasso):** Penalty is proportional to the sum of absolute weights: $\lambda \|w\|_1$. The penalty contour is a diamond/polytope with sharp corners on the coordinate axes. Optimization tends to hit these axis corners, driving irrelevant feature weights to **exactly zero**, thereby performing automatic feature selection and producing sparse models.

**Q2: What is the Bias-Variance Tradeoff, and how do our diagnostic learning curves visualize it?**
> **Answer:** Expected generalization error decomposes into three terms:
> $$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \sigma^2 \text{ (Irreducible Noise)}$$
> - **Bias:** Error introduced by approximating a complex real-world phenomenon with a simplified model (e.g. fitting a straight line to quadratic data). High bias leads to underfitting. On learning curves, both training and validation errors remain high with little gap between them.
> - **Variance:** Error introduced by the model's sensitivity to small fluctuations in the training set (e.g. unconstrained deep decision trees memorizing noise). High variance leads to overfitting. On learning curves, training error is near zero while validation error remains high, resulting in a large generalization gap.
> - **Optimal Tradeoff:** Where the validation error reaches its global minimum, balancing model complexity with generalization.

**Q3: When should you evaluate a model using the ROC Curve vs. the Precision-Recall (PR) Curve?**
> **Answer:**
> - **ROC Curves** plot True Positive Rate against False Positive Rate. Because False Positive Rate incorporates True Negatives ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$), a large majority negative class inflates the denominator, keeping FPR artificially low and making the model look overly optimistic. ROC is appropriate when classes are roughly balanced (e.g. Sentiment Analysis, Iris).
> - **Precision-Recall Curves** plot Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) against Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). Neither metric includes True Negatives in its denominator. Consequently, PR curves focus entirely on performance on the minority positive class, making them the preferred evaluation metric for imbalanced problems (e.g. Spam Email Detection, Fraud Detection).

**Q4: Why is it necessary to scale features before training Neural Networks and Support Vector Machines, but optional for Decision Trees?**
> **Answer:**
> - **Distance/Gradient-based Models (MLP, Logistic Regression, SVM):** These models compute inner products ($w^T x$) or Euclidean distances ($\|x_i - x_j\|_2$). A feature with a large numerical scale (e.g., Annual Income: \$100,000) will dominate features with small numerical ranges (e.g., Age: 25), distorting the distance metric and causing elongated gradient descent paths. Scaling ensures all features contribute equally to optimization.
> - **Tree-based Models (Decision Trees, Random Forest):** Trees evaluate features independently at each split node using monotonic inequality tests ($x_j \le \theta$). The rank order of values is preserved under any monotonic transformation, so feature scaling has zero impact on tree split points or predictions.

---

## 5. COMPLETE VERIFICATION AND EXECUTION COMMANDS

To run any project's end-to-end pipeline or launch its interactive web application:

```bash
# Navigate to project directory (e.g., Project 5: Spam Detection)
cd "/Users/darrehanrasool/Desktop/Lab/5 Spam Email Detection"

# Step 1: Run data validation and raw extraction
/opt/anaconda3/bin/python src/data/make_dataset.py

# Step 2: Build features, perform 3-way split, and fit transformers
/opt/anaconda3/bin/python src/features/build_features.py

# Step 3: Train model, evaluate cross-validation, and dump metrics
/opt/anaconda3/bin/python src/models/train_model.py

# Step 4: Generate all 8 diagnostic figures
/opt/anaconda3/bin/python src/visualization/visualize.py

# Step 5: Launch interactive web application (port 5005)
/opt/anaconda3/bin/python web/app.py
```

### All Web Dashboard Ports:
- Project 1 (Iris): `http://127.0.0.1:5001`
- Project 2 (Titanic): `http://127.0.0.1:5002`
- Project 3 (Boston Housing): `http://127.0.0.1:5003`
- Project 4 (Diabetes): `http://127.0.0.1:5004`
- Project 5 (Spam Detection): `http://127.0.0.1:5005`
- Project 6 (MNIST Digits): `http://127.0.0.1:5006`
- Project 7 (Movie Recommendations): `http://127.0.0.1:5007`
- Project 8 (Fashion MNIST): `http://127.0.0.1:5008`
- Project 9 (Sentiment Analysis): `http://127.0.0.1:5009`
- Project 10 (Stock Prices): `http://127.0.0.1:5010`

---
*Manual compiled and verified for CAI504C Machine Learning Laboratory examination defense.*

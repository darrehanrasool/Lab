# Diabetes Progression Prediction via Multiple Linear Regression

An end-to-end Machine Learning laboratory project that predicts quantitative 1-year diabetes disease progression using authentic clinical vitals and blood serum measurements with **Multiple Linear Regression (Ordinary Least Squares - OLS)**, featuring extensive exploratory data analysis, input feature handling, continuous and categorical imbalance management, zero data leakage verification, and a real-time clinical prognostic web dashboard.

---

## 🎯 Problem Statement & Syllabus Mapping

- **Course:** CAI504C • Machine Learning Laboratory | Semester I • IUST Kashmir
- **Syllabus Mapping:** Experiment 4 (Diabetes Progression Prediction)
- **Clinical Problem Statement (The Story of Dr. Ayesha):**
  - Two Type-2 diabetic patients visit an endocrinologist today, both feeling relatively normal. Yet 12 months later, Patient A remains stable with mild progression, while Patient B suffers rapid microvascular collapse, acute nephropathy (kidney damage), or retinopathy (vision loss).
  - A doctor cannot see future deterioration with a physical exam alone. By the time clinical symptoms manifest, microvascular tissue damage is permanent and irreversible.
  - An accurate Machine Learning model provides a vital **12-month early-warning window**, enabling proactive therapy adjustments before permanent organ failure occurs.
  - **Asymmetric Cost of Mistakes:** Prediction errors are not equal in healthcare. *Under-predicting* (e.g., predicting 110 for a patient whose true score is 290) delays critical therapy, causing irreversible kidney failure. *Over-predicting* (e.g., predicting 300 for a score of 100) triggers excessive insulin, causing life-threatening hypoglycemia (fainting/coma).
- **Objective:** Predict quantitative disease progression in diabetic patients one year after baseline using 10 diagnostic clinical vitals and blood serum biomarkers.
- **Target Variable ($y$):** Quantitative measure of disease progression 1 year post baseline (continuous scale from 25.0 to 346.0).
- **Core ML Technique:** Multiple Linear Regression (Ordinary Least Squares — OLS) — 100% transparent, closed-form Normal Equation, and fully explainable.

---

## 🔬 Mathematical Formulation & Theory

### 1. Multivariate Linear Regression Model
We model disease progression $\hat{y}$ as a linear combination of the $p=10$ standardized clinical features:
$$\hat{y}(\mathbf{x}) = \beta_0 + \sum_{j=1}^{10} \beta_j z_j$$
where $z_j = \frac{x_j - \mu_j}{\sigma_j}$ is the z-score standardized biomarker value.

In matrix notation:
$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta} \quad \text{where } \mathbf{X} \in \mathbb{R}^{N \times 11}, \, \boldsymbol{\beta} \in \mathbb{R}^{11}$$

### 2. Ordinary Least Squares (OLS) Loss Function
The model parameters $\boldsymbol{\beta}$ are trained by minimizing the **Mean Squared Error (MSE)**, also known as the Residual Sum of Squares (RSS):
$$J(\boldsymbol{\beta}) = \frac{1}{2N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{2N} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

### 3. Closed-Form Normal Equation Solution
Setting the gradient of $J(\boldsymbol{\beta})$ with respect to $\boldsymbol{\beta}$ to zero yields the exact analytical global minimum:
$$\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = \frac{1}{N} \left[ -\mathbf{X}^T \mathbf{y} + \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} \right] = \mathbf{0}$$
$$\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

Because $p = 10$, the matrix $(\mathbf{X}^T \mathbf{X})$ is $11 \times 11$, which solves instantaneously in less than 1 millisecond without iterative gradient descent tuning.

---

## 📁 Dataset & Feature Handling

The cohort consists of 442 diabetes patients recorded by Efron et al. (2004), preserved in **authentic unscaled physical units**:

| Feature | Clinical Measurement | Authentic Unit | How Handled in Pipeline |
| :--- | :--- | :--- | :--- |
| `age` | Patient Chronological Age | Years (19 to 79) | Continuous: Standardized via z-score |
| `sex` | Demographic Biological Sex | Categorical (Raw: 1.0, 2.0) | **Mapped to binary indicator: 0.0 (Female), 1.0 (Male)** |
| `bmi` | Body Mass Index | $\text{kg/m}^2$ (18.5 to 42.2) | Continuous: Standardized via z-score |
| `bp` | Mean Arterial Blood Pressure | $\text{mmHg}$ (62 to 133) | Continuous: Standardized via z-score |
| `s1` | Total Serum Cholesterol (TC) | $\text{mg/dL}$ (97 to 301) | Continuous: Standardized via z-score |
| `s2` | Low-Density Lipoproteins (LDL) | $\text{mg/dL}$ (59.6 to 242.4) | Continuous: Standardized via z-score |
| `s3` | High-Density Lipoproteins (HDL) | $\text{mg/dL}$ (22 to 99) | Continuous: Standardized via z-score |
| `s4` | Total Cholesterol / HDL Ratio | Ratio (2 to 9.09) | Continuous: Standardized via z-score |
| `s5` | Log of Serum Triglycerides (LTG) | Log scale (3.25 to 6.11) | Continuous: Log-transformed biologically |
| `s6` | Fasting Blood Glucose | $\text{mg/dL}$ (58 to 124) | Continuous: Standardized via z-score |
| **`progression`** | **1-Year Disease Progression** | **Quantitative Score (25 to 346)** | **Continuous Target (Right-skewed, +0.44)** |

### Why We Standardize & The Zero Data Leakage Rule
- **Why scale?** Total cholesterol spans ~204 units while log-triglycerides spans only 2.8. Standardizing to z-scores ($z = \frac{x - \mu}{\sigma}$) ensures every $\beta_j$ represents the change in progression per **1 standard deviation change** in that clinical biomarker.
- **Zero Data Leakage:** The `StandardScaler` is fitted **EXCLUSIVELY** on the 70% Training set (`X_train`). Validation and test sets are strictly transformed using the training mean and standard deviation.

---

## ⚖️ Managing Dataset Imbalance

Dataset imbalance was managed across two distinct levels:

### 1. Continuous Target Imbalance (Right-Skewed Tail Sparsity)
- Mean = 152.13, Median = 140.50, Skewness = **+0.44** (Right-skewed positive tail).
- Severe progression cases ($>200$) make up only **28.7%** of patients.
- **Management Strategy:** We applied **5-bin quantile stratification** across the 70% Train, 20% Val, and 10% Test partitions.
- **Result:** Train mean (151.21), Val mean (155.05), and Test mean (152.78) match with zero distribution drift.

### 2. Clinical Triage Class Imbalance (Severe vs Controlled)
- Clinical Triage Threshold: Severe Progression ($y \ge 200$, 28.7% prevalence).
- **The Problem:** Naive unweighted Logistic Regression achieves 84.4% accuracy but misses 5 out of 13 severe patients (**Sensitivity is only 61.54%**)!
- **Management Strategy:** Cost-sensitive learning using inverse-frequency class weights:
  $$w_{\text{severe}} = \frac{N}{2 \times N_{\text{severe}}} \approx 1.78, \quad w_{\text{controlled}} \approx 0.70$$
- **Result:** Clinical Sensitivity jumped from **61.54% to 76.92%** (+15.38% improvement!), cutting missed severe cases from 5 down to 3.

---

## 📊 Evaluation & Benchmark Results

### 1. Holdout Test Set Performance (N=45 unseen patients)
- **$R^2$ Score (Coefficient of Determination):** **`0.4882`** (Explains 48.8% of variance in unseen patients)
- **Root Mean Squared Error (RMSE):** **`55.17`**
- **Mean Absolute Error (MAE):** **`44.92`**
- **5-Fold Cross Validation Mean $R^2$:** **`0.4587 ± 0.0538`** (High fold-to-fold stability)

### 2. Standardized Beta Weights & Medical Insights
$$\hat{y} = 151.21 + 34.89(z_{s5}) + 23.72(z_{\text{bmi}}) + 18.96(z_{s2}) + 17.76(z_{\text{bp}}) + 5.30(z_{s4}) + 4.22(z_{s3}) + 3.57(z_{s6}) - 1.05(z_{\text{age}}) - 10.08(z_{\text{sex}}) - 31.89(z_{s1})$$

- **Serum Triglycerides $s_5$ ($\beta = +34.89$):** Dominant metabolic driver of progression.
- **Body Mass Index BMI ($\beta = +23.72$):** Visceral adiposity drives systemic insulin resistance.
- **Blood Pressure BP ($\beta = +17.76$):** Hypertension accelerates microvascular capillary destruction.
- **Sex ($\beta = -10.08$):** Female baseline differential compared to males.
- **Baseline Intercept $\beta_0 = 151.21$:** Expected progression of an average patient.

---

## 📈 Advanced Dataset Visualizations Index

All 12 high-resolution publication figures are generated in `reports/figures/`:

1. `01_target_progression_distribution.png`: Target progression distribution, KDE, skewness, mean vs median, clinical risk tiers, and continuous tail imbalance.
2. `02_clinical_feature_distributions.png`: Multi-panel distribution of all 10 clinical features (Continuous vs Categorical Binary `sex`).
3. `03_clinical_outlier_detection_boxplots.png`: Boxplots showing outliers, IQR bounds, and physiological extremes.
4. `04_correlation_heatmap_vif_multicollinearity.png`: Heatmap with Pearson correlation coefficients and VIF bar diagnostics.
5. `05_key_biomarkers_vs_progression_scatter.png`: Scatter plots with Linear Regression best-fit lines (BMI, S5, BP, HDL vs Progression).
6. `06_managing_dataset_imbalance.png`: Managing dataset imbalance: Quantile stratification across Train (70%), Val (20%), and Test (10%), plus clinical triage class imbalance pie chart.
7. `07_feature_scaling_transformation.png`: StandardScaler before vs after distributions (demonstrating zero data leakage and unit harmonization).
8. `08_learning_curve_bias_variance.png`: Learning curve (Train vs Validation $R^2$ over training sizes), proving low variance and absence of overfitting.
9. `09_model_evaluation_actual_vs_predicted.png`: Actual vs. Predicted on the holdout test set with identity line $y = \hat{y}$, ±1 RMSE error margin, and test metrics ($R^2=0.4882$).
10. `10_residual_diagnostics_homoscedasticity.png`: Residuals vs Fitted values (homoscedasticity test) and Normal Q-Q plot of residuals (checking normality assumption of OLS).
11. `11_clinical_interpretability_coefficients.png`: Standardized Beta coefficients $\beta_j$ bar chart with clinical risk direction annotations (BMI, S5, BP driving progression).
12. `12_clinical_risk_confusion_matrix_imbalance.png`: Side-by-side Confusion Matrix comparison (Unweighted vs Balanced Logistic Regression, showing how managing class imbalance boosts Sensitivity from 61.5% to 76.9%).

---

## ⚡ Quick Start & Reproduction

### 1. Run Complete ML Pipeline from Scratch
```bash
# Step 1: Ingest raw clinical data
python3 src/data/make_dataset.py

# Step 2: Preprocess features, manage imbalance & split (70/20/10)
python3 src/features/build_features.py

# Step 3: Train Multiple Linear Regression & Evaluate
python3 src/models/train_model.py

# Step 4: Generate all 12 advanced visualizations
python3 src/visualization/visualize.py

# Step 5: Test CLI prediction engine
python3 src/models/predict_model.py
```

### 2. Launch Interactive Clinical Web Dashboard
```bash
python3 web/app.py
```
Open `http://localhost:5003` in your browser.
- Real-time patient prognosis gauge with cohort percentile ranking.
- Exact mathematical breakdown of each biomarker's contribution ($\beta_j \cdot z_j$).
- Clinical risk triage alert and personalized care recommendations.

### 3. Open Interactive 3D Presentation
Open `Diabetes_3D_Presentation.html` directly in any web browser to present the 14 interactive defense slides with 3D visualization and speaker delivery scripts.

### 4. Midterm Viva Defense Master Guide
Read `DIABETES_MIDTERM_MASTER_GUIDE.md` for complete mathematical derivations, line-by-line pipeline explanations, and the top 15 viva defense questions with word-for-word spoken answers for acing the examination.

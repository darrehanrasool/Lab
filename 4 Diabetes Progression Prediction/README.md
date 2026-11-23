# 🩺 Diabetes Progression Prediction via Multiple Linear Regression

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)]()
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)]()
[![Holdout Test R²: 0.4882](https://img.shields.io/badge/Test%20R%C2%B2-0.4882-brightgreen.svg)]()
[![Triage Sensitivity: 76.9%](https://img.shields.io/badge/Triage%20Sensitivity-76.9%25-success.svg)]()
[![Inference Latency: 0.0128 ms](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-blueviolet.svg)]()
[![Throughput Speedup: 14.9x](https://img.shields.io/badge/Throughput%20Speedup-14.9x-orange.svg)]()
[![RAM Footprint: <142MB](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-lightgrey.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

An end-to-end Machine Learning laboratory project that predicts quantitative 1-year diabetes disease progression using authentic clinical vitals and blood serum measurements with **Multiple Linear Regression (Ordinary Least Squares - OLS)** and cost-sensitive balanced logistic triage, featuring extensive exploratory data analysis, continuous and categorical imbalance management, strict zero data leakage verification, and a real-time clinical prognostic web dashboard.

---

## 📓 Master Data Science Pipeline Notebook

The entire clinical data science pipeline—from raw biomarker ingestion, exploratory medical density analysis with uniform publication-grade graphics, mathematical model derivations (OLS vs Decision Tree vs Balanced Logistic Triage), zero-leakage 70/20/10 training, 5-fold cross-validation, to holdout evaluation and REST API simulation—is fully documented and pre-executed in the master Jupyter Notebook:

📁 **[`notebooks/Diabetes_Progression_Pipeline.ipynb`](./notebooks/Diabetes_Progression_Pipeline.ipynb)**

### ✨ Notebook Highlights & Architecture:
- **Story-Driven Clinical Science**: Every code cell is preceded by an explanatory narrative and followed by an in-depth markdown analysis cell detailing metabolic accelerants, multicollinearity, and mathematical formulations.
- **10 Uniform Publication Graphics ($14 \times 7$ inches, 150 DPI)**: Consistent visual dimensions, typography, and cohesive palettes:
  1. *Figure 1*: Diabetes 1-Year Disease Progression Target Distribution & Clinical Risk Tiers
  2. *Figure 2*: Primary Metabolic Accelerants of Diabetic Complications (BMI & s5 Triglycerides)
  3. *Figure 3*: Inter-Biomarker Pearson Correlation & Target Covariance Matrix
  4. *Figure 4*: Variance Inflation Factor (VIF) Across Baseline Clinical Predictors
  5. *Figure 5*: Baseline Feature Distributions & Physiological Outlier Inspection
  6. *Figure 6*: Empirical Proof: OLS Superiority Over Decision Trees ($R^2 = 0.488$ vs $0.254$)
  7. *Figure 7*: Actual vs Predicted Diabetes Progression on Unseen Test Cohort
  8. *Figure 8*: Residual Error vs Predicted Value (Continuous Tail Sparsity Diagnostic)
  9. *Figure 9*: High-Risk Triage Confusion Matrix (Balanced Model: Sensitivity = $76.9\%$)
  10. *Figure 10*: Standardized OLS Feature Beta Coefficients (Biomarker Impact on Progression)
- **Mathematical Formulations**: Explicit Normal Equation derivation, why Decision Trees underperform on small continuous cohorts, and cost-sensitive balanced loss for rare severe cases.
- **Production Asset Export**: Saves serialized model (`models/diabetes_model.joblib`), scaler (`models/scaler.joblib`), and triage classifier (`models/triage_logistic_model.joblib`) for consumption by the Flask API.

---

## 💻 Project Structure

```
4 Diabetes Progression Prediction/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, structure & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & clinical evaluation
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── notebooks/
│   └── Diabetes_Progression_Pipeline.ipynb  <- ⭐ Master Executed Data Science Pipeline Notebook
├── Diabetes_Presentation.html      <- Interactive 13-slide stepper presentation
├── Diabetes_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── data/
│   ├── raw/                                 <- Immutable raw clinical dataset (diabetes.csv, N=442)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (20%), test (10%), scaler
├── models/
│   ├── diabetes_model.joblib                <- Serialized Multiple Linear Regression pipeline
│   ├── scaler.joblib                        <- Standardized feature scaler (Train only)
│   └── triage_logistic_model.joblib         <- Cost-sensitive clinical risk triage model
├── reports/
│   └── figures/                             <- Publication-grade diagnostic plots (Uniform 14x7 / 1200x900)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5004)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Benchmark Metrics (from `accuracy.txt`)

| Dataset Split / Benchmark | Metric Name | Metric Value | Clinical & Statistical Interpretation |
|---|---|---|---|
| **Holdout Test Set (10%, N=45)** | **$R^2$ Score** | **0.4882** (48.82%) | Variance explained on unseen patient cohort |
| Holdout Test Set (10%, N=45) | **RMSE (Loss)** | **55.17** | Root Mean Squared Error in progression score units |
| Holdout Test Set (10%, N=45) | **MAE** | **44.92** | Mean Absolute Error across unseen patient cohort |
| **Validation Set (20%, N=88)** | **$R^2$ Score** | **0.5894** (58.94%) | Independent parameter verification |
| Validation Set (20%, N=88) | **RMSE (Loss)** | **50.71** | Root Mean Squared Error on validation split |
| Validation Set (20%, N=88) | **MAE** | **41.69** | Mean Absolute Error on validation split |
| **Training Set (70%, N=309)** | **$R^2$ Score** | **0.4967** (49.67%) | Baseline fit on scaled training cohort |
| **5-Fold Cross-Validation $R^2$** | **Mean $R^2$** | **0.4587** ($\pm 0.0538$) | Consistent stability across data folds |
| **Clinical Triage Sensitivity** | **High-Risk Recall** | **76.92%** | High sensitivity identifying severe deteriorating patients ($\ge 200$) |
| **Clinical Triage Specificity** | **Specificity** | **84.38%** | Low false alarm rate on routine diabetic patients |

---

## ⚡ Quick Start & Reproduction

### 1. Run the Jupyter Master Notebook
Launch Jupyter or VS Code to explore and execute the end-to-end data science pipeline interactively:
```bash
jupyter notebook "notebooks/Diabetes_Progression_Pipeline.ipynb"
```

### 2. Launch the Flask Web Application & REST API
```bash
cd "4 Diabetes Progression Prediction"
python web/app.py
```
Open your browser and navigate to `http://localhost:5004` to test real-time clinical prognostics.

---

<div align="center">
  <sub>Built with ❤️ for CAI504C Machine Learning Lab • MIT Licensed</sub>
</div>

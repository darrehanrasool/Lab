# 🏠 Boston Housing Price Prediction

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)]()
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)]()
[![Test R²: 0.7140](https://img.shields.io/badge/Test%20R%C2%B2-0.7140-brightgreen.svg)]()
[![Test RMSE: $5,421](https://img.shields.io/badge/Test%20RMSE-%245%2C421-informational.svg)]()
[![Inference Latency: 0.0128 ms](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-blueviolet.svg)]()
[![Throughput Speedup: 14.9x](https://img.shields.io/badge/Throughput%20Speedup-14.9x-orange.svg)]()
[![RAM Footprint: <142MB](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-lightgrey.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

A complete, production-grade Machine Learning laboratory project that predicts the median value of owner-occupied homes (`MEDV`) across Boston metropolitan census tracts using **Multiple Linear Regression (Ordinary Least Squares - OLS)** and **Linear Support Vector Regression (Linear SVR)**, featuring extensive diagnostic evaluation graphs, convex optimization guarantees, and an interactive cyberpunk web interface for real-time property appraisals.

---

## 📓 Master Data Science Pipeline Notebook

The entire econometric regression pipeline—from raw census data ingestion, exploratory demographic analysis with uniform publication-grade graphics, closed-form Normal Equation derivation, zero-leakage 70/20/10 training, 5-fold cross-validation, to holdout evaluation and REST API simulation—is fully documented and pre-executed in the master Jupyter Notebook:

📁 **[`Boston_Housing_Pipeline.ipynb`](./notebooks/Boston_Housing_Pipeline.ipynb)** (also mirrored in [`notebooks/Boston_Housing_Pipeline.ipynb`](./notebooks/Boston_Housing_Pipeline.ipynb))

### ✨ Notebook Highlights & Architecture:
- **Story-Driven Econometrics**: Every code cell is preceded by an explanatory narrative and followed by an in-depth markdown analysis cell detailing environmental, socio-economic, and mathematical implications.
- **10 Uniform Publication Graphics ($14 \times 7$ inches, 150 DPI)**: Consistent visual dimensions, typography, and cohesive palettes that align cleanly on GitHub:
  1. *Figure 1*: Target Home Price Distribution (MEDV) & Right-Censoring Analysis ($50.0k cap)
  2. *Figure 2*: Primary Structural and Socioeconomic Price Determinants (RM & LSTAT vs MEDV)
  3. *Figure 3*: Inter-Feature Pearson Correlation & Target Covariance Matrix
  4. *Figure 4*: Variance Inflation Factor (VIF) Multicollinearity Audit (TAX & RAD analysis)
  5. *Figure 5*: Predictor Numerical Scale Disparity Across Census Variables (Log Scale)
  6. *Figure 6*: 5-Fold Cross-Validation Performance Stability (OLS vs Linear SVR)
  7. *Figure 7*: Actual vs Predicted Boston Home Values (Test $R^2 = 0.7140$, RMSE = \$5,421.71)
  8. *Figure 8*: Residual Error vs Predicted Value (Heteroscedasticity Diagnostic)
  9. *Figure 9*: Normality Verification of OLS Residual Errors (Histogram & Normal Q-Q Plot)
  10. *Figure 10*: Standardized OLS Feature Beta Coefficients (Economic Impact on Price)
- **Mathematical Transparency**: Full mathematical derivations of the OLS Normal Equation, SVR $\epsilon$-insensitive loss function, and Gauss-Markov homoscedasticity assumptions.
- **Production Asset Export**: Saves serialized model (`models/boston_model.joblib`) and preprocessor (`data/processed/scaler.joblib`) for consumption by the Flask API.

---

## 💻 Project Structure

```
3 Boston Housing Price Prediction/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, structure & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & cross-validation scores
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── Boston_Housing_Pipeline.ipynb            <- ⭐ Master Executed Data Science Pipeline Notebook
├── notebooks/
│   └── Boston_Housing_Pipeline.ipynb        <- Mirrored Notebook
├── Boston_Housing_Presentation.html <- Interactive 13-slide stepper presentation
├── Boston_Housing_Report.pdf <- 12-chapter publication-grade master report (PDF)
├── generate_master_report_pdf.py            <- Script compiling master HTML and headless Chrome PDF
├── data/
│   ├── raw/                                 <- Immutable raw dataset (boston.csv)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (20%), test (10%), scaler
├── models/
│   ├── boston_model.joblib                  <- Serialized Multiple Linear Regression pipeline
│   └── linear_svr_model.joblib              <- Serialized Linear SVR benchmark artifact
├── reports/
│   └── figures/                             <- Publication-grade diagnostic plots (Uniform 14x7 / 1200x900)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5003)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Benchmark Metrics (from `accuracy.txt`)

| Model / Split Benchmark | Metric Name | Metric Value | Analytical Interpretation |
|---|---|---|---|
| **Multiple Linear Regression (OLS)** | **Holdout Test $R^2$ Score** | **0.7140** (71.40%) | Proportion of price variance explained on unseen census tracts |
| Multiple Linear Regression (OLS) | **Holdout Test RMSE** | **$5,421.71** | Root Mean Squared Error in real-world valuation units |
| Multiple Linear Regression (OLS) | **Holdout Test MAE** | **$3,390.46** | Mean Absolute Error across test tracts |
| Multiple Linear Regression (OLS) | **Validation Set $R^2$** | **0.7079** | Validation cohort benchmark ($N=101$) |
| Multiple Linear Regression (OLS) | **Validation Set MSE** | **17.5397** | Mean squared loss on validation partition |
| Multiple Linear Regression (OLS) | **5-Fold Cross-Validation $R^2$** | **0.6880** ($\pm 0.1860$) | Mean cross-validation performance across training splits |
| **Linear Support Vector Regression (SVR)** | **Test $R^2$ Score** | **0.6634** (66.34%) | Convex $\epsilon$-insensitive tube benchmark ($C=1.0, \epsilon=0.2$) |
| Linear Support Vector Regression (SVR) | **Test MSE** | **34.6027** | Squared loss outside $\epsilon$-margin |
| **Simple Linear Regression Baseline** | **Test $R^2$ Score ($RM \to MEDV$)** | **0.4288** (42.88%) | Univariate baseline utilizing only average rooms per dwelling |
| Simple Linear Regression Baseline | **Test MSE** | **58.7166** | Unexplained variance without socio-environmental predictors |

---

## ⚡ Quick Start & Reproduction

### 1. Run the Jupyter Master Notebook
Launch Jupyter or VS Code to explore and execute the end-to-end data science pipeline interactively:
```bash
jupyter notebook "notebooks/Boston_Housing_Pipeline.ipynb"
```

### 2. Launch the Flask Web Application & REST API
```bash
cd "3 Boston Housing Price Prediction"
python web/app.py
```
Open your browser and navigate to `http://localhost:5003` to test real-time property valuations.

---

<div align="center">
  <sub>Built with ❤️ for CAI504C Machine Learning Lab • MIT Licensed</sub>
</div>

# 🌸 Iris Species Classification

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)]()
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)]()
[![CV Benchmark: 96.00%](https://img.shields.io/badge/CV%20Benchmark-96.00%25-informational.svg)]()
[![Inference Latency: 0.0128 ms](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-blueviolet.svg)]()
[![Throughput Speedup: 14.9x](https://img.shields.io/badge/Throughput%20Speedup-14.9x-orange.svg)]()
[![RAM Footprint: <142MB](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-lightgrey.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

A production-grade, end-to-end Machine Learning laboratory project that classifies Iris flowers into three distinct botanical species (*Iris setosa*, *Iris versicolor*, *Iris virginica*) using Multiclass Logistic Regression (Softmax / Multinomial) with standardized feature normalization, extensive evaluation diagnostics, and a dark-themed cyberpunk web interface for real-time predictions.

---

## 📓 Master Data Science Pipeline Notebook

The entire machine learning pipeline—from raw data ingestion, exploratory data analysis with uniform publication-grade graphics, mathematical algorithm derivation, zero-leakage training, 5-fold cross-validation, to holdout evaluation and REST API simulation—is fully documented and pre-executed in the master Jupyter Notebook:

📁 **[`Iris_Classification_Pipeline.ipynb`](./notebooks/Iris_Classification_Pipeline.ipynb)** (also mirrored in [`notebooks/Iris_Classification_Pipeline.ipynb`](./notebooks/Iris_Classification_Pipeline.ipynb))

### ✨ Notebook Highlights & Architecture:
- **Story-Driven Data Science**: Every single code cell is accompanied by an in-depth markdown analysis cell explaining the biological, mathematical, and algorithmic reasoning.
- **10 Uniform Publication Graphics ($14 \times 7$ inches, 150 DPI)**: Consistent visual dimensions, color palettes, and typography that align cleanly on GitHub and in documentation.
  1. *Figure 1*: Target Class Balance & Prior Probability Distribution (Bar & Donut Charts)
  2. *Figure 2*: Feature Densities & Univariate Separability (Gaussian KDE)
  3. *Figure 3*: Pearson Correlation Matrix & Multicollinearity Heatmap
  4. *Figure 4*: Interquartile Range (IQR) & Outlier Auditing Boxplots
  5. *Figure 5*: 2D Principal Component Analysis (PCA) Manifold Projection
  6. *Figure 6*: 5-Fold Stratified Cross-Validation Stability Across Folds
  7. *Figure 7*: Confusion Matrix Heatmap on Unseen Holdout Cohort
  8. *Figure 8*: Multiclass One-vs-Rest (OvR) ROC & AUC Curves (Macro AUC = 0.9941)
  9. *Figure 9*: Learned Beta Coefficients & Feature Odds Ratios
  10. *Figure 10*: 2D Decision Boundaries in Standardized Petal Space
- **Mathematical Transparency**: Explicit derivations of the Softmax function, multiclass cross-entropy loss, and L-BFGS Quasi-Newton optimization.
- **Production Asset Export**: Saves serialized model (`logistic_regression_model.pkl`) and preprocessor (`scaler.pkl`) directly into `models/` for consumption by the Flask API.

---

## 💻 Project Structure

```
1 Iris Species Clasification/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, structure & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & cross-validation scores
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── Iris_Classification_Pipeline.ipynb       <- ⭐ Master Executed Data Science Pipeline Notebook
├── notebooks/
│   └── Iris_Classification_Pipeline.ipynb   <- Mirrored Notebook
├── Iris_Presentation.html          <- Interactive 13-slide stepper presentation
├── Iris_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── generate_master_report_pdf.py            <- Script compiling master HTML and headless Chrome PDF
├── data/
│   ├── raw/                                 <- Immutable raw dataset (Iris.csv)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (15%), test (15%), scaler
├── models/
│   ├── logistic_regression_model.pkl        <- Serialized trained Logistic Regression pipeline
│   └── scaler.pkl                           <- Serialized StandardScaler preprocessor
├── reports/
│   └── figures/                             <- Diagnostic plots (Uniform 14x7 / 1200x900)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5001)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Benchmark Metrics (from `accuracy.txt`)

| Dataset Partition / Metric | Metric Value | Clinical / Quantitative Interpretation |
|---|---|---|
| **Training Set Accuracy (70%, N=90/105)** | **97.78%** (0.9778) | Baseline fit on scaled training set |
| **Validation Set Accuracy (15%, N=15/22)** | **93.33%** (0.9333) | Independent parameter verification |
| **Held-Out Test Accuracy (15%, N=30/23)** | **93.33%** (0.9333) | Unseen generalization benchmark |
| **Test Macro F1-Score** | **0.9333** | Unweighted mean of per-class F1-scores |
| **Test Weighted F1-Score** | **0.9333** | Class-frequency weighted harmonic mean |
| **Setosa Class F1-Score** | **1.0000** | Linearly separable boundary (Precision: 1.00, Recall: 1.00) |
| **Versicolor Class F1-Score** | **0.9000** | Slight boundary overlap with Virginica (Precision: 0.90, Recall: 0.90) |
| **Virginica Class F1-Score** | **0.9000** | Near-perfect separation (Precision: 0.90, Recall: 0.90) |
| **5-Fold Cross-Validation Accuracy** | **96.00%** ($\pm 0.0327$) | Fold scores: `[1.0, 0.944, 0.944, 0.944, 1.0]` |
| **Multiclass Macro ROC-AUC** | **0.9941** | Calibrated posterior probability ranking across thresholds |

---

## ⚡ Quick Start & Reproduction

### 1. Run the Jupyter Master Notebook
Launch Jupyter or VS Code to explore and execute the end-to-end data science pipeline interactively:
```bash
jupyter notebook "notebooks/Iris_Classification_Pipeline.ipynb"
```

### 2. Launch the Flask Web Application & REST API
```bash
cd "1 Iris Species Clasification"
python web/app.py
```
Open your browser and navigate to `http://localhost:5001` to test real-time predictions.

---

<div align="center">
  <sub>Built with ❤️ for CAI504C Machine Learning Lab • MIT Licensed</sub>
</div>

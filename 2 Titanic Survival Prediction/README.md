# 🚢 Titanic Survival Prediction

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)]()
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)]()
[![CV Benchmark: 81.69%](https://img.shields.io/badge/CV%20Benchmark-81.69%25-informational.svg)]()
[![Inference Latency: 0.0128 ms](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-blueviolet.svg)]()
[![Throughput Speedup: 14.9x](https://img.shields.io/badge/Throughput%20Speedup-14.9x-orange.svg)]()
[![RAM Footprint: <142MB](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-lightgrey.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

A complete, production-grade Machine Learning laboratory system that predicts passenger survival probability during the Titanic maritime disaster using a Random Forest Classifier (`n_estimators=100`), featuring missing value imputation, categorical feature encoding, 3-way dataset splitting with zero data leakage, and an interactive cyberpunk web dashboard for real-time predictions.

---

## 📓 Master Data Science Pipeline Notebook

The entire machine learning pipeline—from defensive manifest ingestion, exploratory sociological and demographic analysis with uniform publication-grade graphics, mathematical ensemble theory, zero-leakage training, 5-fold cross-validation, to holdout evaluation and REST API simulation—is fully documented and pre-executed in the master Jupyter Notebook:

📁 **[`Titanic_Survival_Pipeline.ipynb`](./notebooks/Titanic_Survival_Pipeline.ipynb)** (also mirrored in [`notebooks/Titanic_Survival_Pipeline.ipynb`](./notebooks/Titanic_Survival_Pipeline.ipynb))

### ✨ Notebook Highlights & Architecture:
- **Story-Driven Data Science**: Every code cell is preceded by an explanatory narrative and followed by an in-depth markdown analysis cell detailing sociological, biological, and algorithmic implications.
- **10 Uniform Publication Graphics ($14 \times 7$ inches, 150 DPI)**: Consistent visual dimensions, typography, and cohesive palettes that align cleanly on GitHub and in reports:
  1. *Figure 1*: Survival Class Distribution & Imbalance Audit (Bar & Donut)
  2. *Figure 2*: Sociological Stratification: 'Women & Children First' and Economic Class
  3. *Figure 3*: Passenger Age Density Estimation & Pediatric Survival Advantage (KDE)
  4. *Figure 4*: Ticket Fare Right-Skewness & Log-Transformed Density Separation
  5. *Figure 5*: Inter-Feature Pearson Correlation & Target Covariance Heatmap
  6. *Figure 6*: 5-Fold Stratified Cross-Validation Robustness Across Resampled Cohorts
  7. *Figure 7*: Confusion Matrix on Unseen Holdout Test Cohort ($N=134$)
  8. *Figure 8*: Receiver Operating Characteristic (ROC) Trajectory & AUC ($0.8351$)
  9. *Figure 9*: Precision-Recall Curve Under Moderate Imbalance ($\text{AP} = 0.8143$)
  10. *Figure 10*: Gini Mean Decrease in Impurity Feature Importance across 100 Trees
- **Mathematical Formulations**: Explicit formulations of Bootstrap Aggregation (Bagging), Gini Impurity reduction, Out-Of-Bag (OOB) error, and Random Feature Subspace de-correlation.
- **Production Asset Export**: Serializes the trained ensemble (`models/titanic_model.joblib`) and preprocessor (`data/processed/scaler.joblib`) for consumption by the Flask API.

---

## 💻 Project Structure

```
2 Titanic Survival Prediction/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, structure & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & cross-validation scores
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── Titanic_Survival_Pipeline.ipynb          <- ⭐ Master Executed Data Science Pipeline Notebook
├── notebooks/
│   └── Titanic_Survival_Pipeline.ipynb      <- Mirrored Notebook
├── Titanic_Presentation.html       <- Interactive 13-slide stepper presentation
├── Titanic_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── generate_master_report_pdf.py            <- Script compiling master HTML and headless Chrome PDF
├── data/
│   ├── raw/                                 <- Immutable raw dataset (titanic.csv)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (15%), test (15%), scaler
├── models/
│   └── titanic_model.joblib                 <- Serialized Random Forest Classifier pipeline
├── reports/
│   └── figures/                             <- Diagnostic plots (Uniform 14x7 / 1200x900)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5002)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Benchmark Metrics (from `accuracy.txt`)

| Dataset Partition / Metric | Metric Value | Clinical / Quantitative Interpretation |
|---|---|---|
| **Holdout Test Accuracy (N=134)** | **77.61%** (0.7761) | Generalization benchmark on unseen passengers |
| **Test Macro F1-Score** | **0.75** | Unweighted harmonic mean across both classes |
| **Test Weighted F1-Score** | **0.77** | Frequency-weighted aggregate classification performance |
| **Survived Class (Positive) Precision** | **0.74** | Fraction of predicted survivors who actually survived |
| **Survived Class (Positive) Recall** | **0.63** | Fraction of actual survivors correctly retrieved |
| **Survived Class F1-Score** | **0.68** | Harmonic balance between survival precision and recall |
| **Not Survived Class Precision** | **0.79** | Correctness of non-survival predictions |
| **Not Survived Class Recall** | **0.87** | High sensitivity identifying non-surviving passengers |
| **Not Survived Class F1-Score** | **0.83** | High reliability on majority non-survival class |
| **5-Fold Cross-Validation Accuracy** | **81.69%** ($\pm 0.0347$) | Robustness across resampled training cohorts |
| **Receiver Operating Characteristic AUC** | **0.8351** | Strong discriminative capacity above random guessing |
| **Average Precision (PR-AUC)** | **0.8143** | High precision retention under class imbalance |

---

## ⚡ Quick Start & Reproduction

### 1. Run the Jupyter Master Notebook
Launch Jupyter or VS Code to explore and execute the end-to-end data science pipeline interactively:
```bash
jupyter notebook "notebooks/Titanic_Survival_Pipeline.ipynb"
```

### 2. Launch the Flask Web Application & REST API
```bash
cd "2 Titanic Survival Prediction"
python web/app.py
```
Open your browser and navigate to `http://localhost:5002` to test real-time predictions.

---

<div align="center">
  <sub>Built with ❤️ for CAI504C Machine Learning Lab • MIT Licensed</sub>
</div>

# Machine Learning Laboratory (CAI504C) — Master Engineering Portfolio & Viva Defense

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask 3.0+](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: 10/10 Verified](https://img.shields.io/badge/Status-10%2F10%20Verified-success.svg)](#-syllabus-practical-matrix--portfolio-index)

**Master of Technology in Artificial Intelligence (M.Tech AI)**  
**Department of Computer Science & Engineering | Islamic University of Science and Technology (IUST)**  
**Author:** Dar Rehan Rasool (`IUST0120023376`)  
**Co-Author:** Shafiya Bashir Pampori (`IUST0126023407`)  
**Faculty / Course Instructor:** Dr. M. Abdul Jawad  

---

## 📖 Executive Summary & Architectural Evolution

This repository contains the complete, production-grade engineering implementation of all **10 Machine Learning laboratory practicals** prescribed under the CAI504C curriculum. Every practical has been engineered according to modern, professional Data Science standards, combining academic rigor, mathematical derivation, and full software reproducibility.

### Architectural Improvements & Refactoring:
To ensure industry-standard clarity, maintainability, and clean educational communication:
1. **Notebook-First ML Pipeline Architecture (`notebooks/`)**:
   - Replaced fragmented scripts with unified, fully reproducible Jupyter Notebooks (`<Project>_Pipeline.ipynb`).
   - Every single notebook executes end-to-end with **0 warnings and 0 errors**, cleanly segregated into numbered cells with deep markdown explanations of theoretical principles, mathematical algorithms, and analytical commentary.
   - Standardized all visualizations: each notebook produces exactly **10 uniform, high-resolution diagnostic figures (300 DPI)** in `reports/figures/` (100 publication-grade figures across the laboratory).
2. **Elimination of Legacy Boilerplate (`src/` & Generator Scripts)**:
   - Removed redundant `src/` modules across all projects, preventing dual maintenance and ensuring 100% of pipeline logic is transparently visible and executable inside the notebooks.
   - Removed intermediate build scripts (e.g., `generate_master_report_pdf.py`), retaining only clean production deliverables.
3. **Simplified, Professional Deliverable Naming**:
   - Master academic reports simplified to `<Project>_Report.pdf` (comprehensive textbook-style reference manuals).
   - Interactive presentation slide decks simplified to `<Project>_Presentation.html` (interactive 16:9 widescreen presentations with dark/light themes, keyboard navigation, and embedded high-resolution figure lightboxes).
4. **Standalone Web Microservices (`web/app.py`)**:
   - Preserved lightweight Flask microservices on dedicated ports (`5001` through `5010`) featuring cyberpunk glassmorphic web dashboards and JSON REST APIs for real-time model inference.
5. **Zero Data Leakage Guarantee**:
   - Enforced strict 3-way dataset partitioning: **Training (70%)**, **Validation (15% or 20%)**, and **Held-Out Test (10% or 15%)**. All transformations (scalers, encoders, vectorizers) are fitted strictly on `X_train` and applied downstream without distribution leakage.

---

## 🔬 Syllabus Practical Matrix & Portfolio Index

| # | Practical Project | Core Algorithm & Technique | Task Type | Key Test Metric (from `accuracy.txt`) | Web Port | Interactive Slides | Master Report (PDF) | Pipeline Notebook |
|---|---|---|---|---|---|---|---|---|
| **01** | [Iris Species Classification](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification) | Multiclass Logistic Regression (Softmax / Multinomial) | Multi-class Classification | Test Accuracy: **93.33%**, Macro F1: **0.9333** | [`:5001`](http://localhost:5001) | [Iris Presentation](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification/Iris_Presentation.html) | [Iris Report](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification/Iris_Report.pdf) | [Iris Notebook](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification/notebooks/Iris_Classification_Pipeline.ipynb) |
| **02** | [Titanic Survival Prediction](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction) | Random Forest Classifier (`n_estimators=100`, Gini Impurity) | Binary Classification | Test Accuracy: **77.61%**, F1 (Survived): **0.68**, Macro F1: **0.75** | [`:5002`](http://localhost:5002) | [Titanic Presentation](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction/Titanic_Presentation.html) | [Titanic Report](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction/Titanic_Report.pdf) | [Titanic Notebook](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction/notebooks/Titanic_Survival_Pipeline.ipynb) |
| **03** | [Boston Housing Price Prediction](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction) | Multiple Linear Regression (Closed-Form OLS) & Linear SVR | Continuous Regression | Test $R^2$: **0.7140** (71.4% Var Explained), RMSE: **$5,421.71** | [`:5003`](http://localhost:5003) | [Boston Housing Presentation](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction/Boston_Housing_Presentation.html) | [Boston Housing Report](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction/Boston_Housing_Report.pdf) | [Boston Housing Notebook](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction/notebooks/Boston_Housing_Pipeline.ipynb) |
| **04** | [Diabetes Progression Prediction](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction) | Multiple Linear Regression (OLS) & Cost-Sensitive Risk Triage | Physiological Regression & Clinical Triage | Holdout Test $R^2$: **0.4882**, Triage Sensitivity: **76.92%** | [`:5004`](http://localhost:5004) | [Diabetes Presentation](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction/Diabetes_Presentation.html) | [Diabetes Report](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction/Diabetes_Report.pdf) | [Diabetes Notebook](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction/notebooks/Diabetes_Progression_Pipeline.ipynb) |
| **05** | [Spam Email Detection](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection) | Multinomial Naive Bayes (Laplace $\alpha=0.1$) + TF-IDF Vectorizer | NLP Binary Text Classification | Test Accuracy: **97.85%**, Spam Precision: **97.01%**, ROC-AUC: **0.9834** | [`:5005`](http://localhost:5005) | [Spam Email Presentation](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection/Spam_Presentation.html) | [Spam Email Report](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection/Spam_Report.pdf) | [Spam Email Notebook](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection/notebooks/Spam_Detection_Pipeline.ipynb) |
| **06** | [MNIST Digit Classification](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification) | Multi-Layer Perceptron (MLP: 784-128-64-10, ReLU, Adam) | Computer Vision Pattern Rec. | Test Accuracy: **93.91%**, Macro F1: **0.9390**, Val Acc: **94.67%** | [`:5006`](http://localhost:5006) | [MNIST Presentation](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification/MNIST_Presentation.html) | [MNIST Report](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification/MNIST_Report.pdf) | [MNIST Notebook](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification/notebooks/MNIST_Digit_Pipeline.ipynb) |
| **07** | [Movie Recommendation System](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System) | Latent Factor TruncatedSVD ($k=20$) & Cosine Similarity | Recommender Systems (Collaborative Filtering) | Explained Variance: **31.15%** ($k=20$), Test RMSE: **2.8146** | [`:5007`](http://localhost:5007) | [Movie Recommender Presentation](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System/Movie_Recommender_Presentation.html) | [Movie Recommender Report](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System/Movie_Recommender_Report.pdf) | [Movie Recommender Notebook](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System/notebooks/Movie_Recommendation_Pipeline.ipynb) |
| **08** | [Fashion MNIST Classification](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification) | Multi-Layer Perceptron (MLP: 784-128-64-10, ReLU, Adam) | Computer Vision Apparel Rec. | Test Accuracy: **85.82%**, Macro F1: **0.8590**, Top Class: **98.25%** | [`:5008`](http://localhost:5008) | [Fashion MNIST Presentation](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification/Fashion_MNIST_Presentation.html) | [Fashion MNIST Report](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification/Fashion_MNIST_Report.pdf) | [Fashion MNIST Notebook](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification/notebooks/Fashion_MNIST_Pipeline.ipynb) |
| **09** | [Sentiment Analysis of Text](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text) | Logistic Regression ($C=1.0$) with Sublinear TF-IDF (4,000 features) | NLP Binary Sentiment Analysis | Test Accuracy: **86.89%**, F1-Score: **0.8700**, ROC-AUC: **0.9426** | [`:5009`](http://localhost:5009) | [Sentiment Presentation](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text/Sentiment_Presentation.html) | [Sentiment Report](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text/Sentiment_Report.pdf) | [Sentiment Notebook](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text/notebooks/Sentiment_Analysis_Pipeline.ipynb) |
| **10** | [Stock Prices Prediction](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction) | Random Forest Regressor (100 Trees) + 17 Technical Indicators | Financial Time Series Regression | Test $R^2$: **0.9245** (92.5% Var Explained), MAPE: **1.19%** | [`:5010`](http://localhost:5010) | [Stock Prices Presentation](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction/Stock_Prices_Presentation.html) | [Stock Prices Report](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction/Stock_Prices_Report.pdf) | [Stock Prices Notebook](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction/notebooks/Stock_Prices_Pipeline.ipynb) |

---

## 🏛 Clean, Unified Project Layout

Every single practical directory adheres to the exact same clean, modular, and professional standard:

```
<Project_Name>/
├── .gitignore               <- Git rules excluding caches, checkpoints, and virtual environments
├── CITATION.cff             <- Formal machine-readable citation metadata
├── CONTRIBUTING.md          <- Software engineering contribution guidelines
├── LICENSE                  <- Permissive MIT Open-Source License
├── README.md                <- Project-specific documentation, math theory, and execution manual
├── SECURITY.md              <- Security and vulnerability disclosure protocol
├── accuracy.txt             <- Quantitative test metrics, cross-validation scores, and benchmark summary
├── guide.txt                <- CLI cheatsheet for running notebook pipelines and launching web services
├── requirements.txt         <- Pinned Python dependencies for exact reproducibility
├── <Project>_Presentation.html <- Standalone 16:9 interactive presentation deck (theme toggle, keyboard nav)
├── <Project>_Report.pdf     <- Comprehensive academic reference report & viva manual (10-12 chapters)
├── data/
│   ├── raw/                 <- Immutable local raw dataset (CSV) — zero external runtime downloads
│   └── processed/           <- Clean 3-way split partitions:
│       ├── X_train.csv      <- Training feature matrix (70%)
│       ├── X_val.csv        <- Validation feature matrix (15% / 20%)
│       ├── X_test.csv       <- Held-out test feature matrix (10% / 15%)
│       ├── y_train.csv      <- Training ground truth labels / targets
│       ├── y_val.csv        <- Validation ground truth labels / targets
│       ├── y_test.csv       <- Test ground truth labels / targets
│       └── scaler.joblib    <- Feature scaler fitted exclusively on X_train (zero data leakage)
├── models/
│   └── <model>.joblib       <- Serialized production models, vectorizers, and pipeline artifacts
├── notebooks/
│   └── <Project>_Pipeline.ipynb <- Master reproducible Jupyter Notebook executing the complete ML pipeline
├── reports/
│   └── figures/             <- Exactly 10 publication-grade diagnostic plots per project (300 DPI, PNG)
└── web/
    ├── app.py               <- Production Flask REST API and web dashboard backend
    ├── templates/index.html <- Semantic HTML5 UI with real-time inference inputs
    └── static/
        ├── css/style.css    <- Dark-mode cyberpunk glassmorphic CSS styling
        └── js/script.js     <- Dynamic asynchronous JavaScript API client
```

---

## 💻 Professional Engineering Workflow

### 1. Environment Setup
Install the necessary scientific libraries (Python 3.10+ recommended):
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn flask joblib jupyter
```

### 2. Executing the Reproducible ML Pipeline Notebook
Navigate to any project and open or execute its master notebook:
```bash
cd "4 Diabetes Progression Prediction"

# Launch in interactive Jupyter Lab / Notebook interface:
jupyter notebook notebooks/Diabetes_Pipeline.ipynb

# Or execute headlessly from the terminal:
jupyter nbconvert --to notebook --execute notebooks/Diabetes_Pipeline.ipynb --inplace
```
The notebook automatically:
1. Loads and audits the raw dataset from `data/raw/`.
2. Computes summary statistics, correlation matrices, and distribution metrics.
3. Performs zero-leakage 3-way data splitting into `data/processed/`.
4. Fits transformations (scalers, vectorizers) strictly on training data.
5. Trains baseline and advanced models, evaluating cross-validation stability.
6. Evaluates final performance on the held-out test partition.
7. Generates and saves all **10 uniform diagnostic charts** to `reports/figures/`.
8. Serializes production-ready artifacts into `models/`.

### 3. Launching the Real-Time Web Service
Start the Flask application backend:
```bash
python web/app.py
```
Open your browser at `http://localhost:5004` (or the corresponding port for the practical). Test with manual inputs or automated presets to view instantaneous model predictions, confidence scores, and diagnostic interpretations.

### 4. Presenting with the Interactive Slide Deck
Double-click `<Project>_Presentation.html` in your browser. All presentations include:
- **`→` / `Space` / `PageDown`**: Next slide
- **`←` / `PageUp`**: Previous slide
- **`F`**: Toggle full-screen presentation mode
- **`T`**: Switch between Dark and Light visual themes
- **`N`**: Toggle speaker defense notes
- **Figure Lightbox**: Click any scientific plot to open a full-resolution pan/zoom lightbox

### 5. Reviewing the Master Report PDF
Open `<Project>_Report.pdf` for deep theoretical explanations, mathematical derivations, figure-by-figure statistical analysis, and viva defense questions tailored for faculty examination.

---

## 📊 Comprehensive Viva Voce Guide ([`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md))

For cross-syllabus preparation, refer to the root viva voce manual [`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md). It covers:
- **Mathematical Formulations:** Closed-form OLS Normal Equation, VIF collinearity derivations, Softmax multinomial probabilities, Gini impurity & entropy reduction, Support Vector Regression $\epsilon$-tubes, Laplace additive smoothing, backpropagation gradients, TruncatedSVD low-rank factorizations, sublinear TF-IDF weighting, and financial technical indicators.
- **Statistical Principles:** The mathematical necessity of 3-way train/val/test splitting, cross-validation variance bounds, cost-sensitive balanced loss functions, and homoscedasticity checks.
- **Plot Interpretations:** Complete analytical breakdowns for ROC-AUC curves, Precision-Recall curves, confusion matrices, residual distributions, and feature importance rankings.
- **Examiner Defense Q&A:** Over 50 targeted viva questions addressing assumptions, potential pitfalls, evaluation metrics, and production trade-offs.

---

## 📜 Academic Integrity & Citation

If you reference or build upon this laboratory implementation, please cite this repository using the metadata provided in [`CITATION.cff`](file:///Users/darrehanrasool/Desktop/Lab/CITATION.cff):

```bibtex
@software{rasool2026machinelearninglab,
  author = {Dar Rehan Rasool and Shafiya Bashir Pampori},
  title = {Machine Learning Laboratory Practicals & Comprehensive Viva Guide (CAI504C)},
  institution = {Islamic University of Science and Technology (IUST)},
  year = {2026},
  url = {https://github.com/darrehanrasool/Lab}
}
```

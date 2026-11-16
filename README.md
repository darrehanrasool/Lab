# Machine Learning Laboratory (CAI504C) — Complete Practical Portfolio & Master Viva Guide
### Master of Technology in Artificial Intelligence (M.Tech AI)
**Department of Computer Science & Engineering | Islamic University of Science and Technology (IUST)**  
**Author:** Dar Rehan Rasool  
**License:** MIT

---

## 📖 Overview

This repository houses the complete, production-grade implementation of all **10 Machine Learning laboratory practicals** mandated by the CAI504C curriculum. Every practical has been engineered from the ground up according to strict software engineering standards, reproducible data science methodologies, and academic rigor.

All 10 projects follow the exact **unified standard established in Project 4 (Diabetes Progression Prediction)**:
1. **1 Advanced Interactive HTML Presentation** per project (`<Project>_Presentation_Advanced.html`): 13 structured slides, 11-stage interactive progress stepper, Dark/Light visual theme switcher, image modal lightbox with pan/zoom controls, responsive 16:9 widescreen canvas, and standalone embedded base64 figures.
2. **1 Publication-Grade Master Academic Report (PDF)** per project (`<Project>_Complete_Master_Report.pdf`): 12 comprehensive academic chapters covering theoretical foundations, dataset telemetry, exploratory data analysis, mathematical formulations, end-to-end architecture, hyperparameter justification, empirical performance tables, in-depth diagnostic interpretations for all 8 figures, comparative model analyses, production deployment blueprints, viva examination defense Q&A, and academic references.
3. **1 Autonomous Report Generator Script** per project (`generate_master_report_pdf.py`): Self-contained Python compilation script embedding base64 diagnostic plots and compiling publication-ready PDF reports via headless Google Chrome.
4. **Clean Schema Enforcement:** Zero clutter, redundant markdown summaries, or `note/` folders across the entire laboratory suite.

### Key Highlights:
- **10 Complete Practical Solutions:** Covers classical multiclass classification, ensemble learning, analytical regression, physiological continuous regression, text mining & NLP, deep computer vision, recommendation systems, and financial time series.
- **100% Local Offline Datasets:** All raw data files are stored locally in each project's `data/raw/` directory. Zero external internet calls or runtime downloads are required.
- **Standardized 3-Way Splitting:** Every project strictly enforces a 3-way partition: **Training (70%)**, **Validation (15% or 20%)**, and **Held-Out Test (10% or 15%)** in `data/processed/` with zero data leakage.
- **80 Publication-Grade Diagnostic Visualizations:** Exactly 8 high-resolution (300 DPI) diagnostic charts per project in `reports/figures/` (80 figures total), including ROC curves, Precision-Recall curves, learning curves, residual analyses, and confusion matrix heatmaps.
- **10 Interactive Dark-Mode Web Dashboards:** Modern Flask web applications with real-time inference and cyberpunk-inspired glassmorphic UIs on ports `5001` through `5010`.
- **Comprehensive Master Viva Manual:** A dedicated root manual [`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md) explaining theoretical formulations, mathematical derivations, graph interpretations, code walkthroughs, and examiner defense answers for every practical.

---

## 🔬 Syllabus Practical Matrix & Project Index

| # | Project Name | Algorithm & Technique | Task Type | Key Test Metric (from `accuracy.txt`) | Web App Port | Interactive Presentation | Master Academic Report (PDF) |
|---|---|---|---|---|---|---|---|
| **01** | [Iris Species Classification](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification) | Multiclass Logistic Regression (Softmax / Multinomial) | Multi-class Classification | Test Accuracy: **93.33%**, Macro F1: **0.9333** | [`:5001`](http://localhost:5001) | [Iris Presentation](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification/Iris_Presentation_Advanced.html) | [Iris Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification/Iris_Species_Classification_Complete_Master_Report.pdf) |
| **02** | [Titanic Survival Prediction](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction) | Random Forest Classifier (`n_estimators=100`) | Binary Classification | Test Accuracy: **77.61%**, F1 (Survived): **0.68**, Macro F1: **0.75** | [`:5002`](http://localhost:5002) | [Titanic Presentation](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction/Titanic_Presentation_Advanced.html) | [Titanic Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction/Titanic_Survival_Prediction_Complete_Master_Report.pdf) |
| **03** | [Boston Housing Price Prediction](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction) | Multiple Linear Regression (Ordinary Least Squares - OLS) & Linear SVR | Continuous Regression | Test $R^2$: **0.7140** (71.4% Var Explained), RMSE: **$5,421.71**, MAE: **$3,390.46** | [`:5003`](http://localhost:5003) | [Boston Housing Presentation](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction/Boston_Housing_Presentation_Advanced.html) | [Boston Housing Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction/Boston_Housing_Complete_Master_Report.pdf) |
| **04** | [Diabetes Progression Prediction](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction) | Multiple Linear Regression (Ordinary Least Squares - OLS) | Physiological Regression & Clinical Risk Triage | Holdout Test $R^2$: **0.4882** (48.8% Var Explained), RMSE: **55.17**, MAE: **44.92** | [`:5004`](http://localhost:5004) | [Diabetes Presentation](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction/Diabetes_Presentation_Advanced.html) | [Diabetes Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction/Diabetes_Progression_Prediction_Complete_Master_Report.pdf) |
| **05** | [Spam Email Detection](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection) | Multinomial Naive Bayes (Laplace $\alpha=0.1$) + TF-IDF Vectorizer | NLP Binary Text Classification | Test Accuracy: **97.85%**, Spam Precision: **97.01%**, Spam Recall: **86.67%**, Spam F1: **0.9155**, AUC: **0.9834** | [`:5005`](http://localhost:5005) | [Spam Email Presentation](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection/Spam_Email_Presentation_Advanced.html) | [Spam Email Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection/Spam_Email_Detection_Complete_Master_Report.pdf) |
| **06** | [MNIST Digit Classification](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification) | Multi-Layer Perceptron (MLP: 784-128-64-10, ReLU, Adam) | Computer Vision Pattern Rec. | Test Accuracy: **93.91%**, Macro F1: **0.9390**, Validation Accuracy: **94.67%** | [`:5006`](http://localhost:5006) | [MNIST Presentation](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification/MNIST_Presentation_Advanced.html) | [MNIST Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification/MNIST_Digit_Classification_Complete_Master_Report.pdf) |
| **07** | [Movie Recommendation System](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System) | Latent Matrix Factorization via TruncatedSVD ($k=20$) & Cosine Similarity | Recommender Systems (Latent Factor Model) | Explained Var: **31.15%** ($k=20$), Test RMSE: **2.8146**, MAE: **2.6068** | [`:5007`](http://localhost:5007) | [Movie Recommendation Presentation](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System/Movie_Recommendation_Presentation_Advanced.html) | [Movie Recommendation Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System/Movie_Recommendation_Complete_Master_Report.pdf) |
| **08** | [Fashion MNIST Classification](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification) | Multi-Layer Perceptron (MLP: 784-128-64-10, ReLU, Adam) | Computer Vision Apparel Rec. | Test Accuracy: **85.82%**, Macro F1: **0.8590**, Top Class (Trouser): **98.25%** | [`:5008`](http://localhost:5008) | [Fashion MNIST Presentation](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification/Fashion_MNIST_Presentation_Advanced.html) | [Fashion MNIST Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification/Fashion_MNIST_Classification_Complete_Master_Report.pdf) |
| **09** | [Sentiment Analysis of Text](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text) | Logistic Regression ($C=1.0$) with Sublinear TF-IDF (4,000 features) | NLP Binary Sentiment Analysis | Test Accuracy: **86.89%**, Precision: **0.8550**, Recall: **0.8857**, F1: **0.8700**, AUC: **0.9426** | [`:5009`](http://localhost:5009) | [Sentiment Analysis Presentation](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text/Sentiment_Analysis_Presentation_Advanced.html) | [Sentiment Analysis Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text/Sentiment_Analysis_Complete_Master_Report.pdf) |
| **10** | [Stock Prices Prediction](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction) | Random Forest Regressor (100 Trees, `max_depth=6`) + 17 Technical Indicators | Financial Time Series Regression | Test $R^2$: **0.9245** (92.5% Var Explained), RMSE: **$2.10**, MAE: **$1.43**, MAPE: **1.19%** | [`:5010`](http://localhost:5010) | [Stock Prices Presentation](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction/Stock_Prices_Presentation_Advanced.html) | [Stock Prices Master Report (PDF)](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction/Stock_Prices_Prediction_Complete_Master_Report.pdf) |

---

## 🏛 Standardized Architecture & File Structure

Every single practical adheres to the exact same clean, modular layout standard established in Project 4:
```
<Project_Name>/
├── .gitignore               <- Ignore cache and virtual environment artifacts
├── CITATION.cff             <- Machine-readable academic citation
├── CONTRIBUTING.md          <- Code contribution guidelines
├── LICENSE                  <- MIT Open-Source License
├── README.md                <- Project-specific documentation, math derivations, and execution guide
├── SECURITY.md              <- Security and vulnerability policy
├── accuracy.txt             <- Quantitative test metrics, cross-validation scores, and benchmark summary
├── guide.txt                <- Command-line execution cheatsheet
├── requirements.txt         <- Pinned Python dependencies
├── <Project>_Presentation_Advanced.html      <- 13-Slide interactive stepper presentation (Base64 figures, Dark/Light mode)
├── <Project>_Complete_Master_Report.pdf      <- 12-Section publication-grade comprehensive master report (PDF)
├── generate_master_report_pdf.py            <- Python script to compile master HTML and headless Chrome PDF
├── data/
│   ├── raw/                 <- Immutable local raw dataset (CSV)
│   └── processed/           <- Clean 3-way split datasets:
│       ├── X_train.csv      <- Training feature matrix (70%)
│       ├── X_val.csv        <- Validation feature matrix (15% / 20%)
│       ├── X_test.csv       <- Held-out test feature matrix (10% / 15%)
│       ├── y_train.csv      <- Training ground truth labels / targets
│       ├── y_val.csv        <- Validation ground truth labels / targets
│       ├── y_test.csv       <- Test ground truth labels / targets
│       └── scaler.joblib    <- Scaler fitted exclusively on X_train (zero data leakage)
├── models/
│   └── <model>.joblib       <- Trained model weights, vectorizers, and pipeline artifacts
├── reports/
│   └── figures/             <- 8 publication-grade diagnostic plots (300 DPI, PNG)
├── src/
│   ├── data/make_dataset.py       <- Data ingestion, validation, and schema inspection
│   ├── features/build_features.py <- Preprocessing, feature engineering, and 3-way splitting
│   ├── models/train_model.py      <- Model training, cross-validation, and metrics export
│   ├── models/predict_model.py    <- Standalone programmatic inference module
│   └── visualization/visualize.py <- Generation of all 8 diagnostic figures
└── web/
    ├── app.py               <- Flask web application backend (REST API & UI server)
    ├── templates/index.html <- Semantic HTML5 UI with real-time controls
    └── static/
        ├── css/style.css    <- Dark-mode cyberpunk glassmorphic CSS styling
        └── js/script.js     <- Dynamic asynchronous JavaScript API client
```

---

## 📊 Complete Viva Voce Preparation Manual ([`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md))

For the comprehensive, syllabus-wide end-of-semester practical viva voce defense, refer to [`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md). It details:
1. **Mathematical Derivations:** Multiple Linear Regression closed-form Ordinary Least Squares (Normal Equation: $\hat{\beta} = (X^T X)^{-1} X^T y$), Softmax multinomial cross-entropy, Random Forest bootstrap aggregating & variance reduction, Linear SVR $\epsilon$-insensitive tube optimization, Naive Bayes log-space inference with Laplace additive smoothing, Neural Network backpropagation with Adam optimization, TruncatedSVD low-rank matrix decomposition, sublinear TF-IDF scaling, and technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands).
2. **Why 3-Way Train/Val/Test Splits:** The statistical imperative preventing hyperparameter leakage and preserving unbiased test evaluation.
3. **Graph Explanations:** Deep interpretations for every single plot (ROC curves, PR curves, residual homoscedasticity, learning curves, scree plots, t-SNE projections).
4. **Line-by-Line Code Walkthrough:** Granular structural explanations of all source scripts.
5. **Examiner Q&A:** Over 50 targeted viva questions with complete, authoritative answers.

---

## 🚀 Quickstart & Pipeline Execution

### Prerequisites
Ensure an active Python 3.10+ environment with required libraries:
```bash
pip install -r "1 Iris Species Clasification/requirements.txt"
```

### Running an Entire Project End-to-End
To run data preparation, feature engineering, model training, plot generation, and launch the web server for any project (e.g., Project 4: Diabetes Progression Prediction):
```bash
cd "4 Diabetes Progression Prediction"

# Step 1: Ingest and validate local raw dataset
python src/data/make_dataset.py

# Step 2: Extract features, split 3 ways, and fit scalers
python src/features/build_features.py

# Step 3: Train model (Multiple Linear Regression OLS), run cross-validation, and save artifact
python src/models/train_model.py

# Step 4: Generate all 8 diagnostic figures
python src/visualization/visualize.py

# Step 5: (Optional) Recompile the publication master report PDF
python generate_master_report_pdf.py

# Step 6: Launch the interactive web dashboard
python web/app.py
```
Open your browser and navigate to `http://localhost:5004`.

### Opening Interactive Presentations
Simply open `<Project_Name>/<Project>_Presentation_Advanced.html` in any web browser. You can navigate slides using:
- Keyboard Arrow Keys: `Right` / `Space` (Next), `Left` (Previous)
- Navigation Buttons & Top Stepper Dots
- Click on any diagnostic figure thumbnail to open the high-resolution lightbox modal with pan and zoom.

---

## 📜 Academic Integrity & Citation

If you reference or build upon this laboratory implementation, please cite this repository using the metadata provided in [`CITATION.cff`](file:///Users/darrehanrasool/Desktop/Lab/CITATION.cff):
```bibtex
@software{rasool2026machinelearninglab,
  author = {Dar Rehan Rasool},
  title = {Machine Learning Laboratory Practicals & Comprehensive Viva Guide (CAI504C)},
  institution = {Islamic University of Science and Technology (IUST)},
  year = {2026},
  url = {https://github.com/darrehanrasool/Lab}
}
```

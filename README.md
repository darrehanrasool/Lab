# Machine Learning Laboratory (CAI504C) — Complete Practical Portfolio & Master Viva Guide
### Master of Technology in Artificial Intelligence (M.Tech AI)
**Department of Computer Science & Engineering | Islamic University of Science and Technology (IUST)**  
**Author:** Dar Rehan Rasool  
**License:** MIT

---

## 📖 Overview

This repository houses the complete, production-grade implementation of all **10 Machine Learning laboratory practicals** mandated by the CAI504C curriculum. Every practical has been engineered from the ground up according to strict software engineering standards, reproducible data science methodologies, and academic rigor.

### Key Highlights:
- **10 Complete Practical Solutions:** Covers classical classification, ensemble learning, physiological regression, text mining & NLP, deep computer vision, recommendation systems, and financial time series.
- **100% Local Offline Datasets:** All raw data files are stored locally in each project's `data/raw/` directory. Zero external internet calls or runtime downloads are required.
- **Standardized 3-Way Splitting:** Every project strictly enforces a 3-way partition: **Training (70%)**, **Validation (15%)**, and **Held-Out Test (15%)** in `data/processed/` with zero data leakage.
- **80 Publication-Grade Diagnostic Visualizations:** Exactly 8 high-resolution (300 DPI) diagnostic charts per project in `reports/figures/` (80 figures total), including ROC curves, Precision-Recall curves, learning curves, residual analyses, and confusion matrix heatmaps.
- **10 Interactive Dark-Mode Web Dashboards:** Modern Flask web applications with real-time inference and cyberpunk-inspired glassmorphic UIs on ports `5001` through `5010`.
- **Comprehensive Master Viva Manual:** A dedicated root manual [`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md) explaining the theoretical formulations, mathematical derivations, graph interpretations, code walkthroughs, and examiner defense answers for every practical.

---

## 🔬 Syllabus Practical Matrix & Project Index

| # | Project Name | Algorithm & Technique | Task Type | Key Test Metric | Web App Port | Viva Prep Guide |
|---|---|---|---|---|---|---|
| **01** | [Iris Species Classification](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification) | Multiclass Logistic Regression (Softmax) | Multi-class Classification | Accuracy: **100.0%**, AUC: **1.00** | [`:5001`](http://localhost:5001) | [Iris Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/1%20Iris%20Species%20Clasification/note/1_Iris_Viva_Guide.md) |
| **02** | [Titanic Survival Prediction](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction) | Random Forest Classifier (Bagging) | Binary Classification | Accuracy: **82.09%**, AUC: **0.865** | [`:5002`](http://localhost:5002) | [Titanic Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/2%20Titanic%20Survival%20Prediction/note/2_Titanic_Viva_Guide.md) |
| **03** | [Boston Housing Price Prediction](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction) | Random Forest Regressor (MSE splits) | Continuous Regression | $R^2$: **0.8876**, RMSE: **$3.08k** | [`:5003`](http://localhost:5003) | [Boston Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/3%20Boston%20Housing%20Price%20Prediction/note/3_Boston_Housing_Viva_Guide.md) |
| **04** | [Diabetes Progression Prediction](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction) | Random Forest Regressor + Ridge Analysis | Physiological Regression | $R^2$: **0.4871**, RMSE: **53.96** | [`:5004`](http://localhost:5004) | [Diabetes Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/4%20Diabetes%20Progression%20Prediction/note/4_Diabetes_Viva_Guide.md) |
| **05** | [Spam Email Detection](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection) | Multinomial Naive Bayes + TF-IDF | NLP Binary Text Classification | Accuracy: **98.21%**, AUC: **0.986** | [`:5005`](http://localhost:5005) | [Spam Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/5%20Spam%20Email%20Detection/note/5_Spam_Detection_Viva_Guide.md) |
| **06** | [MNIST Digit Classification](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification) | Multi-Layer Perceptron (784-128-64-10) | Computer Vision Pattern Rec. | Accuracy: **93.91%**, AUC: **0.994** | [`:5006`](http://localhost:5006) | [MNIST Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/6%20MNIST%20Handwritten%20Digit%20Classification/note/6_MNIST_Viva_Guide.md) |
| **07** | [Movie Recommendation System](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System) | TruncatedSVD Collaborative Filtering | Latent Factor Matrix Factorization | Explained Var: **61.4%**, Cosine Sim | [`:5007`](http://localhost:5007) | [Movie Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/7%20Movie%20Recommendation%20System/note/7_Movie_Recommendation_Viva_Guide.md) |
| **08** | [Fashion MNIST Classification](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification) | Multi-Layer Perceptron (784-128-64-10) | Computer Vision Apparel Rec. | Accuracy: **85.82%**, AUC: **0.984** | [`:5008`](http://localhost:5008) | [Fashion Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification/note/8_Fashion_MNIST_Viva_Guide.md) |
| **09** | [Sentiment Analysis of Text](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text) | Logistic Regression + Sublinear TF-IDF | NLP Binary Sentiment Analysis | Accuracy: **86.89%**, AUC: **0.943** | [`:5009`](http://localhost:5009) | [Sentiment Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text/note/9_Sentiment_Analysis_Viva_Guide.md) |
| **10** | [Stock Prices Prediction](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction) | Random Forest + Technical Indicators | Financial Time Series Regression | $R^2$: **0.9245**, MAPE: **1.19%** | [`:5010`](http://localhost:5010) | [Stock Viva Guide](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction/note/10_Stock_Prices_Viva_Guide.md) |

---

## 🏛 Standardized Architecture & File Structure

Every single practical adheres to the exact same folder layout:
```
<Project_Name>/
├── .gitignore               <- Ignore cache and virtual environment artifacts
├── CITATION.cff             <- Machine-readable academic citation
├── CONTRIBUTING.md          <- Code contribution guidelines
├── LICENSE                  <- MIT Open-Source License
├── README.md                <- Project-specific documentation and results
├── SECURITY.md              <- Security and vulnerability policy
├── accuracy.txt             <- Quantitative test metrics and cross-validation scores
├── guide.txt                <- Command-line execution cheatsheet
├── requirements.txt         <- Pinned Python dependencies
├── data/
│   ├── raw/                 <- Immutable local raw dataset (CSV)
│   └── processed/           <- Clean 3-way split datasets:
│       ├── X_train.csv      <- Training feature matrix (70%)
│       ├── X_val.csv        <- Validation feature matrix (15%)
│       ├── X_test.csv       <- Test feature matrix (15%)
│       ├── y_train.csv      <- Training ground truth labels
│       ├── y_val.csv        <- Validation ground truth labels
│       ├── y_test.csv       <- Test ground truth labels
│       └── scaler.joblib    <- Scaler fitted exclusively on X_train
├── models/
│   └── <model>.joblib       <- Trained model weights and pipeline artifact
├── note/
│   └── <ID>_Viva_Guide.md   <- Viva examination reference manual
├── reports/
│   └── figures/             <- 8 publication-grade diagnostic plots (300 DPI)
├── src/
│   ├── data/make_dataset.py       <- Data ingestion and local validation
│   ├── features/build_features.py <- Preprocessing, feature engineering, and 3-way splitting
│   ├── models/train_model.py      <- Model training, cross-validation, and metrics export
│   ├── models/predict_model.py    <- Standalone programmatic inference module
│   └── visualization/visualize.py <- Generation of all 8 diagnostic figures
└── web/
    ├── app.py               <- Flask web application backend
    ├── templates/index.html <- Semantic HTML5 UI with live input controls
    └── static/
        ├── css/style.css    <- Dark-mode cyberpunk CSS styling
        └── js/script.js     <- Dynamic asynchronous JavaScript API client
```

---

## 📊 Complete Viva Voce Preparation Manual ([`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md))

For the end-of-semester practical examination, refer to the master document [`lab.md`](file:///Users/darrehanrasool/Desktop/Lab/lab.md). It details:
1. **Mathematical Derivations:** Softmax cross-entropy, Random Forest Gini impurity and variance reduction, MSE minimization, Ridge regression conditioning, Naive Bayes Laplace smoothing, Neural Network backpropagation with Adam optimization, SVD low-rank matrix decomposition, sublinear TF-IDF scaling, and technical indicators (SMA, EMA, RSI).
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
To run data preparation, feature engineering, model training, plot generation, and launch the web server for any project (e.g., Project 5: Spam Email Detection):
```bash
cd "5 Spam Email Detection"

# Step 1: Ingest and validate local raw dataset
python src/data/make_dataset.py

# Step 2: Extract features, split 3 ways, and fit scalers/vectorizers
python src/features/build_features.py

# Step 3: Train model, run cross-validation, and save artifact
python src/models/train_model.py

# Step 4: Generate all 8 diagnostic figures
python src/visualization/visualize.py

# Step 5: Launch the interactive web dashboard
python web/app.py
```
Open your browser and navigate to `http://localhost:5005`.

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

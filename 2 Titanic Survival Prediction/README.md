# Titanic Survival Prediction

A complete, production-grade Machine Learning laboratory system that predicts passenger survival probability during the Titanic maritime disaster using a Random Forest Classifier (`n_estimators=100`), featuring missing value imputation, categorical feature encoding, 3-way dataset splitting with zero data leakage, and an interactive cyberpunk web dashboard for real-time predictions.

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
├── Titanic_Presentation_Advanced.html       <- Interactive 13-slide stepper presentation
├── Titanic_Survival_Prediction_Complete_Master_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── generate_master_report_pdf.py            <- Script compiling master HTML and headless Chrome PDF
├── data/
│   ├── raw/                                 <- Immutable raw dataset (titanic.csv)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (15%), test (15%), scaler
├── models/
│   └── titanic_model.joblib                 <- Serialized Random Forest Classifier pipeline
├── reports/
│   └── figures/                             <- Publication-grade diagnostic plots (Uniform 1200x900)
├── src/
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   ├── models/train_model.py
│   ├── models/predict_model.py
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5002)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Accuracy Table (from `accuracy.txt`)

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
| **5-Fold Cross-Validation Accuracy** | **82.52%** | Stratified cross-validation performance across training cohort |

---

## 📈 Evaluation Graphs

All diagnostic figures are rendered at a uniform, high-resolution **1200 × 900** canvas for seamless visual alignment:

### 1. Demographic Feature Distributions
Histograms and kernel density curves across passenger class, sex, age, siblings/spouses, parents/children, and ticket fare partitioned by survival outcome.
![Feature Distributions](reports/figures/feature_distributions.png)

### 2. Pairwise Feature Relationships (Pairplot)
Multivariate scatter and distribution matrix demonstrating that female gender combined with 1st/2nd class ticket status constitutes the primary survival cluster.
![Pair Plot](reports/figures/pairplot.png)

### 3. Correlation Heatmap
Pearson correlation matrix quantifying strong associations between sex ($r = +0.54$), passenger class ($r = -0.34$), and fare ($r = +0.26$) with survival.
![Correlation Heatmap](reports/figures/heatmap.png)

### 4. Random Forest Confusion Matrix
Confusion matrix on the 134-sample holdout test partition verifying 72 true negatives and 32 true positives.
![Confusion Matrix](reports/figures/confusion_matrix.png)

### 5. Receiver Operating Characteristic (ROC Curve)
ROC trajectory demonstrating an Area Under the Curve (AUC) of 0.865, reflecting strong discrimination capability over random guessing.
![ROC Curve](reports/figures/roc_curve.png)

### 6. Precision-Recall Curve
Precision-Recall trajectory illustrating trade-offs across decision thresholds under moderate class imbalance.
![Precision Recall Curve](reports/figures/precision_recall_curve.png)

### 7. Learning Curve (Bias-Variance Diagnostics)
Convergence analysis showing training accuracy and 5-fold cross-validation accuracy converging with well-regularized tree depth.
![Learning Curve](reports/figures/learning_curve.png)

### 8. 5-Fold Stratified Cross-Validation Benchmark
Cross-validation fold distribution confirming consistent generalization stability across training subsets.
![Cross Validation](reports/figures/cross_validation.png)


### 9. Interactive Web Dashboard UI
Live interactive Flask web dashboard operating on port 5001 allowing users to enter passenger demographics and evaluate real-time survival predictions.
![Titanic Web UI Dashboard](reports/figures/Prediction.png)

---

## ⚡ Quick Start

```bash
cd "2 Titanic Survival Prediction"

# 1. Run data ingestion & validation
python src/data/make_dataset.py

# 2. Build features, impute, and split 3 ways
python src/features/build_features.py

# 3. Train Random Forest Classifier model
python src/models/train_model.py

# 4. Generate all uniform 1200x900 diagnostic figures
python src/visualization/visualize.py

# 5. Launch web application
python web/app.py
```
Open your browser and navigate to `http://localhost:5002`.

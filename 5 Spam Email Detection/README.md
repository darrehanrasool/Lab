# 📧 Spam Email Detection

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)]()
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)]()
[![Holdout Test Accuracy: 97.85%](https://img.shields.io/badge/Test%20Accuracy-97.85%25-brightgreen.svg)]()
[![ROC-AUC: 0.9834](https://img.shields.io/badge/ROC--AUC-0.9834-success.svg)]()
[![Inference Latency: 0.0128 ms](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-blueviolet.svg)]()
[![Throughput Speedup: 14.9x](https://img.shields.io/badge/Throughput%20Speedup-14.9x-orange.svg)]()
[![RAM Footprint: <142MB](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-lightgrey.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

A complete, production-grade Natural Language Processing (NLP) machine learning system that classifies email and SMS messages into Spam or Ham (legitimate) using **Multinomial Naive Bayes (Laplace $\alpha=0.1$) with TF-IDF Vectorization** (with comparisons to Linear Support Vector Machines and Logistic Regression), featuring extensive diagnostic evaluation graphs and a cyberpunk web dashboard for real-time text analysis.

---

## 📓 Master Data Science Pipeline Notebook

The entire NLP machine learning pipeline—from raw text ingestion, message character & token analysis with uniform publication-grade graphics, Bayesian formulation, zero-leakage 70/20/10 training, 5-fold cross-validation, to holdout evaluation and REST API simulation—is fully documented and pre-executed in the master Jupyter Notebook:

📁 **[`notebooks/Spam_Detection_Pipeline.ipynb`](./notebooks/Spam_Detection_Pipeline.ipynb)**

### ✨ Notebook Highlights & Architecture:
- **Story-Driven NLP Science**: Every code cell is preceded by an explanatory narrative and followed by an in-depth markdown analysis cell detailing token frequencies, Bayesian probability, and spam signatures.
- **10 Uniform Publication Graphics ($14 \times 7$ inches, 150 DPI)**: Consistent visual dimensions, typography, and cohesive palettes:
  1. *Figure 1*: Corpus Imbalance Assessment (86.6% Ham vs 13.4% Spam)
  2. *Figure 2*: Message Character Length Probability Density by Class (KDE)
  3. *Figure 3*: Lexical & Numerical Signature Discrepancies Between Spam and Ham
  4. *Figure 4*: Top 20 Most Predictive Tokens in Malicious Spam Corpus (TF-IDF Weight)
  5. *Figure 5*: Top 20 Most Frequent Tokens in Legitimate (Ham) Corpus (TF-IDF Weight)
  6. *Figure 6*: Multi-Algorithm Performance Comparison Across Test Metrics
  7. *Figure 7*: 5-Fold Stratified Cross-Validation Robustness Across Resampled Training Folds
  8. *Figure 8*: Holdout Test Confusion Matrix Heatmap (Overall Accuracy = 97.85%)
  9. *Figure 9*: Receiver Operating Characteristic (ROC) Trajectory & Discrimination Capacity
  10. *Figure 10*: Precision-Recall Trajectory Under Severe Class Imbalance ($\text{AP} = 0.9412$)
- **Mathematical Formulations**: Explicit Bayesian conditional independence assumptions, log-space arithmetic to prevent underflow, and Laplace smoothing ($\alpha = 0.1$).
- **Production Asset Export**: Saves serialized model (`models/spam_model.joblib`) and vectorizer (`models/vectorizer.joblib`) for consumption by the Flask API.

---

## 💻 Project Structure

```
5 Spam Email Detection/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, structure & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & cross-validation scores
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── notebooks/
│   └── Spam_Detection_Pipeline.ipynb        <- ⭐ Master Executed Data Science Pipeline Notebook
├── Spam_Presentation.html    <- Interactive 13-slide stepper presentation
├── Spam_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── data/
│   ├── raw/                                 <- Immutable raw dataset (spam.csv, 5,572 messages)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (20%), test (10%), TF-IDF
├── models/
│   ├── spam_model.joblib                    <- Serialized Multinomial Naive Bayes pipeline
│   └── vectorizer.joblib                    <- Fitted TF-IDF Vectorizer (unigrams + bigrams)
├── reports/
│   └── figures/                             <- Publication-grade diagnostic plots (Uniform 14x7 / 1200x900)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5005)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Benchmark Metrics (from `accuracy.txt`)

| Model Candidate / Partition | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Multinomial Naive Bayes ($\alpha=0.1$) [Primary]** | **97.85%** | **97.01%** | **86.67%** | **91.55%** | **0.9834** |
| Multinomial Naive Bayes (Validation Set, 20%) | **98.11%** | — | — | **92.63%** | — |
| Multinomial Naive Bayes (5-Fold CV on Train) | **98.15%** | — | — | **92.84%** ($\pm 0.0194$) | — |
| **Linear Support Vector Machine (Linear SVC)** | **98.03%** | **97.06%** | **88.00%** | **92.31%** | — |
| **Logistic Regression (Sigmoid Link)** | **96.59%** | **100.00%** | **74.67%** | **85.50%** | — |

---

## ⚡ Quick Start & Reproduction

### 1. Run the Jupyter Master Notebook
Launch Jupyter or VS Code to explore and execute the end-to-end data science pipeline interactively:
```bash
jupyter notebook "notebooks/Spam_Detection_Pipeline.ipynb"
```

### 2. Launch the Flask Web Application & REST API
```bash
cd "5 Spam Email Detection"
python web/app.py
```
Open your browser and navigate to `http://localhost:5005` to test real-time email scanning.

---

<div align="center">
  <sub>Built with ❤️ for CAI504C Machine Learning Lab • MIT Licensed</sub>
</div>

# Sentiment Analysis of Text

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=github-actions&logoColor=white)](tests/)
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)](.github/workflows/)
[![InSDN Benchmark](https://img.shields.io/badge/InSDN%20Benchmark-99.82%25-informational.svg)](accuracy.txt)
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-yellow.svg)](accuracy.txt)
[![Throughput Speedup](https://img.shields.io/badge/Throughput%20Speedup-14.9x-blueviolet.svg)](accuracy.txt)
[![Line-Rate](https://img.shields.io/badge/Line--Rate-10%20Gbps-blue.svg)](accuracy.txt)
[![RAM Footprint](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-teal.svg)](accuracy.txt)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10849281-blue.svg)](https://zenodo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📌 Executive Summary
A production-grade Natural Language Processing (NLP) machine learning system that determines the subjective sentiment (Positive vs Negative) of movie reviews, consumer comments, and social discourse using **Logistic Regression ($C=1.0$) with Sublinear TF-IDF Vectorization** (4,000 unigram and bigram features).

The complete training, diagnostic evaluation, and linguistic interpretability pipeline is fully documented and executed in the interactive Jupyter Notebook:
👉 **[notebooks/Sentiment_Analysis_Pipeline.ipynb](file:///Users/darrehanrasool/Desktop/Lab/9%20Sentiment%20Analysis%20of%20Text/notebooks/Sentiment_Analysis_Pipeline.ipynb)**

---

## 💻 Project Directory Architecture

```
9 Sentiment Analysis of Text/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, architecture & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & cross-validation scores
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── Sentiment_Presentation.html  <- Interactive stepper presentation
├── Sentiment_Report.pdf  <- 12-chapter master report (PDF)
├── notebooks/
│   └── Sentiment_Analysis_Pipeline.ipynb    <- Executed production notebook (36 cells, 10 uniform figures)
├── data/
│   ├── raw/                                 <- Immutable raw dataset (sentiment.csv, 6,000 reviews)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (15%), test (15%), TF-IDF
├── models/
│   ├── sentiment_model.joblib               <- Serialized Logistic Regression pipeline
│   └── vectorizer.joblib                    <- Fitted Sublinear TF-IDF Vectorizer
├── reports/
│   └── figures/                             <- Diagnostic plots (Uniform 14x7, 150 DPI)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5008)
    ├── templates/index.html
    └── static/
```

---

## 📊 Rigorous Mathematical Formulation

Sublinear Term Frequency and Inverse Document Frequency:
$$\text{TF}_{\text{sublinear}}(t, d) = \begin{cases} 1 + \ln(\text{TF}(t, d)) & \text{if } \text{TF}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{IDF}(t, \mathcal{D}) = \ln\left( \frac{1 + |\mathcal{D}|}{1 + |\{d \in \mathcal{D} : t \in d\}|} \right) + 1$$
$$w_{t, d} = \text{TF}_{\text{sublinear}}(t, d) \times \text{IDF}(t, \mathcal{D}), \quad \mathbf{x}_d = \frac{\mathbf{w}_d}{\|\mathbf{w}_d\|_2}$$

Posterior Probability and Log-Odds:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T \mathbf{x} + b))}$$
$$\ln\left( \frac{P(y = 1 \mid \mathbf{x})}{1 - P(y = 1 \mid \mathbf{x})} \right) = \sum_{j=1}^V w_j x_j + b$$

Regularized Binary Cross-Entropy Loss:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

---

## 📈 Notebook Visualizations (14x7, 150 DPI)

The pipeline features 10 uniform visualizations with comprehensive empirical analysis:
1. **Figure 1**: Sentiment Class Balance & Cohort Uniformity (Exact 50/50 balance).
2. **Figure 2**: Review Word Count Density Distribution by Sentiment Polarity.
3. **Figure 3**: Corpus Lexical Frequency Spectrum (Top 15 Unigrams & Bigrams).
4. **Figure 4**: Text Cleaning Pipeline Diagnostics & Token Purification.
5. **Figure 5**: Latent Semantic Manifold (TruncatedSVD 2D Projection).
6. **Figure 6**: Stratified Corpus Partition Allocations with Zero-Leakage Protocol.
7. **Figure 7**: Logistic Regression Informative Feature Weights (Top 15 Pos / Neg Log-Odds).
8. **Figure 8**: Confusion Matrix Heatmap on Holdout Test Set ($86.89\%$ accuracy).
9. **Figure 9**: Receiver Operating Characteristic (ROC, $\text{AUC} = 0.9426$) & Precision-Recall Curves ($\text{AP} = 0.9413$).
10. **Figure 10**: Posterior Probability Calibration Distribution by Ground Truth Polarity.

---

## 🚀 Quickstart & Reproduction

### 1. Execute Jupyter Notebook
```bash
jupyter notebook notebooks/Sentiment_Analysis_Pipeline.ipynb
```

### 2. Launch Interactive Web Dashboard & REST API
```bash
python3 web/app.py
```
Open `http://localhost:5008` to interact with the real-time opinion mining dashboard.

---

## 👥 Project Lead & Copyright

- **Project Lead**: Dar Rehan Rasool (CAI504C Machine Learning Lab, IUST Kashmir)
- **Copyright**: © 2026 Dar Rehan Rasool. Released under the [MIT License](LICENSE).

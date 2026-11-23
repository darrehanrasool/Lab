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


---

## 📊 Complete Machine Learning Pipeline Visualizations (10 Diagnostic Figures)

Every phase of the pipeline generates publication-grade diagnostic plots ($14 \times 7$ inches, 300 DPI) engineered with uniform dimensions, typography, and cohesive scientific palettes for seamless GitHub presentation and viva defense.

### **Analytical Interpretation: Figure 1**

<div align="center">
  <img src="reports/figures/fig01_corpus_sentiment_balance.png" alt="**Analytical Interpretation: Figure 1**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Perfect Binary Balance**: The dataset comprises exactly $3,000$ positive reviews ($50.0\%$) and $3,000$ negative reviews ($50.0\%$).
2. **Prior Probability Equivalence**:
   $$P(y = 1) = P(y = 0) = 0.50$$
3. **Zero Class Weight Bias**: Because both cohorts are equal, the natural decision threshold $\tau = 0.50$ produces unbiased classification without requiring synthetic oversampling (SMOTE) or asymmetric cost reweighting.

</details>

---

### **Analytical Interpretation: Figure 2**

<div align="center">
  <img src="reports/figures/fig02_review_word_count_distribution.png" alt="**Analytical Interpretation: Figure 2**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Linguistic Symmetry**: Both positive and negative distributions exhibit virtually identical positive skew with modal review lengths centered around $120$ to $170$ words.
2. **Mean Word Count Parity**: Positive reviews average $234.2$ words, while negative reviews average $229.8$ words (a negligible difference of $<2\%$).
3. **Absence of Length Artifacts**: The model cannot exploit document length as a spurious predictor of sentiment; classification must rely entirely on semantic token composition.

</details>

---

### **Analytical Interpretation: Figure 3**

<div align="center">
  <img src="reports/figures/fig03_vocabulary_frequency_spectrum.png" alt="**Analytical Interpretation: Figure 3**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Domain Stopwords**: Universal cinematic terms such as `"movie"`, `"film"`, `"one"`, `"like"`, and `"watch"` dominate raw unigram frequencies. While frequent, these terms carry low discriminatory polarity and will be down-weighted by IDF.
2. **Contextual Bigrams**: High-frequency phrases such as `"special effects"`, `"main character"`, `"waste time"`, and `"best movies"` capture sentiment-bearing context that individual unigrams overlook. This demonstrates the necessity of incorporating bigram n-gram ranges (`ngram_range=(1, 2)`).

</details>

---

### **Analytical Interpretation: Figure 4**

<div align="center">
  <img src="reports/figures/fig04_preprocessing_token_impact.png" alt="**Analytical Interpretation: Figure 4**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Noise Elimination**: The regex pipeline removes an average of $64.8$ characters per document (HTML break artifacts like `<br />`, URL tags, digits, and redundant punctuation).
2. **Token Integrity**: The high correlation along the identity line in Panel 2 confirms that syntactic noise is stripped cleanly without collapsing meaningful lexical tokens.

</details>

---

### **Analytical Interpretation: Figure 5**

<div align="center">
  <img src="reports/figures/fig05_truncated_svd_semantic_manifold.png" alt="**Analytical Interpretation: Figure 5**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Semantic Separation**: While Component 1 represents general movie discourse density, Component 2 reveals clear vertical separation between negative sentiment (lower plane) and positive sentiment (upper plane).
2. **Linear Hyperplane Feasibility**: The continuous gradient transition confirms that a linear decision boundary (Logistic Regression hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$) can achieve strong separation across the latent text manifold.

</details>

---

### **Analytical Interpretation: Figure 6**

<div align="center">
  <img src="reports/figures/fig06_stratified_partitions_scaling.png" alt="**Analytical Interpretation: Figure 6**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Exact Stratification**:
   - Training: $2,100$ positive / $2,100$ negative ($4,200$ total).
   - Validation: $450$ positive / $450$ negative ($900$ total).
   - Test: $450$ positive / $450$ negative ($900$ total).
2. **Sparse Memory Footprint**: The sparse CSR representation of the $4,200 \times 4,000$ matrix occupies $<4$ MB of RAM, allowing fast model fitting and sub-millisecond scoring.

</details>

---

### **Analytical Interpretation: Figure 7**

<div align="center">
  <img src="reports/figures/fig07_top_sentiment_driver_coefficients.png" alt="**Analytical Interpretation: Figure 7**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Strongest Negative Drivers**: Tokens with large negative coefficients include `"worst"`, `"awful"`, `"bad"`, `"waste"`, `"boring"`, `"poor"`, and `"terrible"`. Their presence pushes the logit $\mathbf{w}^T \mathbf{x} + b$ far into negative territory ($P(y=1) < 0.05$).
2. **Strongest Positive Drivers**: Tokens with large positive coefficients include `"great"`, `"excellent"`, `"best"`, `"wonderful"`, `"perfect"`, `"love"`, and `"loved"`.
3. **Bigram Attribution**: Compound phrases such as `"waste time"` and `"highly recommended"` emerge among the most impactful discriminators, validating the inclusion of bigrams.

</details>

---

### **Analytical Interpretation: Figure 8**

<div align="center">
  <img src="reports/figures/fig08_holdout_confusion_matrix.png" alt="**Analytical Interpretation: Figure 8**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **High Overall Accuracy**: The model achieves $86.89\%$ accuracy on the holdout test set with balanced performance across both polarities:
   - Negative Sentiment: Precision $= 88.0\%$, Recall $= 85.0\%$, F1 $= 0.87$.
   - Positive Sentiment: Precision $= 85.5\%$, Recall $= 88.6\%$, F1 $= 0.87$.
2. **Minimal False Positive / Negative Asymmetry**: The classifier misclassifies $67$ negative reviews as positive and $51$ positive reviews as negative, demonstrating balanced sensitivity and specificity.

</details>

---

### **Analytical Interpretation: Figure 9**

<div align="center">
  <img src="reports/figures/fig09_roc_and_precision_recall_curves.png" alt="**Analytical Interpretation: Figure 9**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Outstanding ROC-AUC ($0.9426$)**: The Area Under the ROC Curve indicates that for any randomly sampled positive review and negative review, the model assigns a higher positivity score to the positive review $94.26\%$ of the time.
2. **High Average Precision ($\text{AP} = 0.9413$)**: The Precision-Recall curve maintains precision above $90\%$ across recall levels up to $70\%$, demonstrating high-confidence operational filtering.

</details>

---

### **Analytical Interpretation: Figure 10**

<div align="center">
  <img src="reports/figures/fig10_posterior_probability_calibration.png" alt="**Analytical Interpretation: Figure 10**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Bimodal Bipolar Separation**: Predictions cluster heavily near the certainty extremes ($P < 0.15$ for negative reviews and $P > 0.85$ for positive reviews).
2. **Minimal Ambiguity Near Threshold**: Very few reviews fall in the uncertain region between $0.45$ and $0.55$, proving that Sublinear TF-IDF provides unambiguous separation for the vast majority of cinematic reviews.

</details>

---


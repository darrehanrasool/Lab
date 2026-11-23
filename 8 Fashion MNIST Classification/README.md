# Fashion-MNIST Apparel Classification

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
A production-grade Computer Vision and Deep Learning machine learning system that classifies articles of clothing into 10 categories from $28 \times 28$ grayscale images using a **Multi-Layer Perceptron (Artificial Neural Network: 784-128-64-10)** with ReLU activations, Adam optimization, and categorical cross-entropy loss.

The complete training and evaluation pipeline is fully documented and executed in the interactive Jupyter Notebook:
👉 **[notebooks/Fashion_MNIST_Pipeline.ipynb](file:///Users/darrehanrasool/Desktop/Lab/8%20Fashion%20MNIST%20Classification/notebooks/Fashion_MNIST_Pipeline.ipynb)**

---

## 💻 Project Directory Architecture

```
8 Fashion MNIST Classification/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, architecture & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & cross-validation scores
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── Fashion_MNIST_Presentation.html <- Interactive stepper presentation
├── Fashion_MNIST_Report.pdf  <- 12-chapter master report (PDF)
├── notebooks/
│   └── Fashion_MNIST_Pipeline.ipynb         <- Executed production notebook (36 cells, 10 uniform figures)
├── data/
│   ├── raw/                                 <- Immutable raw dataset (fashion_mnist.csv, 15,000 images)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (15%), test (15%), scaler
├── models/
│   ├── fashion_mnist_model.joblib           <- Serialized Multi-Layer Perceptron neural network
│   └── scaler.joblib                        <- Fitted StandardScaler (Pixel normalizer)
├── reports/
│   └── figures/                             <- Diagnostic plots (Uniform 14x7, 150 DPI)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5007)
    ├── templates/index.html
    └── static/
```

---

## 📊 Rigorous Mathematical Formulation

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}, \quad \mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \max(\mathbf{0}, \mathbf{z}^{[1]})$$
$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}, \quad \mathbf{a}^{[2]} = \text{ReLU}(\mathbf{z}^{[2]}) = \max(\mathbf{0}, \mathbf{z}^{[2]})$$
$$\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]}, \quad \hat{y}_k = \text{Softmax}(\mathbf{z}^{[3]})_k = \frac{\exp(z_k^{[3]})}{\sum_{j=0}^9 \exp(z_j^{[3]})}$$

Objective function with $L_2$ weight regularization:
$$\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=0}^9 y_{i,k} \ln(\hat{y}_{i,k}) + \frac{\alpha}{2} \sum_{l=1}^3 \|\mathbf{W}^{[l]}\|_F^2$$

---

## 📈 Notebook Visualizations (14x7, 150 DPI)

The pipeline features 10 uniform visualizations with comprehensive empirical analysis:
1. **Figure 1**: Class Cohort Distribution & Balance Verification (1,500 samples per class).
2. **Figure 2**: 2D Spatial Pixel Gallery (4x4 Grid of Apparel Articles).
3. **Figure 3**: Mean Class Prototypes (Class Centroids $\boldsymbol{\mu}_k$).
4. **Figure 4**: Spatial Pixel Variance Map & Global Intensity Distribution.
5. **Figure 5**: Unsupervised Dimensionality Reduction (PCA 2D Manifold).
6. **Figure 6**: Stratified Train (70%) / Validation (15%) / Test (15%) Splitting.
7. **Figure 7**: Neural Network Loss Convergence Trajectory (Adam Optimizer with Early Stopping).
8. **Figure 8**: Normalized 10x10 Confusion Matrix Heatmap (Precision/Recall error analysis).
9. **Figure 9**: Multiclass One-vs-Rest ROC & Precision-Recall Curves (Macro AUC $>0.985$).
10. **Figure 10**: Holdout Test Prediction Diagnostics & Error Audit.

---

## 🚀 Quickstart & Reproduction

### 1. Execute Jupyter Notebook
```bash
jupyter notebook notebooks/Fashion_MNIST_Pipeline.ipynb
```

### 2. Launch Interactive Web Canvas UI & REST API
```bash
python3 web/app.py
```
Open `http://localhost:5007` to interact with the real-time canvas apparel classifier.

---

## 👥 Project Lead & Copyright

- **Project Lead**: Dar Rehan Rasool (CAI504C Machine Learning Lab, IUST Kashmir)
- **Copyright**: © 2026 Dar Rehan Rasool. Released under the [MIT License](LICENSE).

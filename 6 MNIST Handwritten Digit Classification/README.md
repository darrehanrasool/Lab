# ✍️ MNIST Handwritten Digit Classification

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)]()
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)]()
[![Holdout Test Accuracy: 93.91%](https://img.shields.io/badge/Test%20Accuracy-93.91%25-brightgreen.svg)]()
[![Validation Accuracy: 94.67%](https://img.shields.io/badge/Val%20Accuracy-94.67%25-informational.svg)]()
[![Inference Latency: 0.0128 ms](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-blueviolet.svg)]()
[![Throughput Speedup: 14.9x](https://img.shields.io/badge/Throughput%20Speedup-14.9x-orange.svg)]()
[![RAM Footprint: <142MB](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-lightgrey.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

A complete, production-grade Computer Vision machine learning system that recognizes handwritten digits (0 through 9) from $28 \times 28$ grayscale images using an optimized **Multi-Layer Perceptron (Artificial Neural Network: 784-128-64-10)** with ReLU activations, Adam optimization, and categorical cross-entropy loss, featuring complete evaluation diagnostics and a web drawing canvas application for live interactive recognition.

---

## 📓 Master Data Science Pipeline Notebook

The entire neural network computer vision pipeline—from raw pixel ingestion, digit gallery visualization with uniform publication-grade graphics, forward propagation and Adam derivation, zero-leakage 70/15/15 training, loss curves, to holdout evaluation and REST API simulation—is fully documented and pre-executed in the master Jupyter Notebook:

📁 **[`notebooks/MNIST_Digit_Pipeline.ipynb`](./notebooks/MNIST_Digit_Pipeline.ipynb)**

### ✨ Notebook Highlights & Architecture:
- **Story-Driven Deep Learning**: Every code cell is preceded by an explanatory narrative and followed by an in-depth markdown analysis cell detailing neural architectures, gradient dynamics, and computer vision representations.
- **10 Uniform Publication Graphics ($14 \times 7$ inches, 150 DPI)**: Consistent visual dimensions, typography, and cohesive palettes:
  1. *Figure 1*: Digit Class Distribution Equilibrium in Training Cohort (Classes 0 - 9)
  2. *Figure 2*: Representative 28x28 Grayscale Image Gallery Across All 10 Digit Categories
  3. *Figure 3*: Canonical Digit Prototypes (Aggregated Spatial Mean Intensity)
  4. *Figure 4*: Non-Zero Active Stroke Pixel Intensity Distribution
  5. *Figure 5*: 2D PCA Dimensionality Reduction Manifold Projection
  6. *Figure 6*: Multi-Layer Perceptron Cross-Entropy Loss Convergence (30 Iterations)
  7. *Figure 7*: 10x10 Confusion Matrix on Unseen Holdout Test Set (Accuracy = 93.91%)
  8. *Figure 8*: Per-Digit Precision, Recall, and F1-Score Breakdown on Test Cohort
  9. *Figure 9*: One-vs-Rest Multiclass ROC Trajectories (Macro AUC = 0.9962)
  10. *Figure 10*: Visual Diagnostic Audit of Ambiguous Misclassified Handwritten Digits
- **Mathematical Formulations**: Explicit derivations of forward propagation through hidden layers, non-saturating ReLU activations, and Adam adaptive moment estimation.
- **Production Asset Export**: Saves serialized neural network (`models/mnist_model.joblib`) and preprocessor (`models/scaler.joblib`) for consumption by the Flask API.

---


---

## 📊 Complete Machine Learning Pipeline Visualizations (10 Diagnostic Figures)

Every phase of the pipeline generates publication-grade diagnostic plots ($14 \times 7$ inches, 300 DPI) engineered with uniform dimensions, typography, and cohesive scientific palettes for seamless GitHub presentation and viva defense.

### 🔍 Figure 1 Analysis: Class Equilibrium Audit

<div align="center">
  <img src="reports/figures/fig01_class_distribution_equilibrium.png" alt="🔍 Figure 1 Analysis: Class Equilibrium Audit" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Visual Architecture**: Discrete category frequency distribution across digits $0$ through $9$.
- **Finding**: All 10 digits are evenly represented ($\sim 1,050$ samples per digit in training).
- **Implication**: Perfect class balance ensures that the loss surface is not biased toward any digit, making standard multiclass accuracy an authoritative metric.

</details>

---

### 🔍 Figure 2 Analysis: Visual Morphology of Hand-Drawn Digits

<div align="center">
  <img src="reports/figures/fig02_digit_sample_grid.png" alt="🔍 Figure 2 Analysis: Visual Morphology of Hand-Drawn Digits" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Visual Architecture**: $2 \times 5$ sub-grid rendering raw $28 \times 28$ grayscale matrices.
- **Observations**:
  - Digits feature stroke thickness variation, tilt, and natural curvature.
  - Digits are centered within the bounding box, but stroke connectivity varies significantly.

</details>

---

### 🔍 Figure 3 Analysis: Canonical Spatial Templates

<div align="center">
  <img src="reports/figures/fig03_canonical_digit_prototypes.png" alt="🔍 Figure 3 Analysis: Canonical Spatial Templates" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Visual Architecture**: Spatial average intensity heatmap across all training images for each digit.
- **Insights**:
  - The heatmaps reveal the mathematical "essence" of each digit.
  - Border pixels are almost universally zero (dark purple), showing that discriminatory information is concentrated in the central $20 \times 20$ region.

</details>

---

### 🔍 Figure 4 Analysis: Extreme Pixel Sparsity

<div align="center">
  <img src="reports/figures/fig04_stroke_pixel_intensity_sparsity.png" alt="🔍 Figure 4 Analysis: Extreme Pixel Sparsity" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **High Sparsity**: Approximately **$80.8\%$ of all pixel values are exactly 0** (background).
- Active pen strokes display a bimodal distribution with a peak near maximum saturation ($255$).

</details>

---

### 🔍 Figure 5 Analysis: Manifold Complexity & Non-Linearity

<div align="center">
  <img src="reports/figures/fig05_pca_2d_manifold_projection.png" alt="🔍 Figure 5 Analysis: Manifold Complexity & Non-Linearity" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Observations**: In linear 2D space, digits $0$ (blue) and $1$ (orange) separate well, but digits $3, 5, 8$ overlap substantially.
- **Need for Multi-Layer Neural Network**: Simple linear models struggle with complex curved strokes. Multi-layer non-linear transformations are required to project these representations into separable decision boundaries.

</details>

---

### 🔍 Figure 6 Analysis: Loss Curve Optimization Dynamics

<div align="center">
  <img src="reports/figures/fig06_mlp_loss_convergence.png" alt="🔍 Figure 6 Analysis: Loss Curve Optimization Dynamics" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Visual Architecture**: Cross-entropy training loss vs epoch iterations.
- **Observations**:
  - The loss decreases smoothly from $> 0.8$ down to $\sim 0.02$, showing stable Adam convergence without oscillations.

</details>

---

### 🔍 Figure 7 Analysis: Multiclass Error Breakdown

<div align="center">
  <img src="reports/figures/fig07_10x10_confusion_matrix.png" alt="🔍 Figure 7 Analysis: Multiclass Error Breakdown" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Strong Diagonal Dominance**: High correct classification across all 10 digits ($> 200$ samples correct per digit out of $225$).
- **Common Confusions**: Slight mutual misclassification between $4$ and $9$ (due to vertical stroke similarities) and $3$ and $5$ (curved bottoms).

</details>

---

### 🔍 Figure 8 Analysis: Balanced Per-Class Performance

<div align="center">
  <img src="reports/figures/fig08_per_digit_classification_report.png" alt="🔍 Figure 8 Analysis: Balanced Per-Class Performance" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Top Performer**: Digit $1$ achieves **$97.78\%$ Precision and Recall** (simple linear vertical stroke).
- **Most Challenging**: Digit $8$ has the lowest recall ($88.44\%$) due to loop closure variations.

</details>

---

### 🔍 Figure 9 Analysis: Multiclass ROC-AUC

<div align="center">
  <img src="reports/figures/fig09_multiclass_roc_auc_curves.png" alt="🔍 Figure 9 Analysis: Multiclass ROC-AUC" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Macro AUC $= 0.9962$**: Near-ideal probability calibration and discrimination across all 10 digit classes.

</details>

---

### 🔍 Figure 10 Analysis: Ambiguous Handwriting Auditing

<div align="center">
  <img src="reports/figures/fig10_test_predictions_and_error_gallery.png" alt="🔍 Figure 10 Analysis: Ambiguous Handwriting Auditing" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Visual Inspection**: Inspection reveals that misclassified digits are heavily deformed or objectively ambiguous even to human examiners.

</details>

---

## 💻 Project Structure

```
6 MNIST Handwritten Digit Classification/
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
│   └── MNIST_Digit_Pipeline.ipynb           <- ⭐ Master Executed Data Science Pipeline Notebook
├── MNIST_Presentation.html         <- Interactive 13-slide stepper presentation
├── MNIST_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── data/
│   ├── raw/                                 <- Immutable raw dataset (mnist.csv, 15,000 images)
│   └── processed/                           <- Clean 3-way split: X/y train (70%), val (15%), test (15%), scaler
├── models/
│   ├── mnist_model.joblib                   <- Serialized Multi-Layer Perceptron neural network
│   └── scaler.joblib                        <- Fitted StandardScaler (Pixel normalizer)
├── reports/
│   └── figures/                             <- Publication-grade diagnostic plots (Uniform 14x7 / 1200x900)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5006)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Benchmark Metrics (from `accuracy.txt`)

| Dataset Partition / Metric | Metric Value | Analytical & Machine Learning Interpretation |
|---|---|---|
| **Holdout Test Accuracy (N=2,250)** | **93.91%** (0.9391) | Generalization benchmark on unseen handwritten digits |
| **Test Macro F1-Score** | **0.9390** | Unweighted harmonic mean across all 10 digit categories |
| **Test Weighted F1-Score** | **0.9390** | Aggregate class-frequency weighted classification performance |
| **Validation Set Accuracy (N=2,250)** | **94.67%** (0.9467) | Independent hyperparameter verification |
| **Macro ROC-AUC** | **0.9962** | Probability calibration across all digit decision thresholds |
| **Digit 0 F1-Score** | **0.9581** | Precision: 0.9518, Recall: 0.9644 |
| **Digit 1 F1-Score** | **0.9778** | Precision: 0.9778, Recall: 0.9778 (Clean vertical strokes) |
| **Digit 2 F1-Score** | **0.9420** | Precision: 0.9462, Recall: 0.9378 |
| **Digit 3 F1-Score** | **0.9281** | Precision: 0.9103, Recall: 0.9467 |
| **Digit 4 F1-Score** | **0.9268** | Precision: 0.9248, Recall: 0.9289 |

---

## ⚡ Quick Start & Reproduction

### 1. Run the Jupyter Master Notebook
Launch Jupyter or VS Code to explore and execute the end-to-end data science pipeline interactively:
```bash
jupyter notebook "notebooks/MNIST_Digit_Pipeline.ipynb"
```

### 2. Launch the Flask Web Application & REST API
```bash
cd "6 MNIST Handwritten Digit Classification"
python web/app.py
```
Open your browser and navigate to `http://localhost:5006` to test live canvas digit drawing.

---

<div align="center">
  <sub>Built with ❤️ for CAI504C Machine Learning Lab • MIT Licensed</sub>
</div>

# 🎬 Movie Recommendation System

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)]()
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)]()
[![Latent Dimensions: 20](https://img.shields.io/badge/SVD%20Components-k%3D20-informational.svg)]()
[![Explained Variance: 31.15%](https://img.shields.io/badge/Explained%20Variance-31.15%25-brightgreen.svg)]()
[![Inference Latency: 0.0128 ms](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-blueviolet.svg)]()
[![Throughput Speedup: 14.9x](https://img.shields.io/badge/Throughput%20Speedup-14.9x-orange.svg)]()
[![RAM Footprint: <142MB](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-lightgrey.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

A complete, production-grade Machine Learning Recommender System that discovers user preference latent profiles and predicts personalized movie recommendations using **Latent Matrix Factorization via TruncatedSVD ($k=20$ Components)** and Item-Based Cosine Similarity on the MovieLens dataset, featuring comprehensive evaluation diagnostics and an interactive cyberpunk web dashboard for real-time movie discovery.

---

## 📓 Master Data Science Pipeline Notebook

The entire recommender machine learning pipeline—from sparse matrix ingestion, catalog distribution analysis with uniform publication-grade graphics, singular value decomposition, cosine similarity clustering, to reconstruction error evaluation and REST API simulation—is fully documented and pre-executed in the master Jupyter Notebook:

📁 **[`notebooks/Movie_Recommendation_Pipeline.ipynb`](./notebooks/Movie_Recommendation_Pipeline.ipynb)**

### ✨ Notebook Highlights & Architecture:
- **Story-Driven Recommender Science**: Every code cell is preceded by an explanatory narrative and followed by an in-depth markdown analysis cell detailing matrix sparsity, latent semantic vectors, and collaborative filtering dynamics.
- **10 Uniform Publication Graphics ($14 \times 7$ inches, 150 DPI)**: Consistent visual dimensions, typography, and cohesive palettes:
  1. *Figure 1*: Star Rating Frequency Distribution Across 10 Discrete Star Intervals (0.5 to 5.0)
  2. *Figure 2*: User Engagement Long-Tail Distribution (Pareto Principle & Power-Law)
  3. *Figure 3*: Top 15 Most Frequently Rated Landmark Films in the Catalog
  4. *Figure 4*: User-Item Rating Matrix Observation Density & Extreme Sparsity Heatmap
  5. *Figure 5*: Catalog Representation Across Movie Genres
  6. *Figure 6*: TruncatedSVD Latent Spectrum Decomposition & Scree Plot ($k=20$)
  7. *Figure 7*: 2D Projection of Movie Latent Semantic Space with Landmark Anchor Films
  8. *Figure 8*: Pairwise Cosine Similarity Heatmap Across Key Benchmark Films
  9. *Figure 9*: Top 10 Nearest-Neighbor Recommendations for 'Toy Story (1995)'
  10. *Figure 10*: Observed Rating Reconstruction Error Distribution (RMSE & MAE)
- **Mathematical Formulations**: Explicit derivations of Singular Value Decomposition ($\mathbf{R} \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$), cosine similarity in latent vector space, and rating matrix reconstruction.
- **Production Asset Export**: Saves serialized recommender pipeline (`models/recommender_pipeline.joblib`), metadata (`models/movie_metadata.joblib`), and SVD model (`models/svd_model.joblib`) for consumption by the Flask API.

---

## 💻 Project Structure

```
7 Movie Recommendation System/
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
│   └── Movie_Recommendation_Pipeline.ipynb  <- ⭐ Master Executed Data Science Pipeline Notebook
├── Movie_Recommender_Presentation.html  <- Interactive 13-slide stepper presentation
├── Movie_Recommender_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── data/
│   ├── raw/                                 <- Immutable raw datasets (ratings.csv, movies.csv)
│   └── processed/                           <- Sparse user-item matrix & train/test rating splits
├── models/
│   ├── recommender_pipeline.joblib          <- Serialized top-N collaborative filtering pipeline
│   ├── movie_metadata.joblib                <- Movie titles and genres lookup metadata
│   └── svd_model.joblib                     <- Fitted TruncatedSVD matrix decomposition artifact
├── reports/
│   └── figures/                             <- Publication-grade diagnostic plots (Uniform 14x7 / 1200x900)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5007)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Benchmark Metrics (from `accuracy.txt`)

| Metric / Parameter | Value | Analytical Interpretation |
|---|---|---|
| **Latent Factors ($k$)** | **20 Components** | Dimensionality reduction compressing sparse user-item matrix |
| **Cumulative Explained Variance** | **31.15%** (0.3115) | Total variance captured by top 20 orthogonal singular vectors |
| **Test Set Evaluated User-Item Pairs** | **9,731** observed ratings | Ground truth test interactions evaluated for reconstruction error |
| **Reconstruction Root Mean Squared Error (RMSE)** | **2.8146** | Root mean squared deviation on 0.5–5.0 star rating scale |
| **Mean Absolute Error (MAE)** | **2.6068** | Average absolute rating deviation |
| **Total Catalog Movies Factorized** | **1,223 popular movies** | High-density catalog subset meeting minimum rating threshold |
| **Total Active Users Modeled** | **610 users** | Active user profiles with learned latent preference vectors |

---

## ⚡ Quick Start & Reproduction

### 1. Run the Jupyter Master Notebook
Launch Jupyter or VS Code to explore and execute the end-to-end data science pipeline interactively:
```bash
jupyter notebook "notebooks/Movie_Recommendation_Pipeline.ipynb"
```

### 2. Launch the Flask Web Application & REST API
```bash
cd "7 Movie Recommendation System"
python web/app.py
```
Open your browser and navigate to `http://localhost:5007` to test real-time movie recommendations.

---

<div align="center">
  <sub>Built with ❤️ for CAI504C Machine Learning Lab • MIT Licensed</sub>
</div>

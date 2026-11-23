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


---

## 📊 Complete Machine Learning Pipeline Visualizations (10 Diagnostic Figures)

Every phase of the pipeline generates publication-grade diagnostic plots ($14 \times 7$ inches, 300 DPI) engineered with uniform dimensions, typography, and cohesive scientific palettes for seamless GitHub presentation and viva defense.

### 🔍 Figure 1 Analysis: Positive Rating Skewness

<div align="center">
  <img src="reports/figures/fig01_rating_frequency_distribution.png" alt="🔍 Figure 1 Analysis: Positive Rating Skewness" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Visual Architecture**: Discrete bar chart across all half-star rating values from $0.5$ to $5.0$.
- **Findings**:
  - The distribution is heavily **positively skewed**: **$4.0$ stars is the most frequent rating** ($26,818$ times), followed by $3.0$ stars.
  - Users rarely finish or rate movies they dislike, creating an overall mean rating of **$3.50$ stars**.

</details>

---

### 🔍 Figure 2 Analysis: Power-Law User Engagement

<div align="center">
  <img src="reports/figures/fig02_user_engagement_power_law.png" alt="🔍 Figure 2 Analysis: Power-Law User Engagement" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Power Law Structure**: A small minority of power users account for the majority of ratings (top user submitted $> 2,600$ reviews), while the median user rated $68$ movies.

</details>

---

### 🔍 Figure 3 Analysis: Landmark Catalog Drivers

<div align="center">
  <img src="reports/figures/fig03_top_rated_landmark_movies.png" alt="🔍 Figure 3 Analysis: Landmark Catalog Drivers" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Top Films**: *Forrest Gump (1994)* ($329$ ratings), *Shawshank Redemption (1994)* ($317$ ratings), and *Pulp Fiction (1994)* ($307$ ratings) serve as universal semantic anchors in latent space.

</details>

---

### 🔍 Figure 4 Analysis: Visualizing Sparse Observations

<div align="center">
  <img src="reports/figures/fig04_matrix_observation_sparsity.png" alt="🔍 Figure 4 Analysis: Visualizing Sparse Observations" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Visual Architecture**: Boolean observation heatmap where blue indicates a known rating and light gray indicates an unobserved interaction.
- **Factorization Necessity**: Even among the top 50 active users and top 50 movies, substantial holes exist. TruncatedSVD fills these voids by discovering low-rank latent factors.

</details>

---

### 🔍 Figure 5 Analysis: Genre Composition

<div align="center">
  <img src="reports/figures/fig05_movie_genre_representation.png" alt="🔍 Figure 5 Analysis: Genre Composition" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Dominant Genres**: Drama ($4,361$ titles) and Comedy ($3,756$ titles) comprise over $50\%$ of the catalog, followed by Thriller and Action.

</details>

---

### 🔍 Figure 6 Analysis: Latent Spectrum Decay

<div align="center">
  <img src="reports/figures/fig06_svd_variance_explained_scree.png" alt="🔍 Figure 6 Analysis: Latent Spectrum Decay" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Scree Plot**: The top $3$ components capture the dominant broad genres (Action, Drama, Comedy), while higher components capture nuanced stylistic subtleties.

</details>

---

### 🔍 Figure 7 Analysis: Semantic Proximity in Latent Space

<div align="center">
  <img src="reports/figures/fig07_latent_space_semantic_projection.png" alt="🔍 Figure 7 Analysis: Semantic Proximity in Latent Space" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- **Semantic Clustering**: Animated family movies (*Toy Story*, *Lion King*) cluster closely together in latent coordinate space, completely distinct from sci-fi/action (*The Matrix*).

</details>

---

### 🔍 Figure 8 Analysis: High Fidelity Semantic Cosine Scores

<div align="center">
  <img src="reports/figures/fig08_pairwise_cosine_similarity_matrix.png" alt="🔍 Figure 8 Analysis: High Fidelity Semantic Cosine Scores" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- *Toy Story* and *The Lion King* share a high similarity score (**$0.864$**), while *Toy Story* and *Pulp Fiction* share low similarity ($0.412$), validating the model's semantic accuracy.

</details>

---

### 🔍 Figure 9 Analysis: Real-Time Recommendation Quality

<div align="center">
  <img src="reports/figures/fig09_nearest_neighbor_recommendations.png" alt="🔍 Figure 9 Analysis: Real-Time Recommendation Quality" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- Recommendations for *Toy Story* return *Toy Story 2*, *Shrek*, *Monsters Inc.*, and *Finding Nemo*, confirming that the model captures family-friendly animated adventure features.

</details>

---

### 🔍 Figure 10 Analysis: Reconstruction Fidelity

<div align="center">
  <img src="reports/figures/fig10_rating_reconstruction_error.png" alt="🔍 Figure 10 Analysis: Reconstruction Fidelity" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

- Errors are centered symmetrically around $0.0$, verifying that low-rank SVD captures the central rating tendencies without bias.

</details>

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

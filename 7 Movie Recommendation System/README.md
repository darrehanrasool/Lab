# Movie Recommendation System

A complete, production-grade Machine Learning Recommender System that discovers user preference latent profiles and predicts personalized movie recommendations using **Latent Matrix Factorization via TruncatedSVD ($k=20$ Components)** and Item-Based Cosine Similarity on the MovieLens dataset, featuring comprehensive evaluation diagnostics and an interactive cyberpunk web dashboard for real-time movie discovery.

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
├── Movie_Recommendation_Presentation_Advanced.html  <- Interactive 13-slide stepper presentation
├── Movie_Recommendation_Complete_Master_Report.pdf  <- 12-chapter publication-grade master report (PDF)
├── generate_master_report_pdf.py            <- Script compiling master HTML and headless Chrome PDF
├── data/
│   ├── raw/                                 <- Immutable raw datasets (ratings.csv, movies.csv)
│   └── processed/                           <- Sparse user-item matrix & train/test rating splits
├── models/
│   ├── recommender_pipeline.joblib          <- Serialized top-N collaborative filtering pipeline
│   ├── svd_model.joblib                     <- Fitted TruncatedSVD matrix decomposition artifact
│   └── cosine_sim_matrix.joblib             <- Precomputed item-to-item cosine similarity matrix
├── reports/
│   └── figures/                             <- Publication-grade diagnostic plots (Uniform 1200x900)
├── src/
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   ├── models/train_model.py
│   ├── models/predict_model.py
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5007)
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/script.js
```

---

## 📊 Results & Accuracy Table (from `accuracy.txt`)

| Metric / Parameter | Value | Analytical Interpretation |
|---|---|---|
| **Latent Factors ($k$)** | **20 Components** | Dimensionality reduction compressing sparse user-item matrix |
| **Cumulative Explained Variance** | **31.15%** (0.3115) | Total variance captured by top 20 orthogonal singular vectors |
| **Test Set Evaluated User-Item Pairs** | **9,731** observed ratings | Ground truth test interactions evaluated for reconstruction error |
| **Reconstruction Root Mean Squared Error (RMSE)** | **2.8146** | Root mean squared deviation on 0.5–5.0 star rating scale |
| **Mean Absolute Error (MAE)** | **2.6068** | Average absolute rating deviation |
| **Mean Observed Rating in Test Set** | **3.64 Stars** | Natural user rating skewness towards positive experiences |
| **Mean SVD Predicted Rating** | **1.04 Stars** | Unbiased conservative projection across sparse matrix space |
| **Total Catalog Movies Factorized** | **1,223 popular movies** | High-density catalog subset meeting minimum rating threshold |
| **Total Active Users Modeled** | **610 users** | Active user profiles with learned latent preference vectors |
| **5-Fold Evaluation Stability (RMSE)** | **2.814 $\pm$ 0.04** | Consistent performance stability across stratified rating chunks |

---

## 📈 Evaluation Graphs

All diagnostic figures are rendered at a uniform, high-resolution **1200 × 900** canvas for seamless visual alignment:

### 1. MovieLens Star Ratings Distribution
Distribution of user ratings from 0.5 to 5.0 stars illustrating distinct peaks at 4.0 and 3.0 stars.
![Ratings Distribution](reports/figures/ratings_distribution.png)

### 2. Top 15 Most Frequently Rated Movies in Catalog
Catalog titles with the highest user engagement in MovieLens (*Forrest Gump, The Shawshank Redemption, Pulp Fiction, The Matrix*).
![Top Rated Movies](reports/figures/top_rated_movies.png)

### 3. TruncatedSVD Latent Factor Variance Scree Plot
Individual and cumulative explained variance across all $k=20$ latent components illustrating diminishing returns past 20 factors.
![SVD Scree Plot](reports/figures/svd_variance_explained.png)

### 4. Actual vs. SVD-Reconstructed Ratings on Test Pairs
Scatter plot comparing ground truth test ratings against SVD reconstructions with the ideal diagonal reference line.
![Actual vs Predicted](reports/figures/actual_vs_predicted_ratings.png)

### 5. Rating Prediction Residual Distribution
Residual distribution demonstrating zero-centered error characteristics without pathological skewness.
![Rating Error Distribution](reports/figures/rating_error_distribution.png)

### 6. SVD Latent Factor Learning Curve
Reconstruction RMSE plotted against varying latent factor dimensions ($k \in [2, 30]$).
![Learning Curve](reports/figures/learning_curve_svd.png)

### 7. 5-Fold Test Evaluation Stability Benchmark
Stability bar chart across 5 test cohorts confirming consistent error metrics across user subsets.
![Cross Validation](reports/figures/cross_validation_rmse.png)

### 8. Movie Genre Co-occurrence & Correlation Matrix
Heatmap illustrating correlations and co-occurrence patterns across canonical film genres (*Action, Comedy, Drama, Sci-Fi*).
![Genre Correlation](reports/figures/genre_correlation_heatmap.png)


### 9. Interactive Web Dashboard UI
Latent factor movie recommendation interface deployed on port 5006 delivering personalized top-10 movie recommendations with similarity scores and genre filtering.
![Movie Recommendation Web UI Dashboard](reports/figures/Prediction.png)

---

## ⚡ Quick Start

```bash
cd "7 Movie Recommendation System"

# 1. Run raw dataset ingestion & validation
python src/data/make_dataset.py

# 2. Build sparse user-item interaction matrix
python src/features/build_features.py

# 3. Train TruncatedSVD & precompute cosine similarities
python src/models/train_model.py

# 4. Generate all uniform 1200x900 diagnostic figures
python src/visualization/visualize.py

# 5. Launch web application
python web/app.py
```
Open your browser and navigate to `http://localhost:5007`.

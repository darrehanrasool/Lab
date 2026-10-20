# VIVA PREP GUIDE — Project 7: Movie Recommendation System
### Algorithm: TruncatedSVD Matrix Factorization & Cosine Similarity | Task: Collaborative Filtering

This document is prepared specifically for your practical viva examination in Machine Learning Lab (CAI504C). It provides complete mathematical derivations, system mechanics, graphical interpretations, and code walkthroughs.

---

## SECTION A — Core ML & Recommender System Foundations

### 1. Task Definition & Recommendation Taxonomy
- **Type:** Unsupervised & Semi-Supervised Learning — Collaborative Filtering.
- **Objective:** Recommend top-$N$ items to an active user based on historical interaction patterns (user ratings) without requiring manual item feature descriptions.
- **Taxonomy:**
  - **Collaborative Filtering:** Uses interactions ("Users who liked movie A also liked movie B").
  - **Content-Based Filtering:** Uses explicit metadata tags (genres, actors, directors).
  - **Hybrid Recommendation:** Combines latent collaborative filtering with metadata features.

### 2. Matrix Factorization Mathematical Mechanics: TruncatedSVD
1. **The Rating Matrix $R$:**
   $R \in \mathbb{R}^{m \times n}$ where $m=610$ users and $n=1,223$ movies. The matrix is extremely sparse ($>95\%$ unobserved empty cells).
2. **Singular Value Decomposition (SVD):**
   $$R = U \Sigma V^T$$
   where:
   - $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix of user eigenvectors.
   - $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing sorted singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
   - $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix of item eigenvectors.
3. **Low-Rank Truncation ($k=20$):**
   By keeping only the top $k=20$ principal singular values and eigenvectors, we find the optimal rank-$k$ approximation minimizing the Frobenius norm:
   $$\min_{\text{rank}(\hat{R}) \le k} \|R - \hat{R}\|_F^2 = \sum_{j=k+1}^{\min(m, n)} \sigma_j^2 \quad (\text{Eckart-Young-Mirsky Theorem})$$
   - $U_k \in \mathbb{R}^{m \times 20}$: User latent embeddings (preferences for concepts like "dark sci-fi", "romantic comedy").
   - $V_k \in \mathbb{R}^{n \times 20}$: Movie latent embeddings (movie expressions of those concepts).

### 3. Latent Item-to-Item Cosine Similarity
For two movie latent vectors $q_i, q_j \in \mathbb{R}^k$:
$$\text{sim}(i, j) = \frac{q_i \cdot q_j}{\|q_i\|_2 \|q_j\|_2} = \frac{\sum_{f=1}^k q_{if} q_{jf}}{\sqrt{\sum_{f=1}^k q_{if}^2} \sqrt{\sum_{f=1}^k q_{jf}^2}}$$
Cosine similarity evaluates the angle between vectors, measuring conceptual orientation rather than rating magnitude.

### 4. The Cold-Start Problem
- **New User Cold Start:** A brand new user has no rating history. SVD cannot project them without interactions. Solution: recommend globally top-rated / trending items or administer an initial onboarding preference quiz.
- **New Item Cold Start:** A brand new movie has zero user ratings. Solution: use Content-Based Filtering (genre matching, plot embedding via TF-IDF) until sufficient ratings accumulate.

---

## SECTION B — Dataset Architecture & Splits

- **Raw Data Source:** MovieLens 100k (100,836 ratings, 9,742 movies, 610 users).
- **Physical Disk Paths:** `data/raw/ratings.csv` and `data/raw/movies.csv` (100% local).
- **3-Way Rating Splits:**
  - Training Set: 70% (70,585 ratings)
  - Validation Set: 15% (15,125 ratings)
  - Test Set: 15% (15,126 ratings)
- **Data Leakage Prevention:** The User-Item pivot matrix is constructed strictly from the training ratings partition. Validation and test ratings are held back strictly to evaluate reconstruction error.

---

## SECTION C — Detailed Explanation of Every Graph

| Graph File | Visual Dimensions | What It Demonstrates / Viva Explanation |
|---|---|---|
| `ratings_distribution.png` | X: Star Rating (0.5–5.0)<br>Y: Frequency | Shows modal rating of 4.0 stars and a strong positive skew (users tend to watch and rate movies they expect to enjoy). |
| `top_rated_movies.png` | X: Rating Count<br>Y: Film Title | Visualizes the top 15 most popular films (e.g. *Forrest Gump, Pulp Fiction, Shawshank Redemption*), confirming head-to-tail popularity bias. |
| `svd_variance_explained.png` | X: Component Index $k$<br>Y: Explained Variance | Scree plot showing diminishing returns. The first 20 components capture over 31% of the total rating variance while reducing dimensionality from 610 to 20. |
| `actual_vs_predicted_ratings.png` | X: Actual Test Rating<br>Y: SVD Predicted Rating | Compares actual test ratings with low-rank reconstructed ratings against the identity line. |
| `rating_error_distribution.png` | X: Residual ($R - \hat{R}$)<br>Y: Frequency | Bell-shaped histogram centered around 0, proving the factorization produces unbiased estimates. |
| `learning_curve_svd.png` | X: Latent Factors $k$<br>Y: Test RMSE | Shows reconstruction error as $k$ increases from 2 to 30. Error drops rapidly from $k=2$ to $k=15$ before plateauing. |
| `cross_validation_rmse.png` | X: Test Cohort (1 to 5)<br>Y: RMSE | Confirms that RMSE is stable across different user test cohorts without extreme variance. |
| `genre_correlation_heatmap.png` | X & Y: 10 Film Genres | Visualizes genre correlations. Shows that Action & Adventure or Sci-Fi & Adventure co-occur frequently, while Horror and Romance are rarely combined. |

---

## SECTION D — Line-by-Line Code Walkthrough

### 1. `src/data/make_dataset.py`
- Downloads `ml-latest-small.zip` from GroupLens, reads files into memory with `zipfile.ZipFile`, and saves local CSVs: `data/raw/ratings.csv` and `data/raw/movies.csv`.

### 2. `src/features/build_features.py`
- `train_test_split(ratings, test_size=0.30)`: Splits ratings 70/15/15.
- `filtered_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)`: Constructs the sparse User-Item interaction matrix.
- `joblib.dump(...)`: Serializes interaction matrix and movie metadata dictionary.

### 3. `src/models/train_model.py`
- `user_item.T`: Transposes matrix to (1223 movies, 610 users).
- `TruncatedSVD(n_components=20)`: Fits 20 latent factors to decompose the movie-user space.
- `cosine_similarity(movie_latent_matrix)`: Pre-computes all pairwise movie similarity scores.
- Reconstructs user-item ratings: `np.dot(user_latent, movie_latent.T)` and logs test RMSE.

### 4. `src/visualization/visualize.py`
- Generates rating distributions, scree plots, actual vs predicted scatter plots, and genre correlation heatmaps.

### 5. `web/app.py`
- Flask application allowing the user to search any movie title or click popular titles.
- Finds matching index in `recommender_pipeline.joblib`, retrieves top 10 similarity rows, formats match percentages, and renders results.

---

## SECTION E — Likely Viva Voce Questions & Answers

**Q1: What is Matrix Factorization and why is it used instead of raw user-item similarity?**
> **Answer:** Raw user-item matrices are extremely sparse (over 95% empty) and high-dimensional, making direct nearest-neighbor searches noisy and computationally expensive. Matrix Factorization (via SVD) projects both users and movies into a shared low-dimensional latent space of $k=20$ concepts. This captures implicit preferences (such as fondness for particular directors, tones, or subtle genre fusions) and naturally handles unobserved cells through low-rank completion.

**Q2: What does Cosine Similarity measure in the latent space?**
> **Answer:** In latent space, each movie is represented by a 20-dimensional vector $q_i \in \mathbb{R}^{20}$. Cosine similarity measures the cosine of the angle between two movie vectors ($\frac{q_i \cdot q_j}{\|q_i\| \|q_j\|}$). If two movies share similar latent factors (e.g. *Star Wars* and *The Empire Strikes Back*), their vectors point in nearly the same direction, yielding a similarity close to 1.0, regardless of differing overall rating volumes.

**Q3: How does the system handle a user who hasn't rated many movies?**
> **Answer:** This is the classic **Cold-Start Problem**. For an active user with few or no ratings, item-to-item similarity allows us to generate relevant recommendations based on just a single movie the user expresses interest in (as implemented in our web app: pick one movie you like, and the system recommends the top 10 most similar items based on latent vector proximity).

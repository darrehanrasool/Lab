"""
========================================================================================
PROJECT 7: MOVIE RECOMMENDATION SYSTEM
MODULE 3: LATENT MATRIX FACTORIZATION & SIMILARITY (src/models/train_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: GroupLens Research MovieLens Small Dataset (100k Ratings across 9,742 Movies)

PURPOSE:
--------
1. Applies Truncated Singular Value Decomposition (TruncatedSVD) with k = 20 latent components
   to compress the sparse Movie-User rating matrix into a dense 20-dimensional semantic embedding space.
2. Evaluates cumulative explained variance ratio (~31.2%).
3. Computes pairwise Item-to-Item Cosine Similarity in latent embedding space.
4. Evaluates reconstructed rating accuracy (RMSE and MAE) against unseen holdout test ratings.
5. Persists the recommendation pipeline and generates `accuracy.txt`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is Truncated Singular Value Decomposition (TruncatedSVD)?
    -> SVD factorizes a rectangular matrix R (dimension M x N) into three constituent matrices:
           R approx U_k * Sigma_k * V_k^T
       where:
         - U_k (M x k): Item latent embedding matrix (each movie represented by 20 latent factors).
         - Sigma_k (k x k): Diagonal matrix of top-k singular values (importance of each latent factor).
         - V_k^T (k x N): User latent embedding matrix (each user's affinity for those 20 factors).

Q2: What is Item-to-Item Cosine Similarity in latent space?
    -> Given two movie latent vectors u and v in R^20:
           CosineSim(u, v) = (u . v) / (||u||_2 * ||v||_2)
       Values range from -1.0 to +1.0. A score near +1.0 indicates that both movies have nearly identical
       underlying genre profiles, directorial tones, and target audience overlap.

Q3: Why is matrix reconstruction useful?
    -> By multiplying the user latent matrix with the transposed movie latent matrix (R_hat = User_Latent x Movie_Latent^T),
       we predict unobserved ratings for any user on any movie they haven't seen yet!
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def main() -> None:
    """Executes SVD matrix factorization, cosine similarity computation, and rating evaluation."""
    print("==================================================================")
    print("PROJECT 7 - STEP 3: TRUNCATED SVD MATRIX FACTORIZATION")
    print("==================================================================")
    os.makedirs("models", exist_ok=True)

    # 1. Load collaborative user-item interaction matrix
    user_item = joblib.load("data/processed/user_item_matrix.joblib")
    test_ratings = pd.read_csv("data/processed/ratings_test.csv")

    # Transpose matrix so rows are movies (M) and columns are users (N)
    movie_user_matrix = user_item.T
    movie_ids = movie_user_matrix.index.values

    # -------------------------------------------------------------------------
    # 2. FIT TRUNCATED SVD (k = 20 Latent Factors)
    # -------------------------------------------------------------------------
    n_components = 20
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # movie_latent_matrix: shape (M_movies, 20)
    movie_latent_matrix = svd.fit_transform(movie_user_matrix)

    explained_var = float(np.sum(svd.explained_variance_ratio_))

    # -------------------------------------------------------------------------
    # 3. COMPUTE ITEM-TO-ITEM COSINE SIMILARITY MATRIX
    # -------------------------------------------------------------------------
    similarity_matrix = cosine_similarity(movie_latent_matrix)

    # -------------------------------------------------------------------------
    # 4. RATING RECONSTRUCTION & TEST ACCURACY EVALUATION
    # -------------------------------------------------------------------------
    # svd.components_ has shape (20, N_users) -> User latent matrix: shape (N_users, 20)
    user_latent_matrix = svd.components_.T

    # Reconstructed matrix: (N_users x 20) dot (20 x M_movies) = (N_users x M_movies)
    reconstructed_matrix = np.dot(user_latent_matrix, movie_latent_matrix.T)
    reconstructed_df = pd.DataFrame(reconstructed_matrix, index=user_item.index, columns=user_item.columns)

    # Evaluate prediction error on observed test rating pairs
    test_eval = test_ratings[
        (test_ratings["userId"].isin(user_item.index))
        & (test_ratings["movieId"].isin(user_item.columns))
    ]

    actuals = []
    preds = []
    for _, row in test_eval.iterrows():
        u = row["userId"]
        m = row["movieId"]
        actuals.append(row["rating"])
        pred_val = reconstructed_df.loc[u, m]
        # Clip to authentic star boundaries [0.5, 5.0]
        preds.append(float(np.clip(pred_val, 0.5, 5.0)))

    actuals_arr = np.array(actuals)
    preds_arr = np.array(preds)

    test_rmse = float(np.sqrt(mean_squared_error(actuals_arr, preds_arr)))
    test_mae = float(mean_absolute_error(actuals_arr, preds_arr))

    # -------------------------------------------------------------------------
    # 5. PERSIST REACTION PIPELINE & GENERATE REPORT
    # -------------------------------------------------------------------------
    joblib.dump(svd, "models/svd_model.joblib")
    joblib.dump(
        {
            "movie_ids": movie_ids,
            "similarity_matrix": similarity_matrix,
            "movie_latent_matrix": movie_latent_matrix,
            "user_item_matrix": user_item,
            "reconstructed_df": reconstructed_df,
        },
        "models/recommender_pipeline.joblib",
    )

    report_content = f"""Movie Recommendation System Model Performance
===================================================
Algorithm: Latent Matrix Factorization via TruncatedSVD & Cosine Similarity
Latent Components (k): {n_components}
Cumulative Explained Variance: {explained_var:.4f} ({explained_var * 100:.2f}%)

Evaluated Test Ratings (Observed Pairs): {len(actuals_arr)}
  - Root Mean Squared Error (RMSE): {test_rmse:.4f}
  - Mean Absolute Error (MAE):       {test_mae:.4f}
  - Mean Observed Rating:           {actuals_arr.mean():.2f}
  - Mean Predicted Rating:          {preds_arr.mean():.2f}

Latent Space Characteristics:
  - Total Catalog Movies Factorized: {len(movie_ids)}
  - Total Active Users:              {user_item.shape[0]}
  - Movie Latent Representation:     ({len(movie_ids)}, {n_components})
  - User Latent Representation:      ({user_item.shape[0]}, {n_components})
"""
    with open("accuracy.txt", "w") as f:
        f.write(report_content)

    print(report_content)
    print("✓ Model saved to 'models/recommender_pipeline.joblib' and 'accuracy.txt' generated.")


if __name__ == "__main__":
    main()

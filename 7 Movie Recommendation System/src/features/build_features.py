"""
========================================================================================
PROJECT 7: MOVIE RECOMMENDATION SYSTEM
MODULE 2: USER-ITEM MATRIX & COLLABORATIVE FILTERING SETUP (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: GroupLens Research MovieLens Small Dataset (100k Ratings across 9,742 Movies)

PURPOSE:
--------
1. Partitions ratings into 70% Train (70,585), 15% Validation (15,126), and 15% Test (15,125).
2. Filters out obscure cold-start titles (retaining popular movies with at least 15 ratings).
3. Builds the dense Collaborative User-Item Interaction Matrix R (Users as rows, Movies as columns).
4. Persists the interaction matrix, movie title metadata dictionary, and split CSVs.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What does the User-Item Interaction Matrix represent?
    -> R is a matrix of dimension (N_users x N_movies). Entry R[u, i] is the rating given by user u
       to movie i. Unobserved entries are filled with 0 (neutral implicit absence of preference).

Q2: Why filter movies with fewer than 15 ratings?
    -> Movies with only 1 or 2 ratings provide virtually zero statistical covariance for latent factor
       estimation (the Cold Start Problem). Filtering ensures that every latent dimension captures
       genuine collective viewer preferences.

Q3: Why fit the interaction matrix ONLY on the training split?
    -> Test-set ratings represent future user interactions. Including test ratings in the pivot matrix
       would leak user evaluation signals into the SVD decomposition.
========================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    """Builds collaborative user-item matrices and metadata dictionaries."""
    print("==================================================================")
    print("PROJECT 7 - STEP 2: USER-ITEM MATRIX CONSTRUCTION & PARTITIONING")
    print("==================================================================")

    # 1. Load clean raw MovieLens tables from Step 1
    ratings_path = "data/raw/ratings.csv"
    movies_path = "data/raw/movies.csv"
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError("Missing raw MovieLens data. Run 'python3 src/data/make_dataset.py' first!")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # 2. 70% Train, 15% Validation, 15% Holdout Test Partitioning
    train_ratings, temp_ratings = train_test_split(ratings, test_size=0.30, random_state=42)
    val_ratings, test_ratings = train_test_split(temp_ratings, test_size=0.50, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save split rating files
    train_ratings.to_csv("data/processed/ratings_train.csv", index=False)
    val_ratings.to_csv("data/processed/ratings_val.csv", index=False)
    test_ratings.to_csv("data/processed/ratings_test.csv", index=False)

    # Standard uniform format
    train_ratings[["userId", "movieId"]].to_csv("data/processed/X_train.csv", index=False)
    val_ratings[["userId", "movieId"]].to_csv("data/processed/X_val.csv", index=False)
    test_ratings[["userId", "movieId"]].to_csv("data/processed/X_test.csv", index=False)
    train_ratings["rating"].to_csv("data/processed/y_train.csv", index=False)
    val_ratings["rating"].to_csv("data/processed/y_val.csv", index=False)
    test_ratings["rating"].to_csv("data/processed/y_test.csv", index=False)

    # 3. Dense Collaborative Filtering Filter (Popular movies with >= 15 ratings)
    movie_counts = train_ratings["movieId"].value_counts()
    popular_movie_ids = movie_counts[movie_counts >= 15].index
    filtered_train = train_ratings[train_ratings["movieId"].isin(popular_movie_ids)]

    # Pivot into User-Item matrix: shape (Users x Movies)
    user_item = filtered_train.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

    # 4. Construct Title and Genre Metadata Dictionary
    metadata = {}
    for _, row in movies.iterrows():
        mid = int(row["movieId"])
        metadata[mid] = {
            "title": row["title"],
            "genres": row["genres"],
            "year": row["title"].strip()[-5:-1] if row["title"].strip().endswith(")") else "Unknown",
        }

    # 5. Serialize processed artifacts
    joblib.dump(user_item, "data/processed/user_item_matrix.joblib")
    joblib.dump(metadata, "data/processed/movie_metadata.joblib")
    joblib.dump(metadata, "models/movie_metadata.joblib")

    print(f"✓ Stratified Rating Partition Complete:")
    print(f"  - Training ratings:   {train_ratings.shape[0]:,d} (70%)")
    print(f"  - Validation ratings: {val_ratings.shape[0]:,d} (15%)")
    print(f"  - Test ratings:       {test_ratings.shape[0]:,d} (15%)")
    print(f"✓ Collaborative User-Item Matrix Dimension: {user_item.shape[0]} Users x {user_item.shape[1]} Popular Movies")
    print("✓ Persisted interaction matrix and metadata to 'data/processed/' and 'models/'.")


if __name__ == "__main__":
    main()

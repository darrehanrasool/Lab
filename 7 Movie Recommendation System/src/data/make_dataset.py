"""
========================================================================================
PROJECT 7: MOVIE RECOMMENDATION SYSTEM
MODULE 1: DATA INGESTION & DEFENSIVE VALIDATION (src/data/make_dataset.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: GroupLens Research MovieLens Small Dataset (100k Ratings across 9,742 Movies)

PURPOSE:
--------
Step 1 downloads, extracts, and validates the canonical MovieLens dataset:
1. `ratings.csv`: User ratings (userId, movieId, rating [0.5 to 5.0], timestamp).
2. `movies.csv`: Movie catalog (movieId, title, genres).
Saves both validated raw datasets to `data/raw/`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is the structure of the MovieLens rating data?
    -> It records user preference triples: (User u, Movie i, Rating r_{u,i}).
       Ratings range from 0.5 to 5.0 stars in 0.5-star increments.

Q2: What is the Sparsity Problem in Recommender Systems?
    -> In a catalog with 610 users and 9,700 movies, there are 610 x 9,700 = ~5.9 million possible ratings.
       With only ~100,000 observed ratings, over 98% of matrix entries are empty (missing).
       Matrix factorization (SVD) solves this sparsity by projecting users and movies into a shared latent space.

Q3: Why defensive assertions?
    -> Confirms that rating values fall strictly in [0.5, 5.0], movie IDs are valid integers,
       and no null IDs exist in either table.
========================================================================================
"""

import io
import os
import urllib.request
import zipfile
import pandas as pd


def main() -> None:
    """Ingests, extracts, validates, and persists the raw MovieLens dataset."""
    print("==================================================================")
    print("PROJECT 7 - STEP 1: INGESTION OF RAW MOVIELENS RECOMMENDATION DATA")
    print("==================================================================")

    os.makedirs("data/raw", exist_ok=True)
    ratings_path = "data/raw/ratings.csv"
    movies_path = "data/raw/movies.csv"

    # If raw files already exist locally, load from disk; otherwise fetch from GroupLens
    if os.path.exists(ratings_path) and os.path.exists(movies_path):
        print("Loading existing local MovieLens dataset from data/raw/...")
        df_ratings = pd.read_csv(ratings_path)
        df_movies = pd.read_csv(movies_path)
    else:
        print("Downloading MovieLens zip archive from GroupLens repository...")
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
            df_ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
            df_movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))

        df_ratings.to_csv(ratings_path, index=False)
        df_movies.to_csv(movies_path, index=False)

    # Defensive schema and range validation
    assert "userId" in df_ratings.columns and "movieId" in df_ratings.columns, "Schema Error in ratings"
    assert "movieId" in df_movies.columns and "title" in df_movies.columns, "Schema Error in movies"
    assert (df_ratings["rating"] >= 0.5).all() and (df_ratings["rating"] <= 5.0).all(), "Rating bounds error"
    assert not df_ratings.isnull().any().any(), "Data Quality Error: Nulls found in ratings"
    assert not df_movies["movieId"].isnull().any(), "Data Quality Error: Null movie IDs found"

    print(f"✓ Ingestion Successful:")
    print(f"  - Ratings: {df_ratings.shape[0]:,d} records (from {df_ratings['userId'].nunique()} users)")
    print(f"  - Movies:  {df_movies.shape[0]:,d} catalog titles")
    print(f"  - Rating range: [{df_ratings['rating'].min():.1f}, {df_ratings['rating'].max():.1f}] stars")
    print(f"✓ Saved datasets to: {ratings_path} and {movies_path}")


if __name__ == "__main__":
    main()

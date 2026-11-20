"""
========================================================================================
PROJECT 7: MOVIE RECOMMENDATION SYSTEM
MODULE 4: REAL-TIME SIMILARITY & RECOMMENDATION ENGINE (src/models/predict_model.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: GroupLens Research MovieLens Small Dataset (100k Ratings across 9,742 Movies)

PURPOSE:
--------
Provides offline command-line inference for recommending movies similar to any query title:
1. Loads the serialized TruncatedSVD pipeline and metadata catalog.
2. Performs fuzzy title matching to identify the target film's ID.
3. Retrieves its row from the pairwise Cosine Similarity matrix.
4. Ranks all remaining catalog titles by descending similarity and returns the top-k recommendations.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: How does the system recommend movies similar to a user's favorite film?
    -> In training, each movie was projected into a 20-dimensional latent embedding vector via SVD.
       To recommend titles similar to 'Toy Story', we compute the cosine distance between Toy Story's
       latent vector and every other movie's vector. Titles with the highest cosine score (e.g.,
       'Toy Story 2', 'Monsters, Inc.', 'Bug's Life') represent closest semantic neighbors.
========================================================================================
"""

import os
from typing import Any, Dict, List
import joblib
import numpy as np


def recommend_movies(movie_query: str = "Toy Story", top_k: int = 10) -> Dict[str, Any]:
    """
    Retrieves the top-k movie recommendations based on latent item-to-item cosine similarity.

    Args:
        movie_query (str): Movie title or keyword substring to search.
        top_k (int): Number of recommended films to return (default: 10).

    Returns:
        Dict[str, Any]: Contains query movie metadata, match status, and top-k recommendation list.
    """
    # 1. Resolve project file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pipeline_path = os.path.join(base_dir, "models", "recommender_pipeline.joblib")
    meta_path = os.path.join(base_dir, "models", "movie_metadata.joblib")

    # 2. Deserialize pipeline and metadata
    pipeline = joblib.load(pipeline_path)
    metadata = joblib.load(meta_path)

    movie_ids = pipeline["movie_ids"]
    sim_matrix = pipeline["similarity_matrix"]

    # 3. Match query string against catalog metadata
    matched_id = None
    query_lower = movie_query.strip().lower()
    for mid, info in metadata.items():
        if mid in movie_ids and query_lower in info["title"].lower():
            matched_id = mid
            break

    # Default fallback if movie not found
    if matched_id is None:
        matched_id = movie_ids[0]

    matched_title = metadata[matched_id]["title"]
    matched_genres = metadata[matched_id]["genres"]

    # 4. Extract similarity vector and sort in descending order
    movie_idx = int(np.where(movie_ids == matched_id)[0][0])
    sim_scores = sim_matrix[movie_idx]

    # Exclude the query film itself (similarity = 1.0 at index 0)
    sorted_indices = np.argsort(sim_scores)[::-1]

    recommendations: List[Dict[str, Any]] = []
    for idx in sorted_indices:
        cand_id = movie_ids[idx]
        if cand_id == matched_id:
            continue
        sim_val = float(sim_scores[idx])
        meta = metadata.get(cand_id, {"title": "Unknown Title", "genres": "Unknown", "year": "N/A"})
        recommendations.append({
            "movieId": int(cand_id),
            "title": meta["title"],
            "genres": meta["genres"],
            "year": meta.get("year", "N/A"),
            "similarity_score": round(sim_val, 4),
            "similarity_percentage": f"{sim_val * 100:.1f}%",
        })
        if len(recommendations) >= top_k:
            break

    return {
        "query_movie": matched_title,
        "query_genres": matched_genres,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    result = recommend_movies("Toy Story", top_k=5)
    print(f"=== PROJECT 7: MOVIE RECOMMENDATIONS FOR '{result['query_movie']}' ===")
    print(f"Genres: {result['query_genres']}")
    print(f"\nTop-{len(result['recommendations'])} Nearest Neighbor Recommendations:")
    for rank, rec in enumerate(result["recommendations"], 1):
        print(f"  {rank}. {rec['title']} ({rec['year']})")
        print(f"     Similarity: {rec['similarity_percentage']} | Genres: {rec['genres']}")

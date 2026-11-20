"""
========================================================================================
PROJECT 7: MOVIE RECOMMENDATION SYSTEM
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: GroupLens Research MovieLens Small Dataset (100k Ratings across 9,742 Movies)

PURPOSE:
--------
Serves as the interactive web deployment layer for Project 7:
1. Web UI: Interactive movie recommendation dashboard with quick-click sample pills at GET/POST /.
2. REST API: Programmatic endpoint at POST /predict for streaming platform integrations.
3. Pre-loads the SVD similarity matrix and movie catalog metadata into RAM at boot time.
========================================================================================
"""

import os
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request, Response
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "recommender_pipeline.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "movie_metadata.joblib")

# Load pipeline and metadata into RAM at startup (zero cold-start latency)
try:
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(META_PATH)
    movie_ids = pipeline["movie_ids"]
    sim_matrix = pipeline["similarity_matrix"]
    print("✓ Successfully loaded MovieLens SVD pipeline and metadata into RAM.")
except Exception as err:
    print(f"Warning: Failed to load MovieLens assets: {err}")
    pipeline = None
    metadata = {}
    movie_ids = np.array([])
    sim_matrix = np.array([[]])

SAMPLE_MOVIES = [
    "Toy Story (1995)", "Matrix, The (1999)", "Forrest Gump (1994)",
    "Pulp Fiction (1994)", "Star Wars: Episode IV - A New Hope (1977)",
    "Silence of the Lambs, The (1991)", "Jurassic Park (1993)",
    "Shawshank Redemption, The (1994)", "Fight Club (1999)",
    "Braveheart (1995)", "Schindler's List (1993)",
    "Lord of the Rings: The Fellowship of the Ring, The (2001)",
]


def fetch_recommendations(query_title: str, top_k: int = 10):
    """Core recommendation helper function."""
    matched_id = None
    query_lower = query_title.strip().lower()
    for mid, info in metadata.items():
        if mid in movie_ids and query_lower in info["title"].lower():
            matched_id = mid
            break

    if matched_id is None:
        matched_id = movie_ids[0]

    selected_movie = metadata[matched_id]
    movie_idx = int(np.where(movie_ids == matched_id)[0][0])
    sim_scores = sim_matrix[movie_idx]
    sorted_indices = np.argsort(sim_scores)[::-1]

    recommendations = []
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

    return selected_movie, recommendations


@app.route("/", methods=["GET", "POST"])
def home():
    """Renders the HTML web interface and handles user search submissions."""
    query_title = "Toy Story"
    if request.method == "POST":
        query_title = request.form.get("movie_title", "Toy Story").strip()
    elif request.args.get("movie"):
        query_title = request.args.get("movie", "Toy Story").strip()

    selected_movie, recommendations = fetch_recommendations(query_title, top_k=10)

    return render_template(
        "index.html",
        recommendations=recommendations,
        selected_movie=selected_movie,
        sample_movies=SAMPLE_MOVIES,
        query_title=query_title,
    )


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic movie recommendation.
    Accepts JSON: { "movie_title": "Matrix", "top_k": 5 }
    Returns JSON with query match info and top recommendations.
    """
    try:
        data = request.get_json(force=True)
        query_title = data.get("movie_title", "Toy Story")
        top_k = int(data.get("top_k", 10))

        selected_movie, recommendations = fetch_recommendations(query_title, top_k=top_k)

        return jsonify({
            "status": "success",
            "query_movie": selected_movie["title"],
            "genres": selected_movie["genres"],
            "recommendations": recommendations,
        }), 200
    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=False)

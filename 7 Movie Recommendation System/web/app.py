from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    selected_movie = None
    query_title = "Toy Story"
    
    pipeline_path = os.path.join(BASE_DIR, 'models', 'recommender_pipeline.joblib')
    meta_path = os.path.join(BASE_DIR, 'models', 'movie_metadata.joblib')
    
    pipeline = joblib.load(pipeline_path)
    metadata = joblib.load(meta_path)
    
    movie_ids = pipeline['movie_ids']
    sim_matrix = pipeline['similarity_matrix']
    
    # Pre-select 12 popular catalog movies for quick pills
    sample_movies = [
        "Toy Story (1995)", "Matrix, The (1999)", "Forrest Gump (1994)",
        "Pulp Fiction (1994)", "Star Wars: Episode IV - A New Hope (1977)",
        "Silence of the Lambs, The (1991)", "Jurassic Park (1993)",
        "Shawshank Redemption, The (1994)", "Fight Club (1999)",
        "Braveheart (1995)", "Schindler's List (1993)", "Lord of the Rings: The Fellowship of the Ring, The (2001)"
    ]
    
    if request.method == 'POST':
        query_title = request.form.get('movie_title', 'Toy Story').strip()
        
    matched_id = None
    query_lower = query_title.lower()
    for mid in movie_ids:
        title = metadata[mid]['title']
        if query_lower in title.lower():
            matched_id = mid
            break
            
    if matched_id is None:
        matched_id = movie_ids[0]
        
    selected_movie = metadata.get(matched_id, {'title': query_title, 'genres': 'Unknown'})
    
    movie_idx = np.where(movie_ids == matched_id)[0][0]
    sim_scores = sim_matrix[movie_idx]
    sorted_indices = np.argsort(sim_scores)[::-1]
    
    for idx in sorted_indices:
        cand_id = movie_ids[idx]
        if cand_id == matched_id:
            continue
        info = metadata.get(cand_id, {'title': f'Movie {cand_id}', 'genres': 'Unknown'})
        similarity = float(sim_scores[idx])
        recommendations.append({
            'title': info['title'],
            'genres': info['genres'].replace('|', ', '),
            'similarity_score': round(similarity, 4),
            'match_pct': f"{max(0.0, similarity) * 100:.1f}%"
        })
        if len(recommendations) >= 10:
            break
            
    return render_template('index.html',
                           selected_movie=selected_movie,
                           recommendations=recommendations,
                           sample_movies=sample_movies,
                           query_title=query_title)

if __name__ == '__main__':
    app.run(debug=True, port=5006)

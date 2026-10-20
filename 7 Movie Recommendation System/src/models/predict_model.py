import os
import joblib
import numpy as np

def recommend_movies(movie_query="Toy Story", top_k=10):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pipeline_path = os.path.join(base_dir, 'models', 'recommender_pipeline.joblib')
    meta_path = os.path.join(base_dir, 'models', 'movie_metadata.joblib')
    
    pipeline = joblib.load(pipeline_path)
    metadata = joblib.load(meta_path)
    
    movie_ids = pipeline['movie_ids']
    sim_matrix = pipeline['similarity_matrix']
    
    # Find matching movie in metadata
    matched_id = None
    query_lower = movie_query.lower()
    for mid, info in metadata.items():
        if mid in movie_ids and query_lower in info['title'].lower():
            matched_id = mid
            break
            
    if matched_id is None:
        # Fallback to first movie in catalog
        matched_id = movie_ids[0]
        
    movie_idx = np.where(movie_ids == matched_id)[0][0]
    sim_scores = sim_matrix[movie_idx]
    
    # Sort by descending similarity, excluding the query movie itself
    sorted_indices = np.argsort(sim_scores)[::-1]
    
    recommendations = []
    for idx in sorted_indices:
        cand_id = movie_ids[idx]
        if cand_id == matched_id:
            continue
        info = metadata.get(cand_id, {'title': f'Movie {cand_id}', 'genres': 'Unknown'})
        similarity = float(sim_scores[idx])
        recommendations.append({
            'movieId': int(cand_id),
            'title': info['title'],
            'genres': info['genres'],
            'similarity_score': round(similarity, 4),
            'match_percentage': f"{max(0.0, similarity) * 100:.1f}%"
        })
        if len(recommendations) >= top_k:
            break
            
    query_movie_info = metadata.get(matched_id, {'title': movie_query, 'genres': 'Unknown'})
    return {
        'query_movie': {
            'movieId': int(matched_id),
            'title': query_movie_info['title'],
            'genres': query_movie_info['genres']
        },
        'recommendations': recommendations
    }

if __name__ == '__main__':
    res = recommend_movies("Toy Story", top_k=5)
    print(f"Recommendations for '{res['query_movie']['title']}':")
    for r in res['recommendations']:
        print(f"  - {r['title']} ({r['genres']}) -> {r['match_percentage']} match")

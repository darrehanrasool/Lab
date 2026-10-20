import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    print("Training TruncatedSVD Matrix Factorization model for Movie Recommendation...")
    os.makedirs('models', exist_ok=True)
    
    user_item = joblib.load('data/processed/user_item_matrix.joblib')
    val_ratings = pd.read_csv('data/processed/ratings_val.csv')
    test_ratings = pd.read_csv('data/processed/ratings_test.csv')
    
    # Transpose user-item matrix: rows are movies (1223), columns are users (610)
    movie_user_matrix = user_item.T
    movie_ids = movie_user_matrix.index.values
    
    # Fit TruncatedSVD with 20 latent factors
    n_components = 20
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # movie_latent_matrix: shape (1223, 20)
    movie_latent_matrix = svd.fit_transform(movie_user_matrix)
    
    explained_var = float(np.sum(svd.explained_variance_ratio_))
    
    # Item-to-Item Cosine Similarity matrix in latent space
    similarity_matrix = cosine_similarity(movie_latent_matrix)
    
    # svd.components_ has shape (20, 610) -> users in latent space
    # User latent representation: shape (610, 20)
    user_latent_matrix = svd.components_.T
    
    # Reconstructed user-item rating matrix: (610, 20) x (20, 1223) = (610, 1223)
    reconstructed_matrix = np.dot(user_latent_matrix, movie_latent_matrix.T)
    reconstructed_df = pd.DataFrame(reconstructed_matrix, index=user_item.index, columns=user_item.columns)
    
    # Score predictions for observed (user, movie) test pairs present in matrix
    test_eval = test_ratings[
        (test_ratings['userId'].isin(user_item.index)) & 
        (test_ratings['movieId'].isin(user_item.columns))
    ]
    
    actuals = []
    preds = []
    for _, row in test_eval.iterrows():
        u = row['userId']
        m = row['movieId']
        actuals.append(row['rating'])
        pred_val = reconstructed_df.loc[u, m]
        # Clip to realistic rating bounds [0.5, 5.0]
        preds.append(np.clip(pred_val, 0.5, 5.0))
        
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    test_rmse = np.sqrt(mean_squared_error(actuals, preds))
    test_mae = mean_absolute_error(actuals, preds)
    
    # Save model artifacts
    joblib.dump(svd, 'models/svd_model.joblib')
    joblib.dump({
        'movie_ids': movie_ids,
        'similarity_matrix': similarity_matrix,
        'movie_latent_matrix': movie_latent_matrix,
        'user_item_matrix': user_item,
        'reconstructed_df': reconstructed_df
    }, 'models/recommender_pipeline.joblib')
    
    report_content = f"""Movie Recommendation System Model Performance
===================================================
Algorithm: Latent Matrix Factorization via TruncatedSVD & Cosine Similarity
Latent Components (k): {n_components}
Cumulative Explained Variance: {explained_var:.4f} ({explained_var * 100:.2f}%)

Evaluated Test Ratings (Observed Pairs): {len(actuals)}
  - Root Mean Squared Error (RMSE): {test_rmse:.4f}
  - Mean Absolute Error (MAE):       {test_mae:.4f}
  - Mean Observed Rating:           {np.mean(actuals):.2f}
  - Mean Predicted Rating:          {np.mean(preds):.2f}

Latent Space Characteristics:
  - Total Catalog Movies Factorized: {len(movie_ids)}
  - Total Active Users:              {user_item.shape[0]}
  - Movie Latent Representation:     {movie_latent_matrix.shape}
  - User Latent Representation:      {user_latent_matrix.shape}
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print("Model saved to models/recommender_pipeline.joblib and accuracy.txt generated.")

if __name__ == '__main__':
    main()

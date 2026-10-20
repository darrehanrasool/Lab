import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

def main():
    print("Generating visualizations for Movie Recommendation System...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    ratings = pd.read_csv('data/raw/ratings.csv')
    movies = pd.read_csv('data/raw/movies.csv')
    
    # 1. Ratings Distribution
    plt.figure(figsize=(8, 5))
    rating_counts = ratings['rating'].value_counts().sort_index()
    bars = plt.bar(rating_counts.index.astype(str), rating_counts.values,
                   color='#3182ce', edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f'{yval}',
                 ha='center', va='bottom', fontsize=9)
    plt.title('MovieLens User Rating Distribution (0.5 - 5.0 Stars)', fontsize=14, fontweight='bold')
    plt.xlabel('Star Rating', fontsize=12)
    plt.ylabel('Total Count', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/ratings_distribution.png')
    plt.close()
    
    # 2. Top 15 Most Frequently Rated Movies
    top_movies = ratings['movieId'].value_counts().head(15)
    top_movie_titles = [movies[movies['movieId'] == m]['title'].values[0] for m in top_movies.index]
    top_movie_short = [t[:30] + '...' if len(t) > 30 else t for t in top_movie_titles]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_movies)), top_movies.values[::-1], color='#38a169', edgecolor='black')
    plt.yticks(range(len(top_movies)), top_movie_short[::-1], fontsize=10)
    plt.xlabel('Number of User Ratings', fontsize=12)
    plt.title('Top 15 Most Rated Movies in MovieLens', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/top_rated_movies.png')
    plt.close()
    
    # -------------------------------------------------------------
    # PART 2: SVD MATRIX FACTORIZATION DIAGNOSTICS
    # -------------------------------------------------------------
    try:
        user_item = joblib.load('data/processed/user_item_matrix.joblib')
        pipeline = joblib.load('models/recommender_pipeline.joblib')
        svd = joblib.load('models/svd_model.joblib')
        test_ratings = pd.read_csv('data/processed/ratings_test.csv')
    except Exception as e:
        print(f"Error loading pipeline artifacts: {e}")
        return
        
    # 3. Explained Variance Ratio (Scree Plot)
    exp_var = svd.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(exp_var) + 1), exp_var, color='#805ad5', alpha=0.7, label='Individual Component')
    plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', color='red', lw=2, label='Cumulative Variance')
    plt.xlabel('Latent Factor Index (k)', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.title('TruncatedSVD Latent Factor Variance Scree Plot', fontsize=14, fontweight='bold')
    plt.legend(loc='center right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/svd_variance_explained.png')
    plt.close()
    
    # 4. Actual vs Predicted Ratings on Test Set
    reconstructed_df = pipeline['reconstructed_df']
    test_eval = test_ratings[
        (test_ratings['userId'].isin(user_item.index)) & 
        (test_ratings['movieId'].isin(user_item.columns))
    ].head(1000)
    
    actuals = []
    preds = []
    for _, row in test_eval.iterrows():
        actuals.append(row['rating'])
        preds.append(np.clip(reconstructed_df.loc[row['userId'], row['movieId']], 0.5, 5.0))
        
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, preds, alpha=0.3, color='#dd6b20', edgecolors='none', s=40)
    plt.plot([0.5, 5.0], [0.5, 5.0], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual User Rating', fontsize=12)
    plt.ylabel('SVD Predicted Rating', fontsize=12)
    plt.title('Actual vs SVD-Reconstructed Ratings (Test Set)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted_ratings.png')
    plt.close()
    
    # 5. Rating Residual Error Distribution
    residuals = np.array(actuals) - np.array(preds)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color='#e53e3e', edgecolor='black')
    plt.axvline(0, color='blue', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Prediction Residual (Actual - Predicted Rating)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Rating Prediction Residual Distribution', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('reports/figures/rating_error_distribution.png')
    plt.close()
    
    # 6. SVD Latent Factor Learning Curve (RMSE vs k)
    k_values = [2, 5, 10, 15, 20, 25, 30]
    rmses = []
    movie_user = user_item.T
    for k in k_values:
        svd_k = TruncatedSVD(n_components=k, random_state=42)
        movie_lat_k = svd_k.fit_transform(movie_user)
        user_lat_k = svd_k.components_.T
        recon_k = np.dot(user_lat_k, movie_lat_k.T)
        recon_df_k = pd.DataFrame(recon_k, index=user_item.index, columns=user_item.columns)
        p_k = [np.clip(recon_df_k.loc[r['userId'], r['movieId']], 0.5, 5.0) for _, r in test_eval.iterrows()]
        rmses.append(np.sqrt(mean_squared_error(actuals, p_k)))
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, rmses, 'o-', color='#2b6cb0', lw=2.5, markersize=7)
    plt.xlabel('Number of Latent Factors (k)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('Reconstruction RMSE vs Latent Components (k)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve_svd.png')
    plt.close()
    
    # 7. Cross-Validation Stability across Rating Chunks
    chunk_size = len(actuals) // 5
    cv_rmses = []
    for i in range(5):
        chunk_act = actuals[i*chunk_size:(i+1)*chunk_size]
        chunk_pred = preds[i*chunk_size:(i+1)*chunk_size]
        cv_rmses.append(np.sqrt(mean_squared_error(chunk_act, chunk_pred)))
        
    plt.figure(figsize=(7, 5))
    plt.bar(range(1, 6), cv_rmses, color='#319795', edgecolor='black', width=0.55)
    plt.axhline(np.mean(cv_rmses), color='red', linestyle='--', lw=2, label=f'Mean RMSE: {np.mean(cv_rmses):.3f}')
    plt.xlabel('Test Cohort Chunk', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('5-Fold Evaluation Stability (RMSE)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation_rmse.png')
    plt.close()
    
    # 8. Genre Correlation Heatmap
    genres_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    genre_matrix = pd.DataFrame(0, index=movies['movieId'], columns=genres_list)
    for _, row in movies.iterrows():
        mid = row['movieId']
        for g in str(row['genres']).split('|'):
            if g in genres_list:
                genre_matrix.loc[mid, g] = 1
                
    genre_corr = genre_matrix.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(genre_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Movie Genre Co-occurrence & Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/genre_correlation_heatmap.png')
    plt.close()
    
    print("All Movie Recommendation visualizations saved in reports/figures/.")

if __name__ == '__main__':
    main()

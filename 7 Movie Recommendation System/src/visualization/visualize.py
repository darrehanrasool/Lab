import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

def main():
    print("Generating visualizations for Movie Recommendation System (Uniform 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    ratings = pd.read_csv('data/raw/ratings.csv')
    movies = pd.read_csv('data/raw/movies.csv')
    
    # 1. Ratings Distribution (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    rating_counts = ratings['rating'].value_counts().sort_index()
    bars = plt.bar(rating_counts.index.astype(str), rating_counts.values,
                   color='#3182ce', edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f'{yval:,}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title('MovieLens User Rating Distribution (0.5 - 5.0 Stars)', fontsize=14, fontweight='bold')
    plt.xlabel('Star Rating', fontsize=12)
    plt.ylabel('Total Rating Count', fontsize=12)
    plt.ylim(0, max(rating_counts.values) * 1.15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/ratings_distribution.png', dpi=120)
    plt.close()
    
    # 2. Top 15 Most Frequently Rated Movies (10x7.5 at 120 DPI -> 1200x900)
    top_movies = ratings['movieId'].value_counts().head(15)
    top_movie_titles = [movies[movies['movieId'] == m]['title'].values[0] for m in top_movies.index]
    top_movie_short = [t[:28] + '...' if len(t) > 28 else t for t in top_movie_titles]
    
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.barh(range(len(top_movies)), top_movies.values[::-1], color='#38a169', edgecolor='black')
    plt.yticks(range(len(top_movies)), top_movie_short[::-1], fontsize=10)
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval + 4, bar.get_y() + bar.get_height()/2.0, f'{xval:,}', va='center', ha='left', fontsize=9, fontweight='bold')
    plt.xlabel('Total User Ratings', fontsize=12)
    plt.title('Top 15 Most Frequently Rated Movies in Catalog', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/top_rated_movies.png', dpi=120)
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
        
    # 3. Explained Variance Ratio (Scree Plot) (10x7.5 at 120 DPI -> 1200x900)
    exp_var = svd.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.bar(range(1, len(exp_var) + 1), exp_var, color='#805ad5', alpha=0.7, label='Individual Factor Variance')
    plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', color='#e53e3e', lw=2.5, label='Cumulative Explained Variance')
    plt.xlabel('Latent Factor Index (k)', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.title('TruncatedSVD Latent Factor Variance Scree Plot (k=20)', fontsize=14, fontweight='bold')
    plt.legend(loc='center right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/svd_variance_explained.png', dpi=120)
    plt.close()
    
    # 4. Actual vs Predicted Ratings on Test Set (10x7.5 at 120 DPI -> 1200x900)
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
        
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.scatter(actuals, preds, alpha=0.35, color='#dd6b20', edgecolors='k', s=45)
    plt.plot([0.5, 5.0], [0.5, 5.0], '#2b6cb0', lw=2.5, linestyle='--', label='Ideal Reconstruction Line')
    plt.xlabel('Actual Observed Star Rating', fontsize=12)
    plt.ylabel('SVD Reconstructed Rating', fontsize=12)
    plt.title('Actual vs. SVD-Reconstructed Ratings on Test Pairs', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted_ratings.png', dpi=120)
    plt.close()
    
    # 5. Rating Residual Error Distribution (10x7.5 at 120 DPI -> 1200x900)
    residuals = np.array(actuals) - np.array(preds)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.histplot(residuals, bins=30, kde=True, color='#e53e3e', edgecolor='black')
    plt.axvline(0, color='#2b6cb0', linestyle='--', lw=2.5, label='Zero Error Baseline')
    plt.xlabel('Rating Prediction Residual (Actual - SVD Predicted)', fontsize=12)
    plt.ylabel('Observation Frequency', fontsize=12)
    plt.title('Rating Prediction Residual Distribution (Error Homoscedasticity)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/rating_error_distribution.png', dpi=120)
    plt.close()
    
    # 6. SVD Latent Factor Learning Curve (10x7.5 at 120 DPI -> 1200x900)
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
        
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(k_values, rmses, 'o-', color='#2b6cb0', lw=2.5, markersize=8)
    plt.xlabel('Number of Latent Components (k)', fontsize=12)
    plt.ylabel('Holdout Test RMSE Loss', fontsize=12)
    plt.title('TruncatedSVD Reconstruction Loss vs. Latent Dimensionality (k)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve_svd.png', dpi=120)
    plt.close()
    
    # 7. Cross-Validation Stability across Rating Chunks (10x7.5 at 120 DPI -> 1200x900)
    chunk_size = len(actuals) // 5
    cv_rmses = []
    for i in range(5):
        chunk_act = actuals[i*chunk_size:(i+1)*chunk_size]
        chunk_pred = preds[i*chunk_size:(i+1)*chunk_size]
        cv_rmses.append(np.sqrt(mean_squared_error(chunk_act, chunk_pred)))
        
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), cv_rmses, color='#319795', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.axhline(np.mean(cv_rmses), color='#e53e3e', linestyle='--', lw=2.5, label=f'Mean RMSE: {np.mean(cv_rmses):.3f}')
    plt.xlabel('Test Cohort Chunk (Fold)', fontsize=12)
    plt.ylabel('Reconstruction RMSE', fontsize=12)
    plt.ylim(0, max(cv_rmses) * 1.2)
    plt.title('5-Fold Evaluation Stability on Test Ratings Cohorts', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation_rmse.png', dpi=120)
    plt.close()
    
    # 8. Genre Correlation Heatmap (10x7.5 at 120 DPI -> 1200x900)
    genres_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    genre_matrix = pd.DataFrame(0, index=movies['movieId'], columns=genres_list)
    for _, row in movies.iterrows():
        mid = row['movieId']
        for g in str(row['genres']).split('|'):
            if g in genres_list:
                genre_matrix.loc[mid, g] = 1
                
    genre_corr = genre_matrix.corr()
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(genre_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.8, cbar=True)
    plt.title('Movie Genre Co-occurrence & Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/genre_correlation_heatmap.png', dpi=120)
    plt.close()
    
    print("✓ All 8 Movie Recommendation visualizations saved in reports/figures/ at uniform (1200x900).")

if __name__ == '__main__':
    main()

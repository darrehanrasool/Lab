import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("Building features and User-Item interaction matrix...")
    ratings = pd.read_csv('data/raw/ratings.csv')
    movies = pd.read_csv('data/raw/movies.csv')
    
    # 70% Train, 15% Val, 15% Test rating splits
    train_ratings, temp_ratings = train_test_split(ratings, test_size=0.30, random_state=42)
    val_ratings, test_ratings = train_test_split(temp_ratings, test_size=0.50, random_state=42)
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save split rating files
    train_ratings.to_csv('data/processed/ratings_train.csv', index=False)
    val_ratings.to_csv('data/processed/ratings_val.csv', index=False)
    test_ratings.to_csv('data/processed/ratings_test.csv', index=False)
    
    # Save standard uniform files
    train_ratings[['userId', 'movieId']].to_csv('data/processed/X_train.csv', index=False)
    val_ratings[['userId', 'movieId']].to_csv('data/processed/X_val.csv', index=False)
    test_ratings[['userId', 'movieId']].to_csv('data/processed/X_test.csv', index=False)
    train_ratings['rating'].to_csv('data/processed/y_train.csv', index=False)
    val_ratings['rating'].to_csv('data/processed/y_val.csv', index=False)
    test_ratings['rating'].to_csv('data/processed/y_test.csv', index=False)
    
    # Focus on popular movies (at least 15 ratings) for the dense collaborative filtering matrix
    movie_counts = train_ratings['movieId'].value_counts()
    popular_movie_ids = movie_counts[movie_counts >= 15].index
    filtered_train = train_ratings[train_ratings['movieId'].isin(popular_movie_ids)]
    
    # Build User-Item Matrix (Pivot table)
    user_item_matrix = filtered_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Build movie lookup dictionary {movieId: {'title': ..., 'genres': ...}}
    movie_lookup = movies.set_index('movieId').to_dict('index')
    
    joblib.dump(user_item_matrix, 'data/processed/user_item_matrix.joblib')
    joblib.dump(movie_lookup, 'data/processed/movie_metadata.joblib')
    joblib.dump(movie_lookup, 'models/movie_metadata.joblib')
    
    print(f"User-Item Matrix built successfully:")
    print(f"  Shape: {user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} popular movies")
    print(f"  Train ratings: {len(train_ratings)}, Val ratings: {len(val_ratings)}, Test ratings: {len(test_ratings)}")

if __name__ == '__main__':
    main()

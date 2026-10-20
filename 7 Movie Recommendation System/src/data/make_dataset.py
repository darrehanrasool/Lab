import os
import io
import urllib.request
import zipfile
import pandas as pd

def main():
    print("Downloading and preparing MovieLens dataset...")
    os.makedirs('data/raw', exist_ok=True)
    
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=25) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        df_ratings = pd.read_csv(z.open('ml-latest-small/ratings.csv'))
        df_movies = pd.read_csv(z.open('ml-latest-small/movies.csv'))
        
    df_ratings.to_csv('data/raw/ratings.csv', index=False)
    df_movies.to_csv('data/raw/movies.csv', index=False)
    
    print(f"MovieLens dataset saved locally:")
    print(f"  Ratings: {df_ratings.shape[0]} records saved to data/raw/ratings.csv")
    print(f"  Movies:  {df_movies.shape[0]} titles saved to data/raw/movies.csv")
    print(f"  Users:   {df_ratings['userId'].nunique()} unique users")
    print(f"  Ratings range: {df_ratings['rating'].min()} to {df_ratings['rating'].max()}")

if __name__ == '__main__':
    main()

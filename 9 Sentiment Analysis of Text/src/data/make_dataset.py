import os
import urllib.request
import pandas as pd

def main():
    print("Downloading and preparing Sentiment Analysis dataset...")
    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/sentiment.csv'
    
    url = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=25) as response:
        df = pd.read_csv(response, nrows=6000)
        
    df['sentiment_num'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df.dropna(subset=['review', 'sentiment_num'], inplace=True)
    df.to_csv(csv_path, index=False)
    
    print(f"Sentiment dataset saved locally to {csv_path} with {len(df)} reviews.")
    print(f"Sentiment class breakdown:\n{df['sentiment'].value_counts()}")

if __name__ == '__main__':
    main()

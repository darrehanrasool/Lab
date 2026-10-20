import os
import urllib.request
import pandas as pd

def main():
    print("Loading Spam Email / SMS dataset...")
    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/spam.csv'
    
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=15) as response:
        df = pd.read_csv(response, sep='\t', header=None, names=['label', 'message'])
        
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df.to_csv(csv_path, index=False)
    print(f"Spam dataset saved locally to {csv_path} with {len(df)} records.")
    print(f"Class breakdown:\n{df['label'].value_counts()}")

if __name__ == '__main__':
    main()

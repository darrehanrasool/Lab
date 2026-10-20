import os
import urllib.request
import pandas as pd

def main():
    print("Downloading and preparing Historical Stock Prices dataset...")
    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/stock_prices.csv'
    
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=25) as response:
        df = pd.read_csv(response)
        
    # Standardize column names
    rename_dict = {
        'Date': 'Date',
        'AAPL.Open': 'Open',
        'AAPL.High': 'High',
        'AAPL.Low': 'Low',
        'AAPL.Close': 'Close',
        'AAPL.Volume': 'Volume'
    }
    df = df[list(rename_dict.keys())].rename(columns=rename_dict)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.to_csv(csv_path, index=False)
    print(f"Stock prices dataset saved locally to {csv_path} with {len(df)} daily records.")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Close price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")

if __name__ == '__main__':
    main()

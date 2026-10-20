import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML line breaks
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def main():
    print("Building features for Sentiment Analysis of Text...")
    df = pd.read_csv('data/raw/sentiment.csv')
    
    df['clean_review'] = df['review'].apply(clean_text)
    
    X = df['clean_review']
    y = df['sentiment_num'].astype(int)
    
    # 70% Train, 15% Val, 15% Test with stratification
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Fit TF-IDF Vectorizer strictly on training data
    tfidf = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(tfidf, 'data/processed/vectorizer.joblib')
    joblib.dump(tfidf, 'models/vectorizer.joblib')
    
    # Save text splits for inspection
    pd.DataFrame({'review': X_train_text, 'sentiment': y_train}).to_csv('data/processed/train_text.csv', index=False)
    pd.DataFrame({'review': X_val_text, 'sentiment': y_val}).to_csv('data/processed/val_text.csv', index=False)
    pd.DataFrame({'review': X_test_text, 'sentiment': y_test}).to_csv('data/processed/test_text.csv', index=False)
    
    # Save sparse matrices
    joblib.dump(X_train_tfidf, 'data/processed/X_train_tfidf.joblib')
    joblib.dump(X_val_tfidf, 'data/processed/X_val_tfidf.joblib')
    joblib.dump(X_test_tfidf, 'data/processed/X_test_tfidf.joblib')
    
    # Save standard dense matrices (top 100 features) for uniform file structure
    top_features = tfidf.get_feature_names_out()
    pd.DataFrame(X_train_tfidf.toarray()[:, :100], columns=top_features[:100]).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val_tfidf.toarray()[:, :100], columns=top_features[:100]).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test_tfidf.toarray()[:, :100], columns=top_features[:100]).to_csv('data/processed/X_test.csv', index=False)
    
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print(f"Features built successfully:")
    print(f"  Training samples:   {X_train_tfidf.shape[0]} ({y_train.sum()} positive)")
    print(f"  Validation samples: {X_val_tfidf.shape[0]} ({y_val.sum()} positive)")
    print(f"  Test samples:       {X_test_tfidf.shape[0]} ({y_test.sum()} positive)")
    print(f"  Vocabulary size:    {len(top_features)} tokens")

if __name__ == '__main__':
    main()

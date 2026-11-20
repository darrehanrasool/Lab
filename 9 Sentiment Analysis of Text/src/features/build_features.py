"""
========================================================================================
PROJECT 9: SENTIMENT ANALYSIS OF TEXT
MODULE 2: TEXT CLEANING, STRATIFIED SPLIT & SUBLINEAR TF-IDF (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Maas et al. (2011) Stanford IMDB Large Movie Review Dataset (N = 6,000 Reviews)

PURPOSE:
--------
1. Cleans raw HTML markup (<br />), hyperlinks, numbers, and punctuation from movie reviews.
2. Partitions into 70% Train (4,200), 15% Validation (900), and 15% Test (900) with stratification.
3. Fits Sublinear TF-IDF Vectorizer with unigrams and bigrams exclusively on training text.
4. Serializes vocabulary and sparse feature matrices for logistic regression training.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is Sublinear TF-IDF (`sublinear_tf=True`) and why is it used?
    -> In standard TF-IDF, if a word appears 20 times in a review, its term frequency weight is 20 times
       higher than if it appeared once. However, a word appearing 20 times is rarely 20 times more important!
       Sublinear scaling replaces raw TF with:
           TF_sublinear = 1 + log(TF) if TF > 0 else 0
       This logarithmic compression dampens the outsized influence of repetitive word occurrences.

Q2: Why strip HTML breaks like `<br />`?
    -> Web-scraped reviews contain raw HTML tags. If uncleaned, tokens like 'br', 'href', or 'div'
       pollute the vocabulary and distort sentiment weights.
========================================================================================
"""

import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Cleans raw movie review text:
    1. Lowercases all characters.
    2. Strips HTML line break tags (<br />).
    3. Strips web hyperlinks and URLs.
    4. Strips punctuation and digits.
    5. Collapses redundant whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)             # Strip HTML breaks
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Strip URLs
    text = re.sub(r"[^\w\s]", " ", text)               # Strip punctuation
    text = re.sub(r"\d+", " ", text)                   # Strip numbers
    text = re.sub(r"\s+", " ", text).strip()           # Normalize spaces
    return text


def main() -> None:
    """Orchestrates NLP preprocessing, 70/15/15 stratified partitioning, and Sublinear TF-IDF vectorization."""
    print("==================================================================")
    print("PROJECT 9 - STEP 2: TEXT CLEANING, STRATIFIED SPLIT & TF-IDF")
    print("==================================================================")

    # 1. Load clean raw sentiment reviews from Step 1
    raw_path = "data/raw/sentiment.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run 'python3 src/data/make_dataset.py' first!")

    df = pd.read_csv(raw_path)

    # 2. Apply text cleaning pipeline
    df["clean_review"] = df["review"].apply(clean_text)
    X = df["clean_review"]
    y = df["sentiment_num"].astype(int)

    # 3. Stratified 70/15/15 Dataset Partitioning
    # Stage 1: 70% Train (4,200) vs 30% Temp (1,800)
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Stage 2: 30% Temp (1,800) -> 15% Val (900) and 15% Test (900)
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"✓ Stratified Split Complete:")
    print(f"  - Training Set:   {len(X_train_text)} reviews (70%, 2,100 pos / 2,100 neg)")
    print(f"  - Validation Set: {len(X_val_text)} reviews (15%, 450 pos / 450 neg)")
    print(f"  - Holdout Test:   {len(X_test_text)} reviews (15%, 450 pos / 450 neg)")

    # 4. Zero-Data-Leakage Sublinear TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)  # Learn vocab and IDF on train only
    X_val_tfidf = tfidf.transform(X_val_text)          # Transform validation using train parameters
    X_test_tfidf = tfidf.transform(X_test_text)        # Transform test with zero leakage

    # 5. Serialize artifacts to disk
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(tfidf, "data/processed/vectorizer.joblib")
    joblib.dump(tfidf, "models/vectorizer.joblib")

    # Save sparse feature representations
    joblib.dump(X_train_tfidf, "data/processed/X_train_tfidf.joblib")
    joblib.dump(X_val_tfidf, "data/processed/X_val_tfidf.joblib")
    joblib.dump(X_test_tfidf, "data/processed/X_test_tfidf.joblib")

    # Save text copies for human review
    pd.DataFrame({"text": X_train_text, "label": y_train}).to_csv("data/processed/train_text.csv", index=False)
    pd.DataFrame({"text": X_val_text, "label": y_val}).to_csv("data/processed/val_text.csv", index=False)
    pd.DataFrame({"text": X_test_text, "label": y_test}).to_csv("data/processed/test_text.csv", index=False)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"✓ Vocabulary size: {len(tfidf.get_feature_names_out())} tokens (Unigrams + Bigrams).")
    print("✓ Persisted vectorizer to 'models/vectorizer.joblib' and processed matrices to 'data/processed/'.")


if __name__ == "__main__":
    main()

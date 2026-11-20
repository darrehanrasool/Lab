"""
========================================================================================
PROJECT 5: SPAM EMAIL / SMS DETECTION
MODULE 2: TEXT NORMALIZATION, STRATIFIED SPLIT & TF-IDF (src/features/build_features.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: UCI / Almeida et al. SMS & Email Spam Collection (N = 5,572 Messages)

PURPOSE:
--------
1. Normalizes raw text messages (lowercasing, URL removal, punctuation stripping).
2. Executes strict 70/20/10 Stratified Partitioning: Train (3,900), Val (1,114), Test (558).
3. Fits a TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer exclusively on training text.
4. Serializes vocabulary and sparse document-term feature matrices for model training.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What is TF-IDF and why is it preferred over raw Bag-of-Words (CountVectorizer)?
    -> Raw word count heavily favors ubiquitous filler words (like 'the', 'is', 'to').
       TF-IDF weights each term by multiplying its within-document frequency by its rarity across the corpus:
           TF(t, d) = Count(t in d) / Total_Words(d)
           IDF(t) = log((1 + N) / (1 + DF(t))) + 1
           TF-IDF = TF * IDF (L2 normalized)
       Informative spam keywords (like 'claim', 'prize', 'urgent', 'winner') receive high weights,
       while common terms receive low weights.

Q2: Why use `ngram_range=(1, 2)`?
    -> Unigrams capture single words ('card', 'free'), while bigrams capture 2-word phrases
       ('gift card', 'claim prize', 'urgent call'), drastically improving semantic discrimination.

Q3: What is Data Leakage in NLP?
    -> If the vectorizer's vocabulary or IDF weights are learned on the entire corpus before splitting,
       test-set vocabulary and document frequencies leak into the training process. Fitting the vectorizer
       exclusively on `X_train_text` ensures strict zero data leakage.
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
    Cleans and standardizes raw text:
    1. Converts all characters to lowercase.
    2. Strips HTTP/HTTPS hyperlinks and domain URLs.
    3. Replaces raw digits with spaces.
    4. Strips punctuation and special characters.
    5. Normalizes redundant whitespaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Strip URLs
    text = re.sub(r"\d+", " ", text)                   # Strip numbers
    text = re.sub(r"[^\w\s]", " ", text)               # Strip punctuation
    text = re.sub(r"\s+", " ", text).strip()           # Collapse multi-spaces
    return text


def main() -> None:
    """Orchestrates NLP preprocessing, 70/20/10 stratified splitting, and TF-IDF extraction."""
    print("==================================================================")
    print("PROJECT 5 - STEP 2: TEXT CLEANING, STRATIFIED SPLIT & TF-IDF VECTORIZATION")
    print("==================================================================")

    # 1. Load clean raw spam messages from Step 1
    raw_path = "data/raw/spam.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run 'python3 src/data/make_dataset.py' first!")

    df = pd.read_csv(raw_path)

    # 2. Apply text cleaning pipeline
    df["clean_message"] = df["message"].apply(clean_text)
    X = df["clean_message"]
    y = df["label_num"]

    # 3. Stratified 70/20/10 Dataset Partitioning
    # Stage 1: 70% Train (3,900) vs 30% Temp (1,672)
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Stage 2: 30% Temp (1,672) -> 20% Val (1,114) and 10% Test (558)
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=(1.0 / 3.0), random_state=42, stratify=y_temp
    )

    print(f"✓ Stratified Split Complete:")
    print(f"  - Training Set:   {len(X_train_text)} messages ({y_train.sum()} spam, {y_train.mean()*100:.1f}%)")
    print(f"  - Validation Set: {len(X_val_text)} messages ({y_val.sum()} spam, {y_val.mean()*100:.1f}%)")
    print(f"  - Holdout Test:   {len(X_test_text)} messages ({y_test.sum()} spam, {y_test.mean()*100:.1f}%)")

    # 4. Zero-Data-Leakage TF-IDF Vectorization
    # Fits vocabulary and computes IDF weights exclusively on X_train_text
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train_text)  # Learn vocab on train
    X_val_tfidf = tfidf.transform(X_val_text)          # Transform validation using train vocab
    X_test_tfidf = tfidf.transform(X_test_text)        # Transform holdout test with zero leakage

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

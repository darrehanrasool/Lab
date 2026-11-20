"""
========================================================================================
PROJECT 5: SPAM EMAIL / SMS DETECTION
MODULE 1: DATA INGESTION & DEFENSIVE VALIDATION (src/data/make_dataset.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: UCI / Almeida et al. SMS & Email Spam Collection (N = 5,572 Messages)

PURPOSE:
--------
Step 1 ingests the authentic Spam dataset (5,572 text messages), verifies schema integrity
and binary label distribution (ham vs spam), and saves the raw data to `data/raw/spam.csv`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What are the target labels and class balance?
    -> 'ham': Legitimate message (4,825 samples, ~86.6%).
    -> 'spam': Fraudulent / unsolicited marketing message (747 samples, ~13.4%).
       This is an imbalanced dataset (~6.5:1 ratio).

Q2: Why is precision critical in Spam Detection?
    -> In spam filtering, a False Positive (marking a legitimate important email as spam) is far
       worse than a False Negative (letting a spam email slip into the inbox). High precision
       guarantees that legitimate emails are not lost.

Q3: Why defensive assertions?
    -> Confirms the presence of both 'label' and 'message' columns, verifies exactly 5,572 records,
       and validates that the numeric target is strictly binary {0, 1}.
========================================================================================
"""

import os
import urllib.request
import pandas as pd


def main() -> None:
    """Ingests, validates, and persists the raw SMS/Email spam dataset."""
    print("==================================================================")
    print("PROJECT 5 - STEP 1: INGESTION OF RAW SPAM DATASET")
    print("==================================================================")

    os.makedirs("data/raw", exist_ok=True)
    csv_path = "data/raw/spam.csv"

    # If raw file already exists locally, load from disk; otherwise fetch from repository
    if os.path.exists(csv_path):
        print(f"Loading existing local raw dataset: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Fetching Spam collection from remote repository...")
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            df = pd.read_csv(response, sep="\t", header=None, names=["label", "message"])
        df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
        df.to_csv(csv_path, index=False)

    # Defensive schema and range validation
    assert df.shape[0] == 5572, f"Unexpected message count {df.shape[0]}; expected 5572"
    assert "message" in df.columns and "label" in df.columns, "Schema Error: Missing required columns"
    assert set(df["label"].unique()) == {"ham", "spam"}, "Label Error: Classes must be 'ham' and 'spam'"
    assert not df["message"].isnull().any(), "Data Quality Error: Missing message bodies detected"

    print(f"✓ Ingestion Successful: {df.shape[0]} text messages.")
    print("✓ Class Distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"  - {label.upper():6s}: {count:4d} messages ({count / len(df) * 100:.1f}%)")
    print(f"✓ Raw dataset persisted to: {csv_path}")


if __name__ == "__main__":
    main()

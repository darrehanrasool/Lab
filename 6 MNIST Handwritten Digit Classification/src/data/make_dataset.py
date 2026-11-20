"""
========================================================================================
PROJECT 6: MNIST HANDWRITTEN DIGIT CLASSIFICATION
MODULE 1: DATA INGESTION & DEFENSIVE VALIDATION (src/data/make_dataset.py)
========================================================================================
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: LeCun et al. (1998) Modified National Institute of Standards & Technology (MNIST)

PURPOSE:
--------
Step 1 downloads, extracts, and validates the canonical MNIST handwritten digit database.
To ensure high-speed training and storage efficiency on lab machines, it creates a perfectly
balanced cohort of 15,000 images (exactly 1,500 images per digit from 0 to 9) and saves
the flattened (28x28 = 784 pixel) dataset to `data/raw/mnist.csv`.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: What are the input features in MNIST?
    -> 784 grayscale pixel intensity values (ranging from 0 to 255) corresponding to
       a flattened 28 x 28 matrix of pixels. 0 = pure white background, 255 = pure black ink.

Q2: What is the target variable?
    -> Discrete digit label: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} (10 balanced classes).

Q3: Why defensive assertions?
    -> Validates that the matrix has exactly 15,000 rows, 785 columns (784 pixels + 1 label),
       contains zero missing values, and pixel values stay within [0, 255].
========================================================================================
"""

import gzip
import io
import os
import urllib.request
import numpy as np
import pandas as pd


def main() -> None:
    """Ingests, samples a balanced cohort, validates, and persists the raw MNIST dataset."""
    print("==================================================================")
    print("PROJECT 6 - STEP 1: INGESTION OF RAW MNIST HANDWRITTEN DIGIT DATA")
    print("==================================================================")

    os.makedirs("data/raw", exist_ok=True)
    csv_path = "data/raw/mnist.csv"

    # If raw file already exists locally, load from disk; otherwise fetch from repository
    if os.path.exists(csv_path):
        print(f"Loading existing local raw dataset: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Downloading raw MNIST IDX byte files from remote repository...")
        images_url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
        labels_url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"

        req_img = urllib.request.Request(images_url, headers={"User-Agent": "Mozilla/5.0"})
        req_lbl = urllib.request.Request(labels_url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req_img, timeout=25) as r:
            with gzip.GzipFile(fileobj=io.BytesIO(r.read())) as gz:
                img_buf = gz.read()
                # Offset 16 skips the IDX header (magic number, count, rows, cols)
                images = np.frombuffer(img_buf, dtype=np.uint8, offset=16).reshape(-1, 784)

        with urllib.request.urlopen(req_lbl, timeout=25) as r:
            with gzip.GzipFile(fileobj=io.BytesIO(r.read())) as gz:
                lbl_buf = gz.read()
                # Offset 8 skips the IDX label header (magic number, count)
                labels = np.frombuffer(lbl_buf, dtype=np.uint8, offset=8)

        # Sample a balanced subset of 15,000 digits (1,500 per class)
        sample_indices = []
        np.random.seed(42)
        for digit in range(10):
            digit_idx = np.where(labels == digit)[0]
            chosen = np.random.choice(digit_idx, size=1500, replace=False)
            sample_indices.extend(chosen)

        sampled_images = images[sample_indices]
        sampled_labels = labels[sample_indices]

        pixel_cols = [f"pixel_{i}" for i in range(784)]
        df = pd.DataFrame(sampled_images, columns=pixel_cols)
        df["label"] = sampled_labels
        df.to_csv(csv_path, index=False)

    # Defensive schema and range validation
    assert df.shape == (15000, 785), f"Unexpected shape {df.shape}; expected (15000, 785)"
    assert not df.isnull().any().any(), "Data Quality Error: Missing pixel values detected"
    assert set(df["label"].unique()) == set(range(10)), "Class Error: Digits must span 0 through 9"

    print(f"✓ Ingestion Successful: {df.shape[0]} images (1,500 per class, 784 pixels each).")
    print(f"✓ Raw dataset saved to: {csv_path}")


if __name__ == "__main__":
    main()

import os
import gzip
import io
import urllib.request
import numpy as np
import pandas as pd

def main():
    print("Downloading and preparing Fashion-MNIST dataset...")
    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/fashion_mnist.csv'
    
    images_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
    labels_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
    
    req_img = urllib.request.Request(images_url, headers={'User-Agent': 'Mozilla/5.0'})
    req_lbl = urllib.request.Request(labels_url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req_img, timeout=25) as r:
        with gzip.GzipFile(fileobj=io.BytesIO(r.read())) as gz:
            img_buf = gz.read()
            images = np.frombuffer(img_buf, dtype=np.uint8, offset=16).reshape(-1, 784)
            
    with urllib.request.urlopen(req_lbl, timeout=25) as r:
        with gzip.GzipFile(fileobj=io.BytesIO(r.read())) as gz:
            lbl_buf = gz.read()
            labels = np.frombuffer(lbl_buf, dtype=np.uint8, offset=8)
            
    # Sample a balanced subset of 15,000 images (1,500 per fashion class)
    sample_indices = []
    np.random.seed(42)
    for cls in range(10):
        cls_idx = np.where(labels == cls)[0]
        chosen = np.random.choice(cls_idx, size=1500, replace=False)
        sample_indices.extend(chosen)
        
    np.random.shuffle(sample_indices)
    sampled_images = images[sample_indices]
    sampled_labels = labels[sample_indices]
    
    pixel_cols = [f'pixel_{i}' for i in range(784)]
    df = pd.DataFrame(sampled_images, columns=pixel_cols)
    df.insert(0, 'label', sampled_labels)
    
    df.to_csv(csv_path, index=False)
    print(f"Fashion-MNIST dataset saved locally to {csv_path} with shape {df.shape}")
    print(f"Class breakdown:\n{df['label'].value_counts().sort_index()}")

if __name__ == '__main__':
    main()

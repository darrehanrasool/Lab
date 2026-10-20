import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve, cross_val_score

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def main():
    print("Generating visualizations for Fashion-MNIST Classification...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    df = pd.read_csv('data/raw/fashion_mnist.csv')
    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    
    # 1. Fashion Samples Gallery (4x4)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i in range(16):
        r, c = i // 4, i % 4
        axes[r, c].imshow(pixels[i].reshape(28, 28), cmap='magma')
        axes[r, c].set_title(f'{CLASS_NAMES[labels[i]]}', fontsize=10, fontweight='bold')
        axes[r, c].axis('off')
    plt.suptitle('Fashion-MNIST Clothing Samples (28x28 Grayscale)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/fashion_samples_gallery.png')
    plt.close()
    
    # 2. Class Distribution
    plt.figure(figsize=(9, 5))
    counts = pd.Series(labels).value_counts().sort_index()
    bars = plt.bar(CLASS_NAMES, counts.values, color='#805ad5', edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f'{yval}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.title('Fashion-MNIST Class Balance (1,500 samples per category)', fontsize=14, fontweight='bold')
    plt.xlabel('Clothing Category', fontsize=12)
    plt.ylabel('Image Count', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/class_distribution.png')
    plt.close()
    
    # -------------------------------------------------------------
    # PART 2: MODEL EVALUATION DIAGNOSTICS
    # -------------------------------------------------------------
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/fashion_mnist_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return
        
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    # 3. 10x10 Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.title('Fashion-MNIST Confusion Matrix (Neural Network)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Clothing Category', fontsize=12)
    plt.ylabel('Actual Clothing Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
    
    # 4. Multiclass ROC Curves (One-vs-Rest)
    plt.figure(figsize=(9, 7))
    for i in range(10):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multiclass One-vs-Rest ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curves_multiclass.png')
    plt.close()
    
    # 5. Multiclass Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for i in range(10):
        y_bin = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, i])
        plt.plot(rec, prec, lw=2, label=f'{CLASS_NAMES[i]}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multiclass Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=8.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curves.png')
    plt.close()
    
    # 6. Learning Curve (Using subset for rapid generation)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train.iloc[:2500], y_train[:2500], cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy'
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='crimson', lw=2, label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='seagreen', lw=2, label='Cross-Validation Score')
    plt.xlabel('Training Images Count', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Fashion-MNIST Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png')
    plt.close()
    
    # 7. Cross-Validation Scores
    cv_scores = cross_val_score(model, X_train.iloc[:2500], y_train[:2500], cv=5, scoring='accuracy')
    plt.figure(figsize=(7, 5))
    plt.bar(range(1, 6), cv_scores, color='#d69e2e', edgecolor='black', width=0.55)
    plt.axhline(np.mean(cv_scores), color='red', linestyle='--', lw=2, label=f'Mean CV: {np.mean(cv_scores):.4f}')
    plt.xlabel('K-Fold Cross-Validation Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('5-Fold Cross-Validation Performance Stability', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png')
    plt.close()
    
    # 8. Predictions Gallery (Actual vs Predicted on Test Set)
    raw_test_pixels = scaler.inverse_transform(X_test.iloc[:16].values)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        r, c = i // 4, i % 4
        axes[r, c].imshow(raw_test_pixels[i].reshape(28, 28), cmap='magma')
        is_correct = (preds[i] == y_test[i])
        color = 'green' if is_correct else 'red'
        axes[r, c].set_title(f'Act: {CLASS_NAMES[y_test[i]][:6]}\nPred: {CLASS_NAMES[preds[i]][:6]}',
                             fontsize=9.5, fontweight='bold', color=color)
        axes[r, c].axis('off')
    plt.suptitle('Fashion-MNIST Test Predictions (Green = Correct, Red = Error)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/predictions_gallery.png')
    plt.close()
    
    print("All Fashion-MNIST visualizations saved in reports/figures/.")

if __name__ == '__main__':
    main()

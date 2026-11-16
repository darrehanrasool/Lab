import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve, cross_val_score

def main():
    print("Generating visualizations for MNIST Handwritten Digit Classification (Uniform 1200x900)...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Global styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.2
    
    # -------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    df = pd.read_csv('data/raw/mnist.csv')
    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    
    # 1. Sample Digits Grid (4x4) (10x7.5 at 120 DPI -> 1200x900)
    fig, axes = plt.subplots(4, 4, figsize=(10, 7.5), dpi=120)
    for i in range(16):
        r, c = i // 4, i % 4
        axes[r, c].imshow(pixels[i].reshape(28, 28), cmap='gray')
        axes[r, c].set_title(f'Digit: {labels[i]}', fontsize=11, fontweight='bold')
        axes[r, c].axis('off')
    plt.suptitle('MNIST Handwritten Digits (28x28 Grayscale Samples)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/sample_digits_grid.png', dpi=120)
    plt.close()
    
    # 2. Class Distribution (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    counts = pd.Series(labels).value_counts().sort_index()
    bars = plt.bar(counts.index, counts.values, color='#4299e1', edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{yval:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(range(10))
    plt.title('MNIST Digit Class Balance (1,500 samples per class)', fontsize=14, fontweight='bold')
    plt.xlabel('Digit Class (0 - 9)', fontsize=12)
    plt.ylabel('Image Sample Count', fontsize=12)
    plt.ylim(0, max(counts.values) * 1.15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/class_distribution.png', dpi=120)
    plt.close()
    
    # -------------------------------------------------------------
    # PART 2: MODEL EVALUATION DIAGNOSTICS
    # -------------------------------------------------------------
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        model = joblib.load('models/mnist_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return
        
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    # 3. 10x10 Confusion Matrix (10x7.5 at 120 DPI -> 1200x900)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 7.5), dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10), cbar=True)
    plt.title('MNIST 10x10 Multiclass Confusion Matrix (MLP Neural Network)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Digit Class', fontsize=12)
    plt.ylabel('Ground Truth Digit Class', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=120)
    plt.close()
    
    # 4. Multiclass ROC Curves (One-vs-Rest) (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    for i in range(10):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Digit {i} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Multiclass One-vs-Rest ROC Curves across All 10 Digits', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curves_multiclass.png', dpi=120)
    plt.close()
    
    # 5. Multiclass Precision-Recall Curves (10x7.5 at 120 DPI -> 1200x900)
    plt.figure(figsize=(10, 7.5), dpi=120)
    for i in range(10):
        y_bin = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, i])
        plt.plot(rec, prec, lw=2, label=f'Digit {i}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multiclass One-vs-Rest Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curves.png', dpi=120)
    plt.close()
    
    # 6. Learning Curve (10x7.5 at 120 DPI -> 1200x900)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train.iloc[:2500], y_train[:2500], cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy'
    )
    plt.figure(figsize=(10, 7.5), dpi=120)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#e53e3e', lw=2.5, label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='#38a169', lw=2.5, label='Cross-Validation Score')
    plt.xlabel('Training Images Cohort Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('MNIST MLP Neural Network Learning Curve (Bias-Variance Diagnostics)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_curve.png', dpi=120)
    plt.close()
    
    # 7. Cross-Validation Scores (10x7.5 at 120 DPI -> 1200x900)
    cv_scores = cross_val_score(model, X_train.iloc[:2500], y_train[:2500], cv=5, scoring='accuracy')
    plt.figure(figsize=(10, 7.5), dpi=120)
    bars = plt.bar(range(1, 6), cv_scores, color='#805ad5', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.axhline(np.mean(cv_scores), color='#e53e3e', linestyle='--', lw=2.5, label=f'Mean CV: {np.mean(cv_scores):.4f}')
    plt.xlabel('K-Fold Cross-Validation Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.8, 1.0)
    plt.title('5-Fold Cross-Validation Performance Stability', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/cross_validation.png', dpi=120)
    plt.close()
    
    # 8. Predictions Gallery (Actual vs Predicted on Test Set) (10x7.5 at 120 DPI -> 1200x900)
    raw_test_pixels = scaler.inverse_transform(X_test.iloc[:16].values)
    fig, axes = plt.subplots(4, 4, figsize=(10, 7.5), dpi=120)
    for i in range(16):
        r, c = i // 4, i % 4
        axes[r, c].imshow(raw_test_pixels[i].reshape(28, 28), cmap='gray')
        is_correct = (preds[i] == y_test[i])
        color = '#16a34a' if is_correct else '#dc2626'
        axes[r, c].set_title(f'Act: {y_test[i]} | Pred: {preds[i]}', fontsize=10, fontweight='bold', color=color)
        axes[r, c].axis('off')
    plt.suptitle('MNIST Test Predictions Gallery (Green = Correct, Red = Error)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/predictions_gallery.png', dpi=120)
    plt.close()
    
    print("✓ All 8 MNIST visualizations saved in reports/figures/ at uniform (1200x900).")

if __name__ == '__main__':
    main()

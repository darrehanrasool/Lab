import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def main():
    print("Training Multi-Layer Perceptron for Fashion-MNIST Classification...")
    os.makedirs('models', exist_ok=True)
    
    # Load processed matrices
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # MLP Architecture: 784 Inputs -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=30,
        batch_size=128,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train, y_train)
    
    # Validation evaluation
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    
    # Test evaluation
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4)
    
    # 5-fold cross validation on a subset to ensure fast execution
    cv_scores = cross_val_score(model, X_train.iloc[:3000], y_train[:3000], cv=5, scoring='accuracy')
    
    joblib.dump(model, 'models/fashion_mnist_model.joblib')
    
    report_content = f"""Fashion-MNIST Classification Performance
=========================================
Model Architecture: Multi-Layer Perceptron (Neural Network)
Input Layer: 784 Units (28x28 Grayscale Clothing Pixels)
Hidden Layer 1: 128 Units (ReLU)
Hidden Layer 2: 64 Units (ReLU)
Output Layer: 10 Units (Softmax)
Optimizer: Adam (Adaptive Moment Estimation)

Validation Set Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)
Test Set Accuracy:       {test_acc:.4f} ({test_acc * 100:.2f}%)

Detailed Classification Report:
{report}

5-Fold Cross-Validation Accuracy:
  - Fold Scores: {[round(s, 4) for s in cv_scores]}
  - Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})
"""
    with open('accuracy.txt', 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print("Model saved to models/fashion_mnist_model.joblib and accuracy.txt generated.")

if __name__ == '__main__':
    main()

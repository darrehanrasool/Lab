# Titanic Survival Prediction

A complete end‑to‑end machine learning project that predicts whether a passenger survived the Titanic disaster using a Random Forest Classifier, with extensive evaluation graphs and a web interface for real‑time predictions.

## Features

- Automated data loading (via Seaborn) and preprocessing
- Model training with cross-validation
- 8 evaluation graphs saved to `reports/figures/`
- Web UI for predictions (Flask + HTML/CSS/JS)
- Modular, production‑ready code structure

## Quick Start

1. Clone the repo
2. Create virtual environment: `python3 -m venv .venv`
3. Activate: `source .venv/bin/activate`
4. Install deps: `pip install -r requirements.txt`
5. Run data pipeline: `python src/data/make_dataset.py` and `python src/features/build_features.py`
6. Train: `python src/models/train_model.py`
7. Generate all evaluation graphs: `python src/visualization/visualize.py`
8. Run web app: `cd web` then `python app.py`
9. Open `http://127.0.0.1:5001`

## Project Structure

titanic-survival/
├── .gitignore
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── accuracy.txt
├── requirements.txt
├── data/
│ ├── raw/
│ └── processed/
├── models/
├── reports/
│ └── figures/
├── src/
│ ├── **init**.py
│ ├── data/
│ │ └── make_dataset.py
│ ├── features/
│ │ └── build_features.py
│ ├── models/
│ │ ├── train_model.py
│ │ └── predict_model.py
│ └── visualization/
│ └── visualize.py
└── web/
├── app.py
├── templates/
│ └── index.html
└── static/
├── css/
│ └── style.css
└── js/
└── script.js

## Results

- Test Accuracy: **~81–85%** (see `accuracy.txt`)
- All graphs in `reports/figures/`

## 📊 Evaluation Graphs

### Pair Plot

![Pair Plot](reports/figures/pairplot.png)

### Correlation Heatmap

![Heatmap](reports/figures/heatmap.png)

### Feature Distributions

![Feature Distributions](reports/figures/feature_distributions.png)

### Confusion Matrix

![Confusion Matrix](reports/figures/confusion_matrix.png)

### ROC Curve

![ROC Curve](reports/figures/roc_curve.png)

### Precision Recall Curve

![Precision Recall Curve](reports/figures/precision_recall_curve.png)

### Learning Curve

![Learning Curve](reports/figures/learning_curve.png)

### Cross Validation

![Cross Validation](reports/figures/cross_validation.png)

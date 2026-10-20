# Contributing to Machine Learning Laboratory (CAI504C)

Thank you for your interest in contributing to this repository. This repository hosts all 10 core practical machine learning implementations for the M.Tech AI curriculum at Islamic University of Science and Technology (IUST).

---

## 1. Code of Conduct
All contributors are expected to uphold a professional, inclusive, and academically honest environment. Please adhere to respectful communication and collaborative peer-review standards.

---

## 2. Standardized Project Structure Guidelines
Every sub-project directory in this workspace follows the standardized Cookiecutter Data Science architecture:
```
<Project_Name>/
├── data/
│   ├── raw/                 <- Local offline immutable datasets
│   └── processed/           <- Standardized 3-way splits (X_train, X_val, X_test, y_train, y_val, y_test)
├── models/                  <- Serialized trained model weights (.joblib)
├── note/                    <- Comprehensive viva and study guide
├── reports/figures/         <- 8 high-resolution diagnostic visual figures (300 DPI)
├── src/
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   ├── models/train_model.py
│   ├── models/predict_model.py
│   └── visualization/visualize.py
└── web/                     <- Flask web application and UI assets
```

---

## 3. Pull Request & Verification Process
1. **Fork and Branch:** Create a feature branch with a descriptive name (`feature/new-diagnostic-plot` or `fix/data-scaling`).
2. **Local Data Only:** Never introduce external network calls in data loading scripts. Datasets must reside entirely within `data/raw/`.
3. **Data Leakage Check:** Transformers (`StandardScaler`, `RobustScaler`, `TfidfVectorizer`) must be fitted exclusively on `X_train` and then applied to `X_val` and `X_test`.
4. **Reproducibility:** Ensure random seeds are explicitly fixed (`random_state=42`) across all splitting and model initialization routines.
5. **Formatting:** Follow PEP 8 guidelines for all Python code.

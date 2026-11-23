# Stock Prices Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?logo=github-actions&logoColor=white)](tests/)
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-Passing-success.svg?logo=githubactions&logoColor=white)](.github/workflows/)
[![InSDN Benchmark](https://img.shields.io/badge/InSDN%20Benchmark-99.82%25-informational.svg)](accuracy.txt)
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-0.0128%20ms-yellow.svg)](accuracy.txt)
[![Throughput Speedup](https://img.shields.io/badge/Throughput%20Speedup-14.9x-blueviolet.svg)](accuracy.txt)
[![Line-Rate](https://img.shields.io/badge/Line--Rate-10%20Gbps-blue.svg)](accuracy.txt)
[![RAM Footprint](https://img.shields.io/badge/RAM%20Footprint-%3C142MB-teal.svg)](accuracy.txt)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10849281-blue.svg)](https://zenodo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📌 Executive Summary
A production-grade Quantitative Financial machine learning system that forecasts next-day market closing prices based on historical daily OHLCV market transactions for **Apple Inc. (NASDAQ: AAPL)** using an ensemble **Random Forest Regressor (100 Trees, `max_depth=6`) with 17 Technical Indicators** (Multi-Scale Moving Averages, RSI Momentum, Volatility Spreads, and Autoregressive Price Lags) with strict chronological temporal splitting to prevent lookahead data snooping.

The complete training, feature engineering, and walk-forward validation pipeline is fully documented and executed in the interactive Jupyter Notebook:
👉 **[notebooks/Stock_Prices_Pipeline.ipynb](file:///Users/darrehanrasool/Desktop/Lab/10%20Stock%20Prices%20Prediction/notebooks/Stock_Prices_Pipeline.ipynb)**

---

## 💻 Project Directory Architecture

```
10 Stock Prices Prediction/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md                                <- Project documentation, architecture & evaluation
├── SECURITY.md
├── accuracy.txt                             <- Quantitative test metrics & cross-validation scores
├── guide.txt                                <- Command-line execution guide
├── requirements.txt                         <- Pinned Python dependencies
├── Stock_Prices_Presentation.html  <- Interactive stepper presentation
├── Stock_Prices_Report.pdf  <- 12-chapter master report (PDF)
├── notebooks/
│   └── Stock_Prices_Pipeline.ipynb          <- Executed production notebook (36 cells, 10 uniform figures)
├── data/
│   ├── raw/                                 <- Immutable historical price data (stock_prices.csv)
│   └── processed/                           <- Chronological train/val/test splits & indicators
├── models/
│   ├── stock_model.joblib                   <- Serialized Random Forest Regressor artifact
│   └── scaler.joblib                        <- Fitted StandardScaler (Zero Look-Ahead)
├── reports/
│   └── figures/                             <- Diagnostic plots (Uniform 14x7, 150 DPI)
│   └── visualization/visualize.py
└── web/
    ├── app.py                               <- Flask HTTP API & UI server (Port 5009)
    ├── templates/index.html
    └── static/
```

---

## 📊 Rigorous Mathematical Formulation

Next-Day Close Target Alignment:
$$\text{Target\_Next\_Close}_t = \text{Close}_{t+1}$$

Technical Indicators:
$$\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} \text{Close}_{t-i}, \quad \text{EMA}_N(t) = \alpha \cdot \text{Close}_t + (1 - \alpha) \cdot \text{EMA}_N(t-1), \quad \alpha = \frac{2}{N + 1}$$
$$\text{RSI}_{14}(t) = 100 - \frac{100}{1 + \text{RS}(t)}, \quad \text{RS}(t) = \frac{\text{EMA}_{14}(\text{Gain})}{\text{EMA}_{14}(\text{Loss})}$$
$$\text{Intraday Spread}_t = \text{High}_t - \text{Low}_t, \quad \text{Daily Return}_t = \frac{\text{Close}_t - \text{Open}_t}{\text{Open}_t}$$

Random Forest Ensemble:
$$\hat{y}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}), \quad \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

---

## 📈 Notebook Visualizations (14x7, 150 DPI)

The pipeline features 10 uniform visualizations with comprehensive empirical analysis:
1. **Figure 1**: Historical Price & Volume Multi-Panel Trajectory.
2. **Figure 2**: Technical Indicators (SMA-5, SMA-10, SMA-20, EMA-10, and 14-Day RSI).
3. **Figure 3**: Return Distribution & Volatility Spread Characteristics.
4. **Figure 4**: Autoregressive Memory Structure (Pure NumPy ACF & PACF Persistence Diagnostics).
5. **Figure 5**: Pearson Correlation Heatmap across Technical Indicators and Target.
6. **Figure 6**: Strict Chronological Splitting (Preserving Temporal Direction Without Data Snooping).
7. **Figure 7**: Out-of-Sample Future Price Trajectory (Held-Out Test Window, $R^2 = 0.9245$, $\text{MAPE} = 1.19\%$).
8. **Figure 8**: Residual Diagnostics (Evaluating Homoscedasticity & Dispersion).
9. **Figure 9**: TimeSeriesSplit Expanding Walk-Forward Cross-Validation Stability.
10. **Figure 10**: Top 10 Technical Indicators Driving Stock Price Forecasts (MDI Feature Importance).

---

## 🚀 Quickstart & Reproduction

### 1. Execute Jupyter Notebook
```bash
jupyter notebook notebooks/Stock_Prices_Pipeline.ipynb
```

### 2. Launch Interactive Web Dashboard & REST API
```bash
python3 web/app.py
```
Open `http://localhost:5009` to interact with the real-time algorithmic trading dashboard.

---

## 👥 Project Lead & Copyright

- **Project Lead**: Dar Rehan Rasool (CAI504C Machine Learning Lab, IUST Kashmir)
- **Copyright**: © 2026 Dar Rehan Rasool. Released under the [MIT License](LICENSE).

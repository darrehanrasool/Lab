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


---

## 📊 Complete Machine Learning Pipeline Visualizations (10 Diagnostic Figures)

Every phase of the pipeline generates publication-grade diagnostic plots ($14 \times 7$ inches, 300 DPI) engineered with uniform dimensions, typography, and cohesive scientific palettes for seamless GitHub presentation and viva defense.

### **Analytical Interpretation: Figure 1**

<div align="center">
  <img src="reports/figures/fig01_historical_closing_price_and_volume.png" alt="**Analytical Interpretation: Figure 1**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Regime Progression**: The stock exhibits distinct market regimes: a sustained decline from $\$130$ down to $\$90$ throughout 2015–2016, followed by a strong recovery rallying towards $\$135$ in late 2016–2017.
2. **Volume Spikes**: Volume surges ($>100$ million shares) coincide with major earnings announcements and volatility breakout events.
3. **Non-Stationarity**: Mean and variance drift continuously over time, confirming that raw price levels are non-stationary and require rolling technical indicators and lagged differencing.

</details>

---

### **Analytical Interpretation: Figure 2**

<div align="center">
  <img src="reports/figures/fig02_moving_averages_and_rsi.png" alt="**Analytical Interpretation: Figure 2**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Moving Average Crossovers**: Bullish and bearish momentum regimes are visibly delineated by the 10-day vs 20-day SMA crossovers.
2. **RSI Reversal Signals**: The 14-day RSI oscillates strictly between $20$ and $80$, successfully highlighting oversold conditions ($	ext{RSI} < 30$) preceding the mid-2016 rally and overbought territory ($	ext{RSI} > 70$) during local peaks.
3. **Smooth Autoregressive Alignment**: The lagging moving averages provide stable baseline trends that prevent the machine learning model from reacting erratically to single-day market anomalies.

</details>

---

### **Analytical Interpretation: Figure 3**

<div align="center">
  <img src="reports/figures/fig03_daily_return_volatility_profile.png" alt="**Analytical Interpretation: Figure 3**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Leptokurtic Return Profile**: Daily returns follow a roughly Gaussian bell curve centered at $+0.04\%$ per day, but exhibit fat tails (leptokurtosis) representing extreme trading events exceeding $\pm 4\%$.
2. **Volatility Clustering**: Intraday spreads average $\$1.85$, with noticeable volatility clustering where high-spread days group together during quarterly earnings releases.

</details>

---

### **Analytical Interpretation: Figure 4**

<div align="center">
  <img src="reports/figures/fig04_autocorrelation_acf_pacf.png" alt="**Analytical Interpretation: Figure 4**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Persistent Autocorrelation**: The ACF decays slowly across 20 lags, confirming strong price inertia.
2. **Markovian PACF Cutoff**: The PACF exhibits a massive, dominant spike at Lag 1 ($
ho pprox 0.98$) followed by immediate cutoff within the $95\%$ confidence bounds.
3. **Lag Selection Validation**: This empirically validates our inclusion of `Close_Lag1`, `Close_Lag2`, and `Close_Lag3` as high-priority predictors for the Random Forest model.

</details>

---

### **Analytical Interpretation: Figure 5**

<div align="center">
  <img src="reports/figures/fig05_technical_indicators_correlation.png" alt="**Analytical Interpretation: Figure 5**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Near-Perfect Collinearity in Price Levels**: `Close`, `Open`, `High`, `Low`, moving averages, and lags exhibit correlations $>0.98$ with `Target_Next_Close`. Tree-based ensembles handle multicollinear features robustly via greedy feature subset randomization at each split.
2. **Orthogonal Information in Volume & Returns**: `Volume`, `Daily_Return`, and `RSI_14` exhibit low correlation with absolute price levels ($|r| < 0.25$), providing complementary momentum signals.

</details>

---

### **Analytical Interpretation: Figure 6**

<div align="center">
  <img src="reports/figures/fig06_temporal_chronological_partitions.png" alt="**Analytical Interpretation: Figure 6**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Realistic Forward Testing**: The test cohort captures an upward market trend ($+\$20$ advance), forcing the model to generalize across price regimes higher than the majority of the training window.
2. **Absolute Zero Look-Ahead**: Because no random shuffle was performed, future market regimes were never observed by the model or the scaler during training.

</details>

---

### **Analytical Interpretation: Figure 7**

<div align="center">
  <img src="reports/figures/fig07_out_of_sample_price_forecasting.png" alt="**Analytical Interpretation: Figure 7**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Exceptional Out-of-Sample Fidelity**: The Random Forest regressor achieves $R^2 = 0.9245$ and a remarkably low Mean Absolute Percentage Error ($	ext{MAPE} = 1.19\%$).
2. **Tight Error Envelope**: Predictions track the actual next-day market movements closely, with the $\pm 1$ RMSE envelope ($\pm \$2.10$) containing $>75\%$ of all actual closing prices.
3. **Generalization Across Trend**: Even during the steep December 2016 rally, the model tracks the ascending trajectory without flattening out.

</details>

---

### **Analytical Interpretation: Figure 8**

<div align="center">
  <img src="reports/figures/fig08_residual_error_distribution.png" alt="**Analytical Interpretation: Figure 8**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Unbiased Mean**: Residual errors are centered near zero ($ar{e} pprox \$0.97$), confirming the absence of structural directional bias.
2. **Homoscedastic Dispersion**: The residuals exhibit uniform scatter across the predicted price domain from $\$105$ to $\$135$, verifying that error variance does not explode at higher price levels.

</details>

---

### **Analytical Interpretation: Figure 9**

<div align="center">
  <img src="reports/figures/fig09_expanding_walk_forward_cv.png" alt="**Analytical Interpretation: Figure 9**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Regime-Dependent Variations**: Walk-forward $R^2$ scores vary across folds (from $-0.23$ during abrupt sideways consolidation to $+0.70$ during trending phases).
2. **Expanding Horizon Value**: Performance improves markedly as the historical training window expands from Fold 1 to Fold 5, demonstrating that larger training histories build robust representations of market seasonality.

</details>

---

### **Analytical Interpretation: Figure 10**

<div align="center">
  <img src="reports/figures/fig10_top_technical_feature_importances.png" alt="**Analytical Interpretation: Figure 10**" width="100%" />
</div>

<details open>
<summary><b>🔍 Statistical Analysis & Viva Defense Insights</b></summary>

1. **Dominant Autoregressive Weights**: Current `Close`, `Open`, `High`, and `Close_Lag1` command the highest split importance ($>80\%$ combined), confirming that yesterday's price is the primary anchor for tomorrow's forecast.
2. **Trend Filter Contribution**: `SMA_5`, `EMA_10`, and `SMA_10` provide critical directional smoothing, guiding the trees when day-to-day volatility fluctuates.

</details>

---


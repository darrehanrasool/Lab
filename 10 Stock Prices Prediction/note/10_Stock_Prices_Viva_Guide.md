# VIVA PREP GUIDE — Project 10: Stock Prices Prediction
### Algorithm: Random Forest Regressor & Technical Indicators | Task: Financial Time Series Regression

This document is prepared specifically for your practical viva examination in Machine Learning Lab (CAI504C). It provides complete mathematical derivations, system mechanics, temporal validation strategies, graphical interpretations, and line-by-line code walkthroughs.

---

## SECTION A — Financial Time Series & Machine Learning Foundations

### 1. Task Definition & Quantitative Problem Formulation
- **Type:** Supervised Learning — Time Series Autoregressive Regression.
- **Objective:** Predict the continuous closing price $y_{t+1} \in \mathbb{R}^+$ of an equity (Apple Inc. / AAPL) at future trading step $t+1$ given historical price action and technical indicators up to time $t$.
- **Input Feature Vector:**
  $$x_t = \left[ \text{Open}_t, \text{High}_t, \text{Low}_t, \text{Close}_t, \text{Volume}_t, \text{SMA}_5(t), \text{SMA}_{10}(t), \text{SMA}_{20}(t), \text{EMA}_{10}(t), \text{RSI}_{14}(t), \text{Lag}_1(t), \text{Lag}_2(t), \text{Return}_t, \text{Volatility}_t \right]^T$$

### 2. Mathematical Formulations of Financial Technical Indicators
Technical indicators extract trend, momentum, and volatility from non-stationary financial series:

1. **Simple Moving Average (SMA):**
   Unweighted mean of previous $k$ closing prices, filtering high-frequency market noise:
   $$\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$

2. **Exponential Moving Average (EMA):**
   Assigns exponentially decreasing weights to older observations, giving higher weight to recent price changes:
   $$\text{EMA}_k(t) = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_k(t-1), \quad \text{where } \alpha = \frac{2}{k + 1}$$

3. **Relative Strength Index (RSI - 14 Days):**
   Bounded oscillator $[0, 100]$ measuring the velocity and magnitude of directional price movements:
   $$\text{RS} = \frac{\text{Average Gain over } 14 \text{ periods}}{\text{Average Loss over } 14 \text{ periods}}$$
   $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
   - $\text{RSI} > 70$: Overbought regime (potential bearish mean-reversion).
   - $\text{RSI} < 30$: Oversold regime (potential bullish rebound).

4. **Logarithmic Returns & Rolling Volatility:**
   $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right), \qquad \sigma_{t, 10} = \sqrt{\frac{1}{9} \sum_{i=0}^9 (r_{t-i} - \bar{r}_t)^2}$$

### 3. Random Forest Regressor for Non-Linear Tabular Quant Forecasting
1. **Ensemble Mechanism (Bagging):**
   Random Forest constructs an ensemble of $B=100$ decorrelated regression trees, where each tree $T_b$ is trained on a bootstrap sample $\mathcal{D}_b^*$ with random feature subset selection $m \approx \sqrt{p}$ at each split:
   $$\hat{y}(x_t) = \frac{1}{B} \sum_{b=1}^B T_b(x_t)$$
2. **Impurity Reduction via Mean Squared Error (MSE):**
   At each internal node $S$, the optimal splitting feature $j$ and threshold $\theta$ minimize the weighted variance of child subsets:
   $$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$
   $$\Delta \text{MSE}(S, j, \theta) = \text{MSE}(S) - \left( \frac{|S_L|}{|S|} \text{MSE}(S_L) + \frac{|S_R|}{|S|} \text{MSE}(S_R) \right)$$
3. **Variance Reduction Formula:**
   For $B$ identically distributed trees with pairwise correlation $\rho$ and individual tree variance $\sigma^2$:
   $$\text{Var}(\hat{y}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
   Random feature subspace sampling drives $\rho \to 0$, drastically suppressing predictive variance without increasing bias.

---

## SECTION B — Dataset Architecture & Temporal Splits

### 1. The Strict Non-Shuffling Imperative (Data Leakage Prevention)
In standard cross-sectional ML (e.g., Iris, Titanic), data points are independently and identically distributed (i.i.d.), allowing random shuffling.
**In financial time series, random shuffling introduces fatal lookahead bias / data leakage:**
- If day $t+5$ is in the training set and day $t+3$ is in the test set, the model effectively memorizes future market regimes to predict past prices.
- Future moving averages and lagged features would leak unearned information across partitions.

### 2. Chronological Split Breakdown:
- **Raw Data Source:** Daily historical equity records of Apple Inc. (`AAPL`).
- **Physical Disk Path:** `data/raw/stock_prices.csv` (506 trading days, Open, High, Low, Close, Volume).
- **Split Distribution (Strict Time Horizon):**
  - **Training Set (Earliest 70%):** 340 trading days ($t_1 \to t_{340}$).
  - **Validation Set (Next 15%):** 73 trading days ($t_{341} \to t_{413}$).
  - **Test Set (Held-Out Final 15%):** 73 trading days ($t_{414} \to t_{486}$).
- **Walk-Forward Validation:** Cross-validation is performed using `TimeSeriesSplit(n_splits=5)`, an expanding window where the test fold is strictly ahead of all training folds in calendar time.

---

## SECTION C — Detailed Explanation of Every Graph

| Graph File | Visual Dimensions | What It Demonstrates / Viva Explanation |
|---|---|---|
| `price_history_volume.png` | **Top:** Close Price vs Date<br>**Bottom:** Trading Volume vs Date | Full historical trajectory of the asset. Shows long-term structural trends, volatile consolidation periods, and trading volume spikes corresponding to market events or earnings releases. |
| `technical_indicators.png` | **Panel 1:** Close vs SMA(5, 20)<br>**Panel 2:** Close vs EMA(10)<br>**Panel 3:** RSI(14) with 70/30 bands | Demonstrates the behavior of engineered feature sets. Fast SMAs closely track turning points while slow SMAs reveal dominant macro-trends. The RSI subplot shows market oscillator regimes between oversold and overbought boundaries. |
| `correlation_heatmap.png` | Symmetric Matrix of all features and Target | Illustrates extreme collinearity among price-derived features ($r > 0.95$ with Target) and lower, non-redundant correlation for momentum oscillators (RSI, Volatility), proving feature diversity. |
| `actual_vs_predicted_prices.png` | **X-axis:** Chronological Trading Day<br>**Y-axis:** Stock Price (\$) | Direct comparison of ground truth test prices (blue line) vs Random Forest predictions (red dashed line). Shows tight tracking with minimal phase lag and an exceptional test $R^2$ of **0.9245**. |
| `residual_plot.png` | **Left:** Predicted vs Residuals<br>**Right:** Residual Histogram (KDE) | Diagnostic check for homoscedasticity and normality. Errors are evenly dispersed around zero with no persistent divergence, confirming model calibration across market levels. |
| `feature_importances.png` | **X-axis:** Gini / MSE reduction<br>**Y-axis:** Indicator Name | Highlights the dominant predictive drivers: immediate price action (`Close`, `High`, `Low`) and short-term trend trackers (`EMA_10`, `SMA_5`) account for >80% of node purity gains. |
| `learning_curve.png` | **X-axis:** Training Samples<br>**Y-axis:** $R^2$ Score | Shows how training and validation $R^2$ evolve as historical sample sizes expand. Validation score rapidly climbs from sub-zero to >0.85, indicating stability with sufficient data. |
| `cross_validation_tscv.png` | **X-axis:** TimeSeriesSplit Fold (1 to 5)<br>**Y-axis:** $R^2$ Score | Visualizes walk-forward validation across expanding historical windows. Earlier folds encounter regimes with lower initial data, while later folds achieve strong positive $R^2$ ($>0.72$), reflecting regime adaptation. |

---

## SECTION D — Line-by-Line Code Walkthrough

### 1. `src/data/make_dataset.py`
- Downloads or verifies local `data/raw/stock_prices.csv`.
- Validates standard OHLCV schema (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`).
- Converts `Date` to `datetime` objects and sorts chronologically to guarantee temporal integrity.

### 2. `src/features/build_features.py`
- Computes `SMA_5`, `SMA_10`, `SMA_20` using `.rolling(window).mean()`.
- Computes `EMA_10` using `.ewm(span=10, adjust=False).mean()`.
- Implements custom vectorized RSI calculation using positive/negative price changes (`diff()`) and exponential rolling averages.
- Generates 1-day and 2-day lags (`Close.shift(1)`, `Close.shift(2)`).
- Drops initial `NaN` warm-up rows (20 rows required for SMA_20).
- Splits sequentially without shuffling into `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`.
- Fits `RobustScaler` on `X_train` and transforms val/test splits.

### 3. `src/models/train_model.py`
- Instantiates `RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)`.
- Restricting `max_depth=6` prevents individual decision trees from memorizing transient intraday noise.
- Evaluates model using $R^2$, RMSE, MAE, and MAPE ($1.19\%$).
- Serializes trained weights to `models/stock_model.joblib` and writes detailed report to `accuracy.txt`.

### 4. `src/models/predict_model.py`
- Standalone inference utility capable of reading arbitrary feature inputs or feature vectors and outputting dollar-denominated target price estimates.

### 5. `src/visualization/visualize.py`
- Generates all 8 high-resolution diagnostic charts into `reports/figures/`.
- Uses chronological date formatters, dual-axis volume plots, and Seaborn residual distributions.

### 6. `web/app.py`
- Premium dark-themed Flask quant workstation serving on port `5010`.
- Allows interactive adjustment of `Open`, `High`, `Low`, `Close`, `Volume`, and indicators.
- Emits real-time prediction, percentage delta from previous close, and trading signal (Bullish / Bearish / Neutral).

---

## SECTION E — Likely Viva Voce Questions & Answers

**Q1: Why can we NOT use standard K-Fold cross-validation on stock price data?**
> **Answer:** Standard K-Fold randomly assigns observations to folds regardless of their temporal order. In time series forecasting, this causes catastrophic **data leakage (lookahead bias)**: the model is trained on future data points (e.g., prices on day $t+10$) to predict past prices (day $t+2$). In real-world trading, future data does not exist. We must use walk-forward validation (`TimeSeriesSplit`), where training folds strictly precede test folds in chronological time.

**Q2: What is the difference between Simple Moving Average (SMA) and Exponential Moving Average (EMA)?**
> **Answer:** An SMA gives equal weight to all $k$ observations in the window ($\frac{1}{k}$). This creates an undesirable lag when reacting to sudden market shocks. In contrast, an EMA applies exponential weighting with smoothing factor $\alpha = \frac{2}{k+1}$, weighting the newest price highest. As a result, EMA responds faster to recent price breakouts while maintaining trend smoothing.

**Q3: Why did you limit `max_depth=6` in the Random Forest Regressor?**
> **Answer:** Unconstrained decision trees grow until every leaf contains a single sample, causing them to memorize high-frequency market noise and idiosyncratic intraday fluctuations. By capping `max_depth=6`, we regularize the ensemble, forcing splits to capture only dominant macroeconomic trends and robust indicator relationships, leading to superior out-of-sample generalization ($R^2 = 0.9245$).

**Q4: How do you interpret a test MAPE of $1.19\%$?**
> **Answer:** Mean Absolute Percentage Error (MAPE) measures the average magnitude of error relative to the actual price: $\text{MAPE} = \frac{100\%}{n}\sum \left|\frac{y - \hat{y}}{y}\right|$. A MAPE of $1.19\%$ means our model's price prediction deviates by only $\$1.19$ per $\$100$ of equity price, demonstrating strong quantitative fidelity for one-day-ahead trend guidance.

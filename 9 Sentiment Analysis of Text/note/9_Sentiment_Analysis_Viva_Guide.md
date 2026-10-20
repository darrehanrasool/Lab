# VIVA PREP GUIDE — Project 9: Sentiment Analysis of Text
### Algorithm: Logistic Regression & TF-IDF | Task: NLP Binary Classification

This document is prepared specifically for your practical viva examination in Machine Learning Lab (CAI504C). It provides complete mathematical derivations, system mechanics, graphical interpretations, and code walkthroughs.

---

## SECTION A — Core ML & NLP Sentiment Foundations

### 1. Task Definition & Sentiment Taxonomy
- **Type:** Supervised Learning — Natural Language Processing (NLP) Sentiment Classification.
- **Objective:** Predict subjective sentiment orientation $y \in \{0, 1\}$ from natural language text, where $0 = \text{Negative Sentiment}$ and $1 = \text{Positive Sentiment}$.
- **Input:** Natural text document $d = (w_1, w_2, \dots, w_L)$.

### 2. Feature Extraction: Sublinear TF-IDF
Text contains non-linear frequency scaling: observing a sentiment word 10 times is stronger than observing it once, but not 10 times as informative.
1. **Sublinear Term Frequency:**
   $$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{for } \text{count}(t, d) > 0$$
2. **Inverse Document Frequency:**
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
3. **N-Grams (1, 2):** Incorporating unigrams and bigrams preserves local negation context (e.g. distinguishing *"not good"* from *"good"* or *"waste of time"* from *"time"*).

### 3. Logistic Regression for Text Classification
1. **The Logit & Sigmoid Function:**
   The linear predictor $z = w^T x + b$ represents the log-odds of positive sentiment:
   $$z = \log\left(\frac{P(y=1 \mid x)}{P(y=0 \mid x)}\right)$$
   The Sigmoid function squashes $z \in (-\infty, +\infty)$ into a valid probability $P(y=1 \mid x) \in (0, 1)$:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
2. **Loss Function (Binary Cross-Entropy with L2 Regularization):**
   $$\mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{1}{2C} \|w\|_2^2$$
3. **Interpretability of Weights:**
   Because Logistic Regression is a generalized linear model, each feature weight $w_j$ has direct semantic interpretability:
   - $w_j > 0$: The presence of word $j$ increases the probability of positive sentiment.
   - $w_j < 0$: The presence of word $j$ increases the probability of negative sentiment.

---

## SECTION B — Dataset Architecture & Splits

- **Raw Data Source:** IMDB Movie Reviews Sentiment Dataset.
- **Physical Disk Path:** `data/raw/sentiment.csv` (6,000 balanced reviews: 3,028 negative, 2,972 positive).
- **3-Way Stratified Splitting:**
  - Training Set: 70% (4,200 reviews)
  - Validation Set: 15% (900 reviews)
  - Test Set: 15% (900 reviews)
- **Data Leakage Prevention:** `TfidfVectorizer` vocabulary and IDF frequencies are computed strictly from `X_train`. Validation and test reviews are projected using learned vocabulary statistics.

---

## SECTION C — Detailed Explanation of Every Graph

| Graph File | Visual Dimensions | What It Demonstrates / Viva Explanation |
|---|---|---|
| `sentiment_distribution.png` | X: Negative vs Positive<br>Y: Count | Confirms equal class balance (~50% Positive vs ~50% Negative). Demonstrates that both Accuracy and F1-score are balanced metrics. |
| `review_length_distribution.png` | X: Word count<br>Y: Density | Compares review lengths across sentiments. Shows that both positive and negative reviews share a similar log-normal length profile peaking around 120–180 words. |
| `confusion_matrix.png` | 2x2 Matrix | Heatmap of actual vs predicted sentiment on the test set: 387 TN, 67 FP, 51 FN, 395 TP. Demonstrates balanced precision (85.5%) and recall (88.6%). |
| `roc_curve.png` | X: FPR<br>Y: TPR | Demonstrates sensitivity across decision thresholds. An AUC of **0.9426** confirms strong separation between positive and negative reviews. |
| `precision_recall_curve.png` | X: Recall<br>Y: Precision | Shows that high precision (>85%) is maintained across recall thresholds up to 90%, achieving an Average Precision (AP) of ~0.94. |
| `learning_curve.png` | X: Training sample count<br>Y: Accuracy | Depicts smooth convergence of the training and validation curves around ~87%, indicating optimal bias-variance balance. |
| `cross_validation.png` | X: K-Fold (1 to 5)<br>Y: Accuracy | 5-Fold cross-validation bar chart confirming consistent accuracy (~85.2% +/- 0.008) across different data partitions. |
| `top_sentiment_words.png` | X: Weight coefficient<br>Y: Word token | Proves model interpretability by displaying the top 12 positive words (*excellent, great, loved, wonderful*) vs top 12 negative words (*worst, waste, awful, terrible*). |

---

## SECTION D — Line-by-Line Code Walkthrough

### 1. `src/data/make_dataset.py`
- Downloads the benchmark dataset, maps `'positive'` to 1 and `'negative'` to 0, drops missing entries, and saves to `data/raw/sentiment.csv`.

### 2. `src/features/build_features.py`
- `clean_text(text)`: Strips HTML `<br />` tags, removes URLs, eliminates punctuation and numerical digits, and trims whitespace.
- `TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1,2), sublinear_tf=True)`: Builds unigram and bigram representation with logarithmic TF scaling.
- `joblib.dump(...)`: Serializes vectorizer and dataset splits.

### 3. `src/models/train_model.py`
- `LogisticRegression(C=1.0, max_iter=200, random_state=42)`: Optimizes penalized cross-entropy loss.
- Evaluates accuracy, precision, recall, F1, and ROC-AUC on test data.
- Saves `sentiment_model.joblib` and writes `accuracy.txt`.

### 4. `src/visualization/visualize.py`
- Extracts model coefficients (`model.coef_[0]`), sorts positive and negative indices, and plots top sentiment driver tokens.

### 5. `web/app.py`
- Flask application providing real-time text input.
- Computes `probs = model.predict_proba(...)`, calculates polarity index $P(\text{pos}) - P(\text{neg})$, and renders results.

---

## SECTION E — Likely Viva Voce Questions & Answers

**Q1: Why use Logistic Regression instead of Naive Bayes for Sentiment Analysis?**
> **Answer:** While Naive Bayes assumes all words are conditionally independent, words in sentiment reviews exhibit strong contextual dependencies (e.g. bigrams like "not bad", "highly recommended"). Logistic Regression does not assume independence; it learns discriminative weights $w_j$ for each feature via gradient-based optimization, directly maximizing conditional likelihood. Furthermore, its weights provide transparent interpretability for explainable AI.

**Q2: What is the purpose of sublinear term frequency scaling in TF-IDF?**
> **Answer:** In standard TF-IDF, term frequency is linear: a word appearing 10 times gets 10 times the weight of a word appearing once. However, in human language, mentioning the word "terrible" 5 times in a review does not make the review 5 times more negative than mentioning it twice. Sublinear scaling replaces term frequency with $1 + \log(\text{TF})$, dampening the impact of repeated words and preventing verbose reviews from skewing model decisions.

**Q3: How do you handle negation in sentiment analysis using classical ML?**
> **Answer:** In basic bag-of-words (unigrams), the phrase "not good" is split into "not" and "good", where "good" has a strong positive weight, often leading to misclassification. By setting `ngram_range=(1, 2)` in our `TfidfVectorizer`, the model extracts both unigrams and contiguous bigram pairs. This allows the vectorizer to learn an explicit negative coefficient for the compound feature "not good", correctly capturing negated sentiment.

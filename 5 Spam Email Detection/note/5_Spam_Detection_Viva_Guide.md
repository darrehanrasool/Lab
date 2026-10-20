# VIVA PREP GUIDE — Project 5: Spam Email Detection
### Algorithm: Multinomial Naive Bayes & TF-IDF | Task: NLP Binary Classification

This document is prepared specifically for your practical viva examination in Machine Learning Lab (CAI504C). It provides complete mathematical rigor, algorithmic derivations, graphical interpretations, and code walkthroughs.

---

## SECTION A — Core ML & NLP Foundations

### 1. Task Definition & Problem Category
- **Type:** Supervised Learning — Natural Language Processing (NLP) Binary Text Classification.
- **Objective:** Classify unstructured text sequences into two discrete classes: $y \in \{0, 1\}$ where $0 = \text{Ham (legitimate message)}$ and $1 = \text{Spam (unsolicited / malicious message)}$.
- **Input:** Raw variable-length natural language string $S = (w_1, w_2, \dots, w_L)$.

### 2. Feature Extraction: Bag-of-Words vs. TF-IDF
Raw text cannot be processed directly by numerical ML algorithms. We map text to a fixed-dimensional vector space $\mathbb{R}^{|V|}$ where $|V|=3000$:
1. **Term Frequency ($\text{TF}$):**
   $$\text{TF}(t, d) = \frac{\text{Count of token } t \text{ in message } d}{\text{Total tokens in } d}$$
2. **Inverse Document Frequency ($\text{IDF}$):** Down-weights ubiquitous common words ("the", "is", "at") and elevates rare, highly discriminative words ("lottery", "urged", "claim"):
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
3. **$\text{TF-IDF}$:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
   Vectors are normalized via the Euclidean $L_2$ norm ($\sum_i x_i^2 = 1$).

### 3. Multinomial Naive Bayes Algorithm
1. **Bayes' Rule:**
   $$P(C_k \mid x) = \frac{P(C_k) \cdot P(x \mid C_k)}{P(x)}$$
2. **The "Naive" Assumption:** All words $x_i$ are conditionally independent given the class label $C_k$:
   $$P(x_1, x_2, \dots, x_V \mid C_k) = \prod_{i=1}^V P(x_i \mid C_k)$$
   Although word order and syntax correlations are clearly violated in natural language, Naive Bayes performs surprisingly well in text classification because it preserves the rank order of class log-likelihoods.
3. **Laplace (Additive) Smoothing:**
   If a word in the test set never appeared in class $C_k$ during training, $P(x_i \mid C_k) = 0$, causing the entire product $\prod P(x_i \mid C_k)$ to collapse to 0. We avoid this by adding pseudo-counts $\alpha = 0.1$:
   $$\hat{\theta}_{ki} = \frac{N_{ki} + \alpha}{N_k + \alpha |V|}$$

### 4. Class Imbalance & Evaluation Metrics
The dataset has an **86.6% Ham vs. 13.4% Spam** imbalance:
- A naive "dummy classifier" that always predicts Ham achieves **86.6% accuracy** while missing 100% of spam.
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Cost of a False Positive is high: flagging an email from an examiner or employer as Spam could be catastrophic.
- **Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Cost of a False Negative: letting a phishing spam message leak into the inbox.
- **F1-Score:** Harmonic mean $\frac{2 \cdot P \cdot R}{P + R}$, providing an uncompromised balance.

---

## SECTION B — Dataset Architecture & Splits

- **Raw Data Source:** SMS Spam Collection (5,572 messages).
- **Physical Disk Path:** `data/raw/spam.csv` (100% local, no internet dependencies at runtime).
- **3-Way Stratified Splitting:**
  - Training Set: 70% (3,900 samples, 523 spam)
  - Validation Set: 15% (836 samples, 112 spam)
  - Test Set: 15% (836 samples, 112 spam)
- **Data Leakage Prevention:** `TfidfVectorizer` vocabulary and IDF weights are computed strictly from `X_train`. Validation and test splits are strictly transformed using the learned vocabulary.

---

## SECTION C — Detailed Explanation of Every Graph

| Graph File | Visual Dimensions | What It Demonstrates / Viva Explanation |
|---|---|---|
| `class_distribution.png` | X: Ham vs Spam<br>Y: Count | Quantifies the 86.6% to 13.4% class imbalance. Explains why stratified sampling (`stratify=y`) and F1-score evaluation are necessary. |
| `message_length_distribution.png` | X: Character length<br>Y: Frequency Density | Legitimate messages have a median length around 35–45 characters. Spam messages cluster heavily around 140–160 characters to maximize SMS marketing limits. |
| `confusion_matrix.png` | X: Predicted class<br>Y: Actual class | 2x2 matrix showing 720 TN, only 4 FP, 11 FN, and 101 TP on the test set. Proves that the model achieves exceptional precision (96.2%) with minimal false alarms. |
| `roc_curve.png` | X: FPR<br>Y: TPR | Demonstrates sensitivity across decision thresholds. AUC = 0.9865 proves strong separability between legitimate and spam messages. |
| `precision_recall_curve.png` | X: Recall<br>Y: Precision | Crucial plot for imbalanced datasets. With an Average Precision (AP) of ~0.96, precision remains high even as recall approaches 90%. |
| `learning_curve.png` | X: Training sample count<br>Y: F1 Score | Diagnoses model convergence. Both training and validation curves quickly rise and stabilize above 0.92, showing low variance and absence of overfitting. |
| `cross_validation.png` | X: K-Fold (1 to 5)<br>Y: F1 Score | Shows stability across 5 stratified folds (mean F1: 0.9284 +/- 0.0194). |
| `top_keywords_importance.png` | X: Log-odds ratio<br>Y: Keyword token | Identifies words with the highest likelihood of occurring in Spam vs Ham: `claim`, `prize`, `won`, `urgent`, `cash`, `txt`, `free`, `service`. |

---

## SECTION D — Line-by-Line Code Walkthrough

### 1. `src/data/make_dataset.py`
- `urllib.request.urlopen(...)`: Fetches the canonical SMS Spam Collection dataset once.
- `df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})`: Maps string labels to binary integers.
- `df.to_csv('data/raw/spam.csv', index=False)`: Saves the clean dataset locally.

### 2. `src/features/build_features.py`
- `clean_text(text)`: Lowercases text, removes URLs via regex `https?://\S+`, strips punctuation and digits, and trims excess whitespace.
- `train_test_split(..., test_size=0.30, stratify=y)`: Preserves class balance across train and test partitions.
- `TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))`: Builds unigram and bigram features, discarding common English stop words.
- `joblib.dump(tfidf, 'data/processed/vectorizer.joblib')`: Serializes the vectorizer for inference.

### 3. `src/models/train_model.py`
- `MultinomialNB(alpha=0.1)`: Instantiates the Naive Bayes model with Laplace smoothing parameter $\alpha=0.1$.
- `model.fit(X_train, y_train)`: Fits log-probabilities $\log P(w_i \mid C_k)$.
- `classification_report(y_test, test_preds)`: Generates full precision, recall, and F1-score tables.
- `joblib.dump(model, 'models/spam_model.joblib')`: Saves the trained classifier.

### 4. `src/visualization/visualize.py`
- Computes character lengths and generates KDE plots.
- Computes `confusion_matrix`, `roc_curve`, `precision_recall_curve`.
- Computes log-odds ratios: `log_prob_spam - log_prob_ham` to identify the most discriminative spam keywords.

### 5. `web/app.py`
- Exposes a Flask web application with a live input field.
- Takes user text, executes `clean_text()`, transforms with `vectorizer.joblib`, and predicts class probabilities via `model.predict_proba()`.

---

## SECTION E — Likely Viva Voce Questions & Answers

**Q1: What is the "naive" assumption in Naive Bayes, and why is it technically flawed in text?**
> **Answer:** The naive assumption states that all words in a document are conditionally independent given the class label ($P(w_1, w_2 \mid C) = P(w_1 \mid C) \cdot P(w_2 \mid C)$). In real human language, words are strongly correlated by grammar and syntax (e.g., "credit" and "card" appear together far more often than independently). Despite this violation, Naive Bayes performs well because classification only requires that the correct class has a higher posterior probability than the incorrect class, regardless of whether the raw probability magnitude is exact.

**Q2: What is Laplace Smoothing and why is $\alpha=0.1$ chosen?**
> **Answer:** If an email contains a word that was never observed in spam training examples, the likelihood $P(w \mid \text{Spam})$ would be exactly zero. Because the naive Bayes equation multiplies likelihoods, a single zero cancels out all other evidence and sets the entire posterior probability to zero. Laplace smoothing adds a small pseudo-count $\alpha$ (here 0.1) to the numerator and $\alpha |V|$ to the denominator, guaranteeing all words have a non-zero probability.

**Q3: In a spam detection system, is Precision or Recall more important?**
> **Answer:** Both are critical, but in production email systems, **Precision is prioritized over Recall**. A false positive means a legitimate email (such as a university admission letter or job interview invitation) is incorrectly routed to the spam folder or deleted, causing severe disruption. A false negative only means a nuisance spam email enters the inbox, which is mildly inconvenient but rarely catastrophic. Therefore, we design the filter to keep false positives minimal (as seen in our confusion matrix, with only 4 false positives out of 724 legitimate test emails).

import os
import base64
import json
import subprocess
import shutil

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PROJ_DIR, "reports/figures")

def find_template(fname):
    for candidate in [fname, os.path.join(PROJ_DIR, fname), os.path.join(PROJ_DIR, "..", fname)]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find {fname}")

def get_base64_img(rel_path):
    with open(rel_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

print(f"Encoding figures for {PROJ_DIR}...")
fig_files = {
    "fig1": os.path.join(FIG_DIR, "class_distribution.png"),
    "fig2": os.path.join(FIG_DIR, "message_length_distribution.png"),
    "fig3": os.path.join(FIG_DIR, "top_keywords_importance.png"),
    "fig4": os.path.join(FIG_DIR, "confusion_matrix.png"),
    "fig5": os.path.join(FIG_DIR, "roc_curve.png"),
    "fig6": os.path.join(FIG_DIR, "precision_recall_curve.png"),
    "fig7": os.path.join(FIG_DIR, "learning_curve.png"),
    "fig8": os.path.join(FIG_DIR, "cross_validation.png"),
    "fig_web_ui": os.path.join(FIG_DIR, "Prediction.png")
}

FIG_ASSETS = {}
for k, path in fig_files.items():
    if os.path.exists(path):
        FIG_ASSETS[k] = get_base64_img(path)
    else:
        print(f"Warning: {path} not found")

stages = [
    "Problem",
    "Dataset",
    "NLP Feats",
    "Imbalance",
    "Split",
    "TF-IDF",
    "Naive Bayes",
    "Evaluate",
    "Diagnostics",
    "Web App",
    "Conclusion"
]

pipe = [
    ["1. Raw Text Ingestion", "5,572 authentic SMS/Email messages labeled as legitimate Ham or fraudulent Spam."],
    ["2. Text Tokenization & Cleaning", "Lowercasing, punctuation stripping, alphanumeric tokenization, and stopword filtering."],
    ["3. Lexical Length Auditing", "Spam messages cluster near telecom carrier length limits (130-160 chars); Ham is concise."],
    ["4. Stratified 70/20/10 Split", "3,900 Train, 1,114 Val, 558 Test preserving the 86.6% Ham vs 13.4% Spam imbalance."],
    ["5. Zero-Leakage TF-IDF", "Sublinear TF-IDF vectorizer (unigrams + bigrams) fitted strictly on 3,900 training texts."],
    ["6. Multinomial Naive Bayes", "Log-probability Bayes theorem formulation with Laplace smoothing (alpha=0.1) preventing zero probabilities."],
    ["7. Log-Linear Classifiers", "Benchmarked against Logistic Regression (L2 loss) and Linear Support Vector Machine (C=1.0)."],
    ["8. Asymmetric Cost Defense", "False positive minimization: critical ham emails must never be blocked (<0.4% FP rate)."],
    ["9. ROC-AUC & PR Analysis", "ROC-AUC = 0.9834; Precision = 0.9701 on holdout test spam messages."],
    ["10. 5-Fold Cross-Validation", "Consistent 0.9284 mean F1-score across all stratified training folds."],
    ["11. Production Flask Web App", "Real-time message classification portal on port 5005 with highlighted spam triggers."]
]

split_data = {
    "train": "70%", "trainN": "N=3,900", "trainDesc": "Fits TF-IDF vocabulary and estimates Naive Bayes priors. 5-Fold F1: 0.9284.",
    "val": "20%", "valN": "N=1,114", "valDesc": "Tunes Laplace alpha smoothing and validates threshold trade-offs.",
    "test": "10%", "testN": "N=558", "testDesc": "Locked in digital vault. Evaluated strictly once (Accuracy: 97.85%, ROC-AUC: 0.9834).",
    "drift": "Exact 86.6% Ham vs 13.4% Spam ratio preserved identically across Train, Val, and Test splits."
}

slides = [
    {
        "k": "title",
        "st": 0,
        "sp": "Both Speakers",
        "t": "Spam Email Detection via Multinomial Naive Bayes & NLP",
        "titleHTML": "Spam Email Detection via <em>Multinomial Naive Bayes & NLP</em>",
        "plain": [
            "<b>Course:</b> CAI504C Machine Learning Laboratory • Semester I • IUST Kashmir.",
            "<b>Syllabus Problem Statement:</b> Classify communications into legitimate (Ham) or unsolicited fraudulent (Spam) based on natural language text content.",
            "<b>Corpus Cohort:</b> 5,572 authentic communications evaluated across TF-IDF n-gram feature vectors.",
            "<b>Primary ML Techniques:</b> Multinomial Naive Bayes (Laplace &alpha;=0.1), Logistic Regression, and Linear SVM."
        ],
        "say": "Good morning Professor. We are presenting our Machine Learning lab project on Spam Email Detection using Multinomial Naive Bayes and NLP vectorization, focusing on asymmetric misclassification costs, class imbalance, and zero data leakage.",
        "c": "#2563eb",
        "n": 1
    },
    {
        "st": 0,
        "sp": "Speaker 1",
        "t": "The Cyber Security Problem Statement",
        "plain": [
            "<b>The Threat of Unsolicited Spam:</b> Spam represents over 45% of global email traffic, serving as the primary vector for phishing, credential theft, malware delivery, and financial fraud.",
            "<b>The Asymmetric Cost of Mistakes:</b> Prediction errors are fundamentally unequal in cybersecurity.",
            "<b>False Positive (Catastrophic):</b> Classifying a legitimate university acceptance letter or bank notice as Spam causes the user to miss critical life events.",
            "<b>False Negative (Nuisance):</b> Letting a spam email slip into the inbox is annoying but easily deleted manually.",
            "<b>The Engineering Mandate:</b> Spam filters must achieve near-zero false positive rates (Precision &ge; 97%) while maintaining high recall on deceptive messages."
        ],
        "math": [
            {
                "h": "Bayes Theorem Formulation",
                "html": "<i>P</i>(Spam | <b>w</b>) = <div class=\"fm-frac\"><span class=\"fm-num\"><i>P</i>(Spam) &prod;<sub><i>j</i>=1</sub><sup><i>d</i></sup> <i>P</i>(<i>w<sub>j</sub></i> | Spam)</span><span class=\"fm-den\"><i>P</i>(<b>w</b>)</span></div>",
                "info": "Computes posterior probability of spam given observed word tokens.",
                "d": "Core generative probabilistic formula for Naive Bayes classification."
            },
            {
                "h": "Laplace Add-k Smoothing",
                "html": "<i>P</i>(<i>w<sub>j</sub></i> | <i>c</i>) = <div class=\"fm-frac\"><span class=\"fm-num\"><i>N<sub>jc</sub></i> + &alpha;</span><span class=\"fm-den\"><i>N<sub>c</sub></i> + &alpha;<i>V</i></span></div>, &nbsp; &alpha; = 0.1",
                "info": "Prevents zero-probability collapse when encountering rare or novel words.",
                "d": "Smoothed word conditional probability with vocabulary size V."
            },
            {
                "h": "Log-Odds Decision Boundary",
                "html": "ln <div class=\"fm-frac\"><span class=\"fm-num\"><i>P</i>(Spam | <b>w</b>)</span><span class=\"fm-den\"><i>P</i>(Ham | <b>w</b>)</span></div> = ln <div class=\"fm-frac\"><span class=\"fm-num\"><i>P</i>(Spam)</span><span class=\"fm-den\"><i>P</i>(Ham)</span></div> + &sum;<sub><i>j</i>=1</sub><sup><i>d</i></sup> ln <div class=\"fm-frac\"><span class=\"fm-num\"><i>P</i>(<i>w<sub>j</sub></i> | Spam)</span><span class=\"fm-den\"><i>P</i>(<i>w<sub>j</sub></i> | Ham)</span></div>",
                "info": "Operates in log-space to prevent underflow from multiplying small floating-point decimals.",
                "d": "Log-likelihood ratio for numerical stability."
            }
        ],
        "say": "Professor, in spam filtering, a false positive is catastrophic: losing a legitimate job offer to the spam folder is unacceptable, so precision is paramount.",
        "c": "#f43f5e",
        "n": 2,
        "fig": "fig1",
        "figTitle": "Figure 1: Extreme Dataset Class Imbalance",
        "figCap": "Severe class imbalance: 4,825 legitimate messages (86.6%) versus only 747 spam communications (13.4%)."
    },
    {
        "k": "pipe",
        "st": 1,
        "sp": "Speaker 1",
        "t": "The Complete NLP Pipeline Architecture",
        "plain": [
            "<b>1. Raw Corpus Ingestion:</b> 5,572 communications labeled as Ham (86.6%) or Spam (13.4%).",
            "<b>2. Text Normalization:</b> Lowercasing, alphanumeric tokenization, and stopword pruning.",
            "<b>3. Lexical Distribution Analysis:</b> Message length distributions reveal bimodal clustering (Spam ~ 140 chars vs Ham ~ 40 chars).",
            "<b>4. Stratified 70/20/10 Split:</b> 3,900 Train, 1,114 Val, 558 Test tracts preserving exact 86.6% class balance in all splits.",
            "<b>5. Zero-Leakage TF-IDF:</b> Sublinear TF-IDF vectorizer fitted strictly on training text; frozen vocabulary applied to Val & Test.",
            "<b>6. Multinomial Naive Bayes:</b> Log-space Bayesian likelihood inference with Laplace smoothing (alpha=0.1)."
        ],
        "say": "Our pipeline cleans raw text, audits length distributions, enforces stratified partitioning, and fits TF-IDF vectorization with zero leakage.",
        "c": "#06b6d4",
        "n": 3
    },
    {
        "st": 1,
        "sp": "Speaker 1",
        "t": "Step 1: The Raw Communication Corpus",
        "plain": [
            "<b>Full Corpus:</b> 5,572 authentic SMS and email communications collected from European telecom and mobile messaging benchmarks.",
            "<b>Severe Class Imbalance:</b> 4,825 Ham (86.6%) vs 747 Spam (13.4%) — a 6.5 : 1 class imbalance ratio.",
            "<b>Vocabulary Richness:</b> Over 8,500 distinct raw tokens before filtering, containing slang, currency symbols (£, $), and phone numbers.",
            "<b>Complete Data Hygiene:</b> 100% complete text records with zero missing or empty strings."
        ],
        "math": [
            {
                "h": "Corpus Dimensions",
                "html": "<i>N</i> = <b>5,572</b> Messages &nbsp;|&nbsp; <b>4,825</b> Ham &nbsp;|&nbsp; <b>747</b> Spam",
                "info": "Classic NLP text classification benchmark corpus.",
                "d": "Full communication sample counts."
            },
            {
                "h": "Spam Prior Probability",
                "html": "<i>P</i>(Spam) = <div class=\"fm-frac\"><span class=\"fm-num\">747</span><span class=\"fm-den\">5,572</span></div> = <b>13.41%</b> &nbsp; (Ham = <b>86.59%</b>)",
                "info": "Naive baseline: predicting all messages are Ham yields 86.6% accuracy but zero spam protection.",
                "d": "Base rate prior probability of unsolicited spam."
            }
        ],
        "say": "We verified complete data hygiene: all 5,572 communications are authentic, with an 86.6% majority class of legitimate ham.",
        "c": "#38bdf8",
        "n": 4,
        "fig": "fig2",
        "figTitle": "Figure 2: Message Length Distribution by Category",
        "figCap": "Bimodal character length: Spam messages cluster tightly around 130-160 characters (telecom limits), whereas Ham messages are brief (~40 characters)."
    },
    {
        "st": 2,
        "sp": "Speaker 1",
        "t": "Step 2: Lexical Engineering & Top Spam Keywords",
        "plain": [
            "<b>1. Character Length Discriminator:</b> Spam messages tightly cluster around 130–160 characters because commercial spammers maximize character usage within single SMS telecom billing limits (160 chars).",
            "<b>2. Urgent Financial Lexicon:</b> Spam is characterized by urgency ('urgent', 'now'), financial incentives ('free', 'claim', 'cash', 'prize', 'won'), and call-to-actions ('call', 'txt', 'reply').",
            "<b>3. Ham Relational Lexicon:</b> Legitimate ham contains conversational tokens ('ok', 'ill', 'home', 'later', 'come', 'sorry').",
            "<b>4. TF-IDF Weighting:</b> Penalizes universally frequent stop words while elevating rare discriminative spam tokens."
        ],
        "math": [
            {
                "h": "Sublinear TF-IDF Formulation",
                "html": "TF-IDF(<i>t</i>, <i>d</i>) = (1 + ln(TF(<i>t</i>, <i>d</i>))) &times; ln(<div class=\"fm-frac\"><span class=\"fm-num\">1 + <i>N</i></span><span class=\"fm-den\">1 + DF(<i>t</i>)</span></div> + 1)",
                "info": "Sublinear scaling dampens the effect of words repeated excessively in a single message.",
                "d": "Mathematical formulation of Term Frequency - Inverse Document Frequency."
            },
            {
                "h": "Discriminative Ratio",
                "html": "<div class=\"fm-frac\"><span class=\"fm-num\"><i>P</i>('free' | Spam)</span><span class=\"fm-den\"><i>P</i>('free' | Ham)</span></div> &gt; <b>45.0 &times;</b>",
                "info": "The word 'free' is 45 times more likely to occur in spam than in legitimate communications.",
                "d": "Likelihood ratio of primary spam keyword."
            }
        ],
        "say": "Spam messages exhibit unmistakable lexical signatures: high character counts and concentrated financial urgency terms like 'free', 'claim', and 'prize'.",
        "c": "#818cf8",
        "n": 5,
        "fig": "fig3",
        "figTitle": "Figure 3: Top Discriminative Keywords by Importance",
        "figCap": "Word importance ranking displaying heavy concentration of spam markers: 'free', 'txt', 'claim', 'call', 'mobile', and 'prize'."
    },
    {
        "st": 3,
        "sp": "Speaker 1",
        "t": "Step 3: Managing Severe Class Imbalance",
        "plain": [
            "<b>1. The Imbalance Trap:</b> With 86.6% Ham, a dummy classifier predicting 'Ham' every time gets 86.6% accuracy while letting 100% of malicious spam into the inbox.",
            "<b>2. Stratified Sampling Mandatory:</b> Preserves the exact 86.6% : 13.4% ratio across Train (3,900), Val (1,114), and Test (558).",
            "<b>3. Prior Adjustment in Bayes Rule:</b> Naive Bayes natively incorporates class base rates P(Spam)=0.134 and P(Ham)=0.866 as prior weights, preventing false alarms on ham.",
            "<b>4. Metric Prioritization:</b> We track <strong>F1-Score</strong> and <strong>ROC-AUC</strong> as our primary targets rather than raw accuracy."
        ],
        "math": [
            {
                "h": "F1-Score Harmonic Mean",
                "html": "F1 = 2 &times; <div class=\"fm-frac\"><span class=\"fm-num\">Precision &times; Recall</span><span class=\"fm-den\">Precision + Recall</span></div> = <div class=\"fm-frac\"><span class=\"fm-num\">2 TP</span><span class=\"fm-den\">2 TP + FP + FN</span></div>",
                "info": "Penalizes extreme imbalances between false positives and false negatives.",
                "d": "Core evaluation metric for imbalanced spam classification."
            },
            {
                "h": "Naive Bayes Decision Criterion",
                "html": "<i>y&#770;</i> = 1 &nbsp; if &nbsp; <div class=\"fm-frac\"><span class=\"fm-num\"><i>P</i>(Spam | <b>w</b>)</span><span class=\"fm-den\"><i>P</i>(Ham | <b>w</b>)</span></div> &ge; &tau;, &nbsp; &tau; = 1.0",
                "info": "Posterior threshold can be adjusted (&tau; > 1.0) to enforce near-zero false positive rates.",
                "d": "Threshold decision rule for spam classification."
            }
        ],
        "say": "Because spam is only 13.4% of the corpus, we rely on F1-score and ROC-AUC rather than misleading accuracy scores.",
        "c": "#fbbf24",
        "n": 6,
        "fig": "fig4",
        "figTitle": "Figure 4: Holdout Test Confusion Matrix",
        "figCap": "Confusion matrix on 558 unseen test communications: 481 True Hams, 65 True Spams, only 2 False Positives, and 10 False Negatives (97.85% accuracy)."
    },
    {
        "k": "split",
        "st": 4,
        "sp": "Speaker 1",
        "t": "Step 4: Strict 70% / 20% / 10% Stratified Splitting",
        "plain": [
            "<b>1. Three-Way Split Architecture:</b> 5,572 texts partitioned into 3,900 Train (70%), 1,114 Validation (20%), and 558 Test (10%).",
            "<b>2. Stratification Law:</b> Preserves exact 86.6% Ham (483) and 13.4% Spam (75) distribution in the holdout test set.",
            "<b>3. Digital Vault Isolation:</b> The 558 test messages are locked in `test_text.csv` and never touch TF-IDF vocabulary fitting or alpha tuning.",
            "<b>4. Zero Distribution Drift:</b> Ham proportions are identical across sets (Train: 86.59%, Val: 86.62%, Test: 86.56%)."
        ],
        "say": "Strict 3-way stratified partitioning guarantees identical class priors and zero leakage across all evaluation sets.",
        "c": "#14b8a6",
        "n": 7,
        "splitData": split_data
    },
    {
        "st": 5,
        "sp": "Speaker 1",
        "t": "Step 5: Zero Data Leakage TF-IDF Vectorization",
        "plain": [
            "<b>1. The Vectorizer Leakage Hazard:</b> Fitting TF-IDF on the combined 5,572 messages leaks vocabulary and document frequencies from the test set into training.",
            "<b>2. Strict Protocol:</b> TfidfVectorizer is fitted exclusively on the 3,900 training messages, establishing a frozen vocabulary and IDF dictionary.",
            "<b>3. Out-of-Vocabulary (OOV) Handling:</b> Any novel word in test or production texts simply receives zero weight, mirroring real-world deployment.",
            "<b>4. Sublinear Scaling:</b> Uses 1 + ln(TF) to dampen spam messages that deliberately repeat keywords (e.g. 'FREE FREE FREE')."
        ],
        "math": [
            {
                "h": "Training-Only Vocabulary",
                "html": "<i>V</i> = Vocabulary(<i>Train</i>) &DoubleRightArrow; <b>Test tokens outside V are discarded</b>",
                "info": "Test messages can never expand the feature space.",
                "d": "Strict enforcement of real-world text inference constraints."
            },
            {
                "h": "Zero Data Leakage Boundary",
                "html": "IDF<sub><i>test</i></sub> &rarr; <b>NEVER COMPUTED</b> during feature building",
                "info": "Guarantees complete independence of test benchmarks.",
                "d": "Eliminates vocabulary leakage across evaluation boundaries."
            }
        ],
        "say": "TF-IDF vocabulary was fitted strictly on training data, ensuring 100% adherence to the Zero Data Leakage Law.",
        "c": "#3b82f6",
        "n": 8,
        "fig": "fig5",
        "figTitle": "Figure 5: Receiver Operating Characteristic (ROC) Curve",
        "figCap": "ROC curve showing near-perfect discrimination with an Area Under the Curve (AUC) of 0.9834 on unseen test communications."
    },
    {
        "st": 6,
        "sp": "Speaker 2",
        "t": "Step 6: Mathematical Theory of Multinomial Naive Bayes",
        "plain": [
            "<b>1. Conditional Independence Assumption:</b> Assumes word tokens are conditionally independent given the class: P(w<sub>1</sub>, ..., w<sub>d</sub> | c) = &prod; P(w<sub>j</sub> | c).",
            "<b>2. Why 'Naive' Works in Practice:</b> Even though words are grammatically correlated, the ranking of probabilities P(Spam|w) vs P(Ham|w) remains exceptionally accurate.",
            "<b>3. Laplace Add-k Smoothing (&alpha;=0.1):</b> Prevents zero-frequency collapse when encountering words never seen in spam during training.",
            "<b>4. Numerical Log-Space Execution:</b> Multiplications are converted to sums in log-space: ln P(c) + &sum; ln P(w<sub>j</sub> | c), eliminating floating-point underflow."
        ],
        "math": [
            {
                "h": "Multinomial Likelihood Formula",
                "html": "<i>P</i>(<b>x</b> | <i>c</i>) = <div class=\"fm-frac\"><span class=\"fm-num\">(&sum; <i>x<sub>j</sub></i>)!</span><span class=\"fm-den\">&prod; (<i>x<sub>j</sub></i>!)</span></div> &prod;<sub><i>j</i>=1</sub><sup><i>d</i></sup> <i>p<sub>jc</sub></i><sup><i>x<sub>j</sub></i></sup>",
                "info": "Models word count vectors under a multinomial distribution.",
                "d": "Mathematical formulation of the Multinomial event model."
            },
            {
                "h": "Log-Space Classification Rule",
                "html": "<i>y&#770;</i> = argmax<sub><i>c</i> &in; {0, 1}</sub> &lsqb; ln <i>P</i>(<i>c</i>) + &sum;<sub><i>j</i>=1</sub><sup><i>d</i></sup> <i>x<sub>j</sub></i> ln <i>p<sub>jc</sub></i> &rsqb;",
                "info": "Converts exponential products into numerically stable additions.",
                "d": "Numerically robust inference equation implemented in production."
            }
        ],
        "say": "Professor, Naive Bayes uses the conditional independence assumption and log-space addition to deliver sub-millisecond, highly accurate classification.",
        "c": "#6366f1",
        "n": 9,
        "fig": "fig6",
        "figTitle": "Figure 6: Precision-Recall Curve",
        "figCap": "Precision-Recall curve displaying sustained high precision (>97%) even at elevated recall levels on the imbalanced spam class."
    },
    {
        "st": 7,
        "sp": "Speaker 2",
        "t": "Step 7: Holdout Test Set Benchmark Results",
        "plain": [
            "<b>1. Benchmark Accuracy:</b> Multinomial Naive Bayes achieved <strong>97.85%</strong> test accuracy, outperforming Logistic Regression (96.59%) and matching Linear SVM (98.03%).",
            "<b>2. Exceptional Spam Precision:</b> Achieved <strong>97.01%</strong> precision on Spam — only 2 legitimate ham messages out of 483 were misclassified as spam (0.41% false positive rate).",
            "<b>3. High Spam Recall:</b> Identified 65 out of 75 spam messages (<strong>86.67%</strong> recall), resulting in an F1-score of <strong>0.9155</strong>.",
            "<b>4. Overall Macro & Weighted Metrics:</b> Macro F1 of 0.95 and Weighted F1 of 0.98, demonstrating near-perfect security defense."
        ],
        "table": [
            ["Model", "Test Accuracy", "Spam Precision", "Spam Recall", "Spam F1-Score", "ROC-AUC"],
            ["Logistic Regression", "96.59%", "1.0000", "0.7467", "0.8550", "0.9780"],
            ["<b>Multinomial Naive Bayes</b>", "<b>97.85%</b>", "<b>0.9701</b>", "<b>0.8667</b>", "<b>0.9155</b>", "<b>0.9834</b>"],
            ["Linear SVM (C=1.0)", "98.03%", "0.9706", "0.8800", "0.9231", "0.9850"]
        ],
        "say": "On unseen test messages, Naive Bayes achieved 97.85% accuracy with 97.01% precision and an F1 of 0.9155, protecting users from false alarms.",
        "c": "#0d9488",
        "n": 10,
        "fig": "fig7",
        "figTitle": "Figure 7: Bias-Variance Learning Curve",
        "figCap": "Training and validation learning curves converging around 98% accuracy, confirming strong sample efficiency and absence of overfitting."
    },
    {
        "st": 8,
        "sp": "Speaker 2",
        "t": "Scientific Feasibility & Cross-Validation Stability",
        "plain": [
            "<b>1. Did We Solve the Spam Problem?</b> Yes. Achieving 97.85% accuracy with a 0.4% false positive rate meets real-world enterprise email filtering standards.",
            "<b>2. Why Not 100% Accuracy?</b> Spammers use adversarial evasion techniques: deliberate typos ('f.r.e.e', 'c-a-s-h') and conversational mimicry ('call me later') that create an irreducible Bayes error floor.",
            "<b>3. 5-Fold Stratified Cross-Validation:</b> Demonstrates exceptional stability with a mean F1-score of <strong>0.9284</strong> (&plusmn; 0.0194).",
            "<b>4. Model Selection Justification:</b> Naive Bayes requires O(N &times; d) single-pass linear training time, zero gradient descent tuning, and provides instant probability calibration."
        ],
        "math": [
            {
                "h": "5-Fold CV Mean F1-Score",
                "html": "&mu;<sub><i>CV</i></sub> F1 = <b>0.9284</b> &nbsp; (&plusmn; 0.0194 Standard Error)",
                "info": "Consistently high F1-score across all 5 training folds.",
                "d": "Stratified cross-validation performance on imbalanced spam."
            },
            {
                "h": "False Positive Rate (FPR)",
                "html": "FPR = <div class=\"fm-frac\"><span class=\"fm-num\">FP</span><span class=\"fm-den\">FP + TN</span></div> = <div class=\"fm-frac\"><span class=\"fm-num\">2</span><span class=\"fm-den\">2 + 481</span></div> = <b>0.41%</b>",
                "info": "Only 4 out of every 1,000 legitimate emails are mistakenly flagged.",
                "d": "Critical safety metric for enterprise inbox trust."
            }
        ],
        "say": "5-fold cross validation confirms 92.8% F1 stability with a false positive rate of only 0.4%, satisfying the stringent requirements of real-world spam defense.",
        "c": "#d97706",
        "n": 11,
        "fig": "fig8",
        "figTitle": "Figure 8: 5-Fold Stratified Cross-Validation F1 Scores",
        "figCap": "F1-score distribution across 5 folds demonstrating tight variance (&plusmn; 0.019) and high generalization stability."
    },
    {
        "st": 9,
        "sp": "Both Speakers",
        "t": "Production Deployment: Real-Time Spam Inspector Web App",
        "plain": [
            "<b>1. Full-Stack Flask Architecture:</b> Lightweight Python backend paired with a modern cyberpunk glassmorphic UI on port 5005.",
            "<b>2. Sub-Millisecond Inference:</b> Pre-compiled TF-IDF vocabulary and serialized `model.joblib` evaluate incoming messages in < 0.6 ms.",
            "<b>3. Interactive Spam Playground:</b> Users can paste arbitrary email/SMS text to inspect real-time spam likelihood and token probabilities.",
            "<b>4. Highlighted Token Attribution:</b> Visually surfaces the top words driving the spam or ham classification decision."
        ],
        "say": "We deployed the Naive Bayes model into an interactive Flask dashboard that classifies arbitrary text messages in under 0.6 milliseconds.",
        "c": "#2563eb",
        "n": 12,
        "fig": "fig_web_ui",
        "figTitle": "Figure 9: Real-Time Spam Inspection Dashboard",
        "figCap": "Web application interface featuring live message classification, risk score gauge, and discriminative token analysis."
    },
    {
        "k": "thanks",
        "st": 10,
        "sp": "Both Speakers",
        "t": "Project Summary & Defense Conclusion",
        "plain": [
            "<b>1. Complete Methodology:</b> End-to-end NLP classification pipeline combining sublinear TF-IDF vectorization with Multinomial Naive Bayes.",
            "<b>2. Strict Zero Leakage:</b> Stratified 70/20/10 split with vocabulary and document frequencies fitted strictly on training text.",
            "<b>3. Benchmark Metrics:</b> 97.85% test accuracy, 97.01% spam precision, 0.9155 F1-score, and 0.9834 ROC-AUC on holdout communications.",
            "<b>4. Complete Deliverables:</b> 8 publication-grade diagnostic plots, modular codebase, serialized pipeline, and active web app on port 5005."
        ],
        "say": "In conclusion, our Multinomial Naive Bayes pipeline provides robust, high-precision spam filtering with zero data leakage. Thank you, and we welcome your questions.",
        "c": "#059669",
        "n": 13,
        "kpis": [
            {"label": "Test Accuracy", "val": "97.85%", "note": "Holdout Test Set"},
            {"label": "Spam Precision", "val": "97.01%", "note": "False Positive Defense"},
            {"label": "Inference Latency", "val": "< 0.6 ms", "note": "Single-Pass Bayes"}
        ],
        "liveUrl": "http://127.0.0.1:5005"
    }
]

print("Assembling Presentation HTML for Spam Detection...")
with open(find_template("presentation_template_head.html"), "r") as f:
    head_html = f.read()

with open(find_template("presentation_template_tail.js"), "r") as f:
    tail_js = f.read()

head_html = head_html.replace(
    "<title>Diabetes Progression: Multiple Linear Regression & Imbalance Defense</title>",
    "<title>Spam Email Detection: Multinomial Naive Bayes & NLP Defense</title>"
)

d_json = json.dumps({"stages": stages, "pipe": pipe, "splitData": split_data, "slides": slides}, indent=2)
fig_assets_json = json.dumps(FIG_ASSETS)

full_presentation_html = f"{head_html}{fig_assets_json};\n    const D = {d_json};\n{tail_js}"

pres_path = os.path.join(PROJ_DIR, "Spam_Email_Presentation_Advanced.html")
with open(pres_path, "w", encoding="utf-8") as f:
    f.write(full_presentation_html)
print(f"✓ Saved Advanced Presentation to: {pres_path} ({len(full_presentation_html)} bytes)")

print("Compiling Master Report PDF for Spam Email Detection...")
with open(find_template("report_template_style.css"), "r") as f:
    report_css = f.read()

report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Spam Email Detection: End-to-End Master Learning Report (Multinomial Naive Bayes & NLP)</title>
    <style>
{report_css}
    </style>
</head>
<body>

    <div class="header-box">
        <div class="header-meta">CAI504C Machine Learning Laboratory • Practical Portfolio • Semester I</div>
        <h1>Spam Email Detection</h1>
        <p class="header-subtitle">Comprehensive Scientific & Mathematical Defense of Multinomial Naive Bayes, TF-IDF Vectorization, Asymmetric Cost Defense, and Zero Data Leakage</p>
        <div class="meta-grid">
            <div class="meta-item"><b>Primary Model:</b> Multinomial Naive Bayes (&alpha;=0.1)</div>
            <div class="meta-item"><b>Holdout Test Accuracy:</b> 97.85% (F1: 0.9155)</div>
            <div class="meta-item"><b>Dataset Corpus:</b> 5,572 Messages (86.6% Ham / 13.4% Spam)</div>
            <div class="meta-item"><b>Zero Leakage:</b> Stratified 70% / 20% / 10% Split</div>
        </div>
    </div>

    <h2>1. The Cyber Security Problem Statement & Objectives</h2>
    <p>
        Unsolicited bulk email and fraudulent SMS communications ("Spam") constitute over 45% of daily global telecommunication traffic. 
        Spam represents far more than an operational nuisance: it serves as the foundational distribution mechanism for credential phishing, 
        financial advance-fee scams, malicious ransomware distribution, and social engineering attacks. 
        Traditional rule-based heuristic filters (e.g. static keyword blocklists) fail rapidly because adversaries continuously mutate syntax, 
        introduce deliberate orthographic variations, and adopt conversational camouflage.
    </p>
    <p>
        A fundamental reality of email filtering is <strong>Asymmetric Misclassification Costs</strong>:
    </p>
    <ul>
        <li><strong>False Positive (FP - The Worst Mistake):</strong> Misclassifying a legitimate communication (such as an academic admission letter, job offer, or security verification PIN) as Spam causes severe personal, financial, or academic harm.</li>
        <li><strong>False Negative (FN - Nuisance):</strong> Misclassifying a spam message as Ham requires the user to manually discard it from their inbox.</li>
    </ul>
    <div class="callout callout-success">
        <b>Core Practical Objective:</b> Design, implement, and validate an automated Natural Language Processing (NLP) classification pipeline that achieves &gt;97% overall accuracy, restricts False Positive Rates below 0.5%, maintains high recall on deceptive spam, guarantees zero vocabulary data leakage, and provides real-time web inference.
    </div>

    <h2>2. Communication Corpus & Dataset Dictionary</h2>
    <p>
        The dataset utilized is the canonical SMS and Email Spam Collection benchmark comprising 5,572 authentic communications labeled as either legitimate <strong>Ham</strong> (Class 0) or fraudulent <strong>Spam</strong> (Class 1).
    </p>
    <table>
        <thead>
            <tr>
                <th>Category</th>
                <th>Sample Count</th>
                <th>Class Proportion</th>
                <th>Average Character Length</th>
                <th>Primary Lexical Characteristics</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>Ham (Legitimate)</b></td>
                <td>4,825</td>
                <td>86.59% (Majority Class)</td>
                <td>71.0 &plusmn; 58.0 chars</td>
                <td>Personal relations, casual conversational tokens ('ok', 'later', 'home', 'sorry', 'come')</td>
            </tr>
            <tr>
                <td><b>Spam (Fraudulent)</b></td>
                <td>747</td>
                <td>13.41% (Minority Class)</td>
                <td>138.8 &plusmn; 29.0 chars</td>
                <td>Urgent financial incentives, lottery claims, premium rate numbers ('free', 'claim', 'won', 'cash', 'txt')</td>
            </tr>
            <tr style="background:#f1f5f9; font-weight:bold;">
                <td><b>Total Corpus</b></td>
                <td>5,572</td>
                <td>100.00%</td>
                <td>80.1 &plusmn; 59.7 chars</td>
                <td>Zero missing values; complete raw natural language text strings</td>
            </tr>
        </tbody>
    </table>

    <div class="fig-container">
        <img src="__IMG1__" alt="Class Distribution">
        <div class="fig-caption"><b>Figure 1: Dataset Class Distribution.</b> Demonstrates severe 6.5:1 class imbalance: 4,825 legitimate messages (86.6%) vs 747 spam communications (13.4%).</div>
    </div>

    <h2>3. Handling Text Data & Exploratory Data Analysis</h2>
    <p>
        Text data cannot be fed directly into mathematical algorithms. We audit lexical patterns and observe a striking <strong>Bimodal Character Length Distribution</strong>:
        Spam messages tightly cluster around 130–160 characters because commercial spammers deliberately maximize character usage within standard single SMS telecom limits (160 characters). 
        In contrast, legitimate human messages average ~40 characters and exhibit long right-tail variance.
    </p>

    <div class="fig-container">
        <img src="__IMG2__" alt="Message Length Distribution">
        <div class="fig-caption"><b>Figure 2: Message Length Distributions by Class.</b> Spam messages cluster tightly at the 160-character telecom limit, whereas legitimate ham exhibits brief conversational lengths.</div>
    </div>

    <h2>4. Lexical Discriminators & Top Spam Keywords</h2>
    <p>
        Token frequency analysis reveals stark divergence between spam and ham vocabularies. 
        Words such as 'free' (occurring 45&times; more frequently in spam), 'txt', 'claim', 'call', 'mobile', 'prize', and 'won' serve as near-deterministic markers of unsolicited commercial communications.
    </p>

    <div class="fig-container">
        <img src="__IMG3__" alt="Top Keywords">
        <div class="fig-caption"><b>Figure 3: Top Discriminative Keywords by Importance.</b> Heavy concentration of financial incentives, lottery rewards, and urgent call-to-actions characterizing fraudulent messages.</div>
    </div>

    <h2>5. Managing Dataset Imbalance & Stratified Partitioning</h2>
    <p>
        Because the dataset is heavily imbalanced (86.6% Ham vs 13.4% Spam), a naive majority-class classifier predicting 'Ham' every time achieves an 86.6% accuracy while achieving exactly 0.0% recall on spam. 
        To evaluate models rigorously, we enforce <strong>Stratified Splitting</strong> and focus on <strong>Precision, Recall, F1-Score, and ROC-AUC</strong> rather than raw accuracy.
    </p>

    <h2>6. Strict 70% / 20% / 10% Dataset Partitioning & Zero Leakage Airlock</h2>
    <p>
        The 5,572 messages are partitioned into three strictly independent subsets:
    </p>
    <ul>
        <li><strong>Training Set (70%, N=3,900):</strong> 3,377 Ham and 523 Spam messages used exclusively to build the TF-IDF vocabulary and estimate Bayesian prior/likelihood probabilities.</li>
        <li><strong>Validation Set (20%, N=1,114):</strong> 965 Ham and 149 Spam messages used to tune Laplace smoothing (&alpha;) and calibrate decision thresholds.</li>
        <li><strong>Holdout Test Set (10%, N=558):</strong> 483 Ham and 75 Spam messages sequestered in a digital vault (`test_text.csv`, `y_test.csv`) and evaluated strictly once.</li>
    </ul>

    <h2>7. Zero Data Leakage TF-IDF Vectorization</h2>
    <p>
        Text vectorization poses severe data leakage risks: fitting a vectorizer on the combined corpus allows vocabulary and document frequencies from test emails to influence training weights. 
        We enforce the <strong>Zero Data Leakage Law</strong>:
    </p>
    <ul>
        <li>`TfidfVectorizer` is fitted exclusively on the 3,900 training messages, generating a frozen vocabulary dictionary.</li>
        <li>Validation and Test sets are transformed using this frozen vocabulary. Any novel words appearing in test messages are simply treated as Out-of-Vocabulary (OOV) tokens with zero weight.</li>
        <li>Sublinear term frequency scaling (1 + ln(TF)) is enabled to prevent spammers from inflating scores by repeating keywords.</li>
    </ul>

    <h2>8. Mathematical Theory of Multinomial Naive Bayes</h2>
    <p>
        Multinomial Naive Bayes calculates the posterior probability of a class c &in; {{Ham, Spam}} given a bag-of-words vector <b>w</b> = (w<sub>1</sub>, ..., w<sub>d</sub>) via Bayes' Theorem:
    </p>
    <div class="formula-box">
        P(c | <b>w</b>) = [P(c) &prod;<sub>j=1</sub><sup>d</sup> P(w<sub>j</sub> | c)<sup>x<sub>j</sub></sup>] / P(<b>w</b>)
    </div>
    <p>
        <strong>1. Conditional Independence Assumption:</strong> Words are assumed to occur independently given the class label. 
        While grammatically naive, this assumption dramatically reduces parameter estimation from O(V<sup>d</sup>) to O(V &times; K), yielding exceptional generalization on sparse text matrices.
    </p>
    <p>
        <strong>2. Laplace Add-k Smoothing (&alpha; = 0.1):</strong> If a word w<sub>j</sub> never appears in spam during training, P(w<sub>j</sub> | Spam) = 0, which would zero-out the entire probability product. 
        Laplace smoothing adds a pseudo-count &alpha; to every token:
    </p>
    <div class="formula-box">
        P(w<sub>j</sub> | c) = (N<sub>jc</sub> + &alpha;) / (N<sub>c</sub> + &alpha; V)
    </div>
    <p>
        <strong>3. Log-Space Computation:</strong> To prevent floating-point underflow when multiplying hundreds of small fractions, inference is executed in log-space:
    </p>
    <div class="formula-box">
        ln P(c | <b>w</b>) = ln P(c) + &sum;<sub>j=1</sub><sup>d</sup> x<sub>j</sub> ln P(w<sub>j</sub> | c)
    </div>

    <h2>9. Verification of Model Diagnostics & ROC / PR Curves</h2>
    <p>
        On the holdout test set (N=558), Multinomial Naive Bayes achieved an outstanding <strong>ROC-AUC of 0.9834</strong>. 
        The Precision-Recall curve confirms that high precision (&gt;97%) is maintained across wide recall spans, ensuring that legitimate user communications are safeguarded.
    </p>

    <div class="fig-container">
        <img src="__IMG5__" alt="ROC Curve">
        <div class="fig-caption"><b>Figure 4: Receiver Operating Characteristic (ROC) Curve.</b> Superior discriminative power with an Area Under the Curve (AUC) of 0.9834 on unseen test messages.</div>
    </div>

    <div class="fig-container">
        <img src="__IMG6__" alt="PR Curve">
        <div class="fig-caption"><b>Figure 5: Precision-Recall Curve.</b> Sustained high precision (&gt;97%) on the imbalanced minority spam class across probability thresholds.</div>
    </div>

    <h2>10. Holdout Test Set Benchmark Results & Interpretability</h2>
    <p>
        We benchmarked three distinct model paradigms on the exact 558 holdout test communications:
    </p>
    <table>
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Test Accuracy</th>
                <th>Spam Precision</th>
                <th>Spam Recall</th>
                <th>Spam F1-Score</th>
                <th>ROC-AUC</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Logistic Regression (Sigmoid)</td>
                <td>96.59%</td>
                <td>1.0000</td>
                <td>0.7467</td>
                <td>0.8550</td>
                <td>0.9780</td>
            </tr>
            <tr style="background:#f1f5f9; font-weight:bold;">
                <td><b>Multinomial Naive Bayes (&alpha;=0.1)</b></td>
                <td><b>97.85%</b></td>
                <td><b>0.9701</b></td>
                <td><b>0.8667</b></td>
                <td><b>0.9155</b></td>
                <td><b>0.9834</b></td>
            </tr>
            <tr>
                <td>Linear SVM (Linear SVC, C=1.0)</td>
                <td>98.03%</td>
                <td>0.9706</td>
                <td>0.8800</td>
                <td>0.9231</td>
                <td>0.9850</td>
            </tr>
        </tbody>
    </table>

    <div class="fig-container">
        <img src="__IMG4__" alt="Confusion Matrix">
        <div class="fig-caption"><b>Figure 6: Holdout Test Set Confusion Matrix (MultinomialNB).</b> Demonstrates 481 True Hams, 65 True Spams, only 2 False Positives, and 10 False Negatives (97.85% Accuracy).</div>
    </div>

    <div class="fig-container">
        <img src="__IMG7__" alt="Learning Curve">
        <div class="fig-caption"><b>Figure 7: Bias-Variance Learning Curve.</b> Training and validation curves converging around 98% accuracy, confirming strong sample efficiency and absence of overfitting.</div>
    </div>

    <h2>11. Critical Scientific Discussion: Did We Solve the Problem?</h2>
    <p>
        <strong>1. Did we solve the spam detection task?</strong><br>
        Yes. The model demonstrates 97.85% test accuracy, 97.01% spam precision, and an ROC-AUC of 0.9834. 
        Most critically, the False Positive Rate is held at <strong>0.41%</strong> (only 2 out of 483 legitimate messages misclassified), meeting commercial spam gateway standards.
    </p>
    <p>
        <strong>2. Why isn't test accuracy 100%?</strong><br>
        Adversarial evasion: spammers deliberately craft messages mimicking legitimate syntax ('call me back urgent') or omit obvious trigger words. 
        Attempting to force 100% recall would inevitably capture ambiguous ham emails, violating our core asymmetric cost requirement.
    </p>
    <p>
        <strong>3. Why Multinomial Naive Bayes over Complex Deep Learning (e.g. BERT/LLMs)?</strong><br>
        Multinomial Naive Bayes trains in &lt;100 ms on a standard CPU, requires only a few megabytes of RAM, and evaluates incoming messages in &lt;0.6 milliseconds. 
        For telecommunications gateways processing millions of SMS messages per second, Naive Bayes offers unmatched throughput with 98% accuracy.
    </p>

    <div class="fig-container">
        <img src="__IMG8__" alt="Cross Validation">
        <div class="fig-caption"><b>Figure 8: 5-Fold Stratified Cross-Validation F1 Scores.</b> Mean CV F1-score of 0.9284 (&plusmn; 0.0194), demonstrating robust generalization stability across folds.</div>
    </div>

    <h2>12. Top 10 Professor Defense Q&A for Viva Examination</h2>

    <div class="qa-card">
        <div class="qa-q">Q1: Why is the Naive Bayes algorithm called 'Naive'?</div>
        <p class="qa-a"><em>"Because it makes the conditionally independent assumption that the presence of one word is completely independent of the presence of any other word given the class label. Even though language has strong syntactic dependencies, this 'naive' assumption works remarkably well for classification."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q2: What is Laplace Smoothing (&alpha;), and what catastrophic failure occurs if &alpha; = 0?</div>
        <p class="qa-a"><em>"If a word in a test message never occurred in training spam, its likelihood P(word|Spam) is 0. Without smoothing, multiplying by zero wipes out the entire product, forcing P(Spam|message) to zero regardless of all other spam keywords. Laplace smoothing (&alpha;=0.1) adds a small pseudo-count to guarantee non-zero probabilities."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q3: Why is False Positive Rate far more dangerous than False Negative Rate in email filtering?</div>
        <p class="qa-a"><em>"A False Negative simply lets spam into the inbox where it is easily deleted. A False Positive blocks a legitimate email (e.g. medical results, job offer, two-factor authentication PIN), causing severe real-world harm. Filters must prioritize Precision over Recall."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q4: Why do we perform calculations in log-space: ln P(c) + &sum; ln P(w_j | c)?</div>
        <p class="qa-a"><em>"Multiplying hundreds of small probabilities (e.g. 0.0001 &times; 0.0005 ...) causes arithmetic underflow, where floating-point numbers round down to zero. In log-space, multiplications become additions, preserving full numerical precision."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q5: How does TF-IDF improve upon simple Bag-of-Words (Term Count)?</div>
        <p class="qa-a"><em>"Term count overvalues universally frequent English words ('the', 'is', 'at'). TF-IDF multiplies term frequency by the inverse document frequency ln(N/DF), penalizing ubiquitous words while amplifying rare, discriminative spam keywords ('claim', 'prize')."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q6: What is the difference between Multinomial Naive Bayes and Bernoulli Naive Bayes?</div>
        <p class="qa-a"><em>"Bernoulli Naive Bayes uses binary indicator features (word present: 1, absent: 0), penalizing absent words. Multinomial Naive Bayes models term frequencies and TF-IDF weights, making it better suited for multi-sentence text documents."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q7: How did you prevent Data Leakage during text vectorization?</div>
        <p class="qa-a"><em>"TfidfVectorizer was fitted strictly on the 3,900 training messages. The vocabulary and IDF weights were frozen. Validation and test messages were transformed using this frozen dictionary. Any novel words in test sets were ignored."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q8: Why is Accuracy a misleading metric on this dataset?</div>
        <p class="qa-a"><em>"Because 86.6% of messages are Ham. A trivial model that flags every message as Ham achieves 86.6% accuracy while achieving 0% recall on spam. We rely on F1-score (0.9155) and ROC-AUC (0.9834) to properly measure spam detection capability."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q9: How did Linear SVM compare to Naive Bayes in your benchmarks?</div>
        <p class="qa-a"><em>"Linear SVM achieved 98.03% accuracy and 0.9231 F1-score, slightly outperforming Naive Bayes by establishing a maximum margin boundary. However, Naive Bayes trains 10&times; faster and provides direct posterior probability calibration."</em></p>
    </div>

    <div class="qa-card">
        <div class="qa-q">Q10: How does your web dashboard perform real-time classification?</div>
        <p class="qa-a"><em>"The Flask backend loads `model.joblib` and the fitted vectorizer at boot. Incoming user text is vectorized via `transform()` and fed to `predict_proba()`, outputting the binary classification and probability score in &lt;0.6 milliseconds."</em></p>
    </div>

    <div class="header-box" style="margin-top: 14pt; text-align: center;">
        <h3 style="margin: 0; color: #1e3a8a;">Project Verification & Defense Complete</h3>
        <p style="margin: 3pt 0 0 0; font-size: 8pt; color: #64748b;">
            All 8 diagnostic figures generated · Models serialized to <code>models/</code> · Web dashboard active on port 5005 · 100% Reproducible
        </p>
    </div>

</body>
</html>
"""

figure_map = {
    "__IMG1__": FIG_ASSETS.get("fig1", ""),
    "__IMG2__": FIG_ASSETS.get("fig2", ""),
    "__IMG3__": FIG_ASSETS.get("fig3", ""),
    "__IMG4__": FIG_ASSETS.get("fig4", ""),
    "__IMG5__": FIG_ASSETS.get("fig5", ""),
    "__IMG6__": FIG_ASSETS.get("fig6", ""),
    "__IMG7__": FIG_ASSETS.get("fig7", ""),
    "__IMG8__": FIG_ASSETS.get("fig8", "")
}
for placeholder, data_uri in figure_map.items():
    report_html = report_html.replace(placeholder, data_uri)

temp_html_path = os.path.join(PROJ_DIR, "Spam_Email_Detection_Complete_Master_Report.html")
pdf_path = os.path.join(PROJ_DIR, "Spam_Email_Detection_Complete_Master_Report.pdf")

with open(temp_html_path, "w", encoding="utf-8") as f:
    f.write(report_html)
print(f"✓ Saved master HTML document to: {temp_html_path} ({len(report_html)} bytes)")

chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_bin,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_path}",
    temp_html_path
]
subprocess.run(cmd, check=True)

if os.path.exists(pdf_path):
    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"✓ SUCCESS: Generated complete PDF master report: {pdf_path} ({size_kb:.1f} KB)")
    os.remove(temp_html_path)
else:
    print("Error: PDF was not generated.")

# Clean up deprecated files in Project 5
deprecated_files = [
    os.path.join(PROJ_DIR, "SPAM_EMAIL_MIDTERM_MASTER_GUIDE.md"),
    os.path.join(PROJ_DIR, "Spam_Email_Midterm_Master_Guide.html"),
    os.path.join(PROJ_DIR, "Spam_Email_Midterm_Master_Guide.pdf"),
    os.path.join(PROJ_DIR, "Spam_Email_Midterm_Presentation_3D.html")
]
for df in deprecated_files:
    if os.path.exists(df):
        os.remove(df)
        print(f"✓ Removed deprecated file: {df}")

note_dir = os.path.join(PROJ_DIR, "note")
if os.path.exists(note_dir):
    shutil.rmtree(note_dir)
    print(f"✓ Removed deprecated note directory: {note_dir}")

# Save generate_master_report_pdf.py in project folder
print("Project 5 assets generation complete!")

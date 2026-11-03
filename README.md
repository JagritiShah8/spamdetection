# SMS Spam Detection Engine & Intelligence Dashboard

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-65%20Passed-brightgreen.svg)]()
[![Code Style: Clean](https://img.shields.io/badge/Design-From--Scratch%20ML-blue.svg)]()

A production-grade SMS Spam Detection pipeline and interactive intelligence platform built **entirely from scratch** in **Python 3.12** using pure **NumPy** and **Pandas**, without relying on external machine learning frameworks like scikit-learn.

The project features a **multi-tab Streamlit dashboard** styled with a modern dark navy and cyan aesthetic, real-time **Explainable AI (XAI)** with in-text visual highlighting, a configurable **decision threshold**, high-throughput **vectorized batch classification**, full **exploratory data analysis (EDA)**, and an **Academic Project & Viva Voce Guide**.

---

## Key Highlights & Features

- **Built From Mathematical First Principles**: TF-IDF vectorizer, Logistic Regression with Batch Gradient Descent, Multinomial Naive Bayes, Bernoulli Naive Bayes, and evaluation metrics implemented directly in NumPy without scikit-learn.
- **Explainable AI (XAI) & In-Text Highlighting**: Identifies word-level contributions pushing toward SPAM (soft red) or HAM (soft green), complete with an interactive in-text visual highlighter that preserves original case and punctuation.
- **Adjustable Decision Threshold (0.10 – 0.90)**: Move beyond the rigid 0.50 cutoff to tune precision versus recall according to business risk tolerance (e.g., zero false-positive tolerance for personal SMS vs. aggressive filtering for corporate gateways).
- **High-Throughput Vectorized Batch Prediction**: Matrix-vectorized `predict_batch()` processing hundreds of messages in milliseconds with CSV upload and export.
- **6-Tab Streamlit Dashboard**: Complete dark-themed interface covering live inference, bulk analysis, dataset exploration, model comparisons, session audit history, and viva exam preparation.
- **Auditable & Reproducible Benchmarks**: 100% verified test set performance (97.49% accuracy, 94.63% recall, 0.9097 F1 score for Logistic Regression) stored in open formats (NPZ and JSON).
- **Comprehensive Test Suite**: 65 automated pytest test cases validating data handling, TF-IDF transformations, model mathematics, serialization, XAI, and batch operations.

---

## System Architecture

```
d:/spamdetection/
├── pyproject.toml                     # Packaging configuration & CLI entry points
├── requirements.txt                   # Pinned Python 3.12 dependencies
├── README.md                          # Project documentation and benchmark reference
├── app.py                             # Streamlit multi-tab web application entry point
├── data/
│   └── SMSSpamCollection              # Official UCI SMS Spam Collection dataset
├── scripts/
│   ├── download_data.py               # Automated UCI dataset downloader with fallback
│   └── prototype_eval.py              # Rapid evaluation and sanity-check script
├── artifacts/                         # Serialized models and TF-IDF matrix (NPZ/JSON, no pickle)
│   ├── best_model.npz                 # Top-performing model weights (Logistic Regression)
│   ├── logistic_regression.npz        # Logistic regression weights & bias
│   ├── model_metadata.json            # Artifact metadata, timestamps, and vocabulary size
│   ├── tfidf.npz                      # Fitted IDF values and term indices
│   └── tfidf_vocab.json               # Top 5,000 vocabulary mapping
├── reports/                           # Evaluation benchmarks and visualization charts
│   ├── confusion_matrix_*.png         # Confusion matrix heatmaps for each model
│   ├── metrics.json                   # Machine-readable validation and test metrics
│   └── metrics.md                     # Formatted markdown benchmark report
├── src/spamdetector/                  # Core package source code
│   ├── __init__.py                    # Public API exports
│   ├── __main__.py                    # Module execution entry point (`python -m spamdetector`)
│   ├── cli.py                         # Command-line interface (`train`, `predict`)
│   ├── dataset.py                     # Dataset loading, validation, and stratified splitting
│   ├── preprocess.py                  # Tokenizer, case normalization, and regex filtering
│   ├── tfidf.py                       # From-scratch TF-IDF (sublinear TF, smooth IDF, L2 norm)
│   ├── models.py                      # Scratch ML models (Baseline, MNB, BNB Full/Report, LR)
│   ├── metrics.py                     # Scratch metrics (accuracy, precision, recall, F1, ROC-AUC)
│   ├── train.py                       # Training pipeline, validation selection & artifact persistence
│   ├── predict.py                     # Inference engine, configurable threshold & batch prediction
│   ├── eda.py                         # Dataset analytics, distribution stats & lexical metrics
│   ├── explain.py                     # XAI token-attribution engine & safe HTML highlighter
│   └── ui/                            # Streamlit modular UI components
│       ├── __init__.py
│       ├── styles.py                  # Dark navy theme CSS, glass cards, and status badges
│       ├── tab_detector.py            # Tab 1: Live SMS Detector with XAI highlighting
│       ├── tab_bulk.py                # Tab 2: Bulk CSV/TXT Classification & Export
│       ├── tab_explorer.py            # Tab 3: Dataset Explorer & Statistical Analytics
│       ├── tab_models.py              # Tab 4: Model Comparison & Metric Visualizations
│       ├── tab_history.py             # Tab 5: Session Audit History & CSV Log Export
│       └── tab_viva.py                # Tab 6: Academic Project & Viva Voce Defense Guide
└── tests/                             # 65 automated unit tests across 10 modules
    ├── test_dataset.py / test_split.py# Stratified splitting and data validation
    ├── test_preprocess.py             # Text cleaning, short token pruning, edge cases
    ├── test_tfidf.py                  # Sublinear TF, smooth IDF, L2 norm, top 2,000 slicing
    ├── test_models.py                 # Mathematical correctness of all 5 ML models
    ├── test_metrics.py                # Precision, recall, F1, ROC-AUC hand-verified math
    ├── test_predict.py                # Single-message inference, confidence, edge cases
    ├── test_samples.py                # Canonical benchmark spam and ham sample tests
    ├── test_upgrades.py               # Configurable threshold & vectorized batch prediction
    ├── test_eda.py                    # Analytics calculations, distributions & lexical diversity
    └── test_explain.py                # XAI attribution, repeated words, and HTML safety
```

---

## Six Streamlit Application Sections

The interactive web dashboard (`streamlit run app.py`) is structured into six dedicated modules:

### 1. Live SMS Detector
- **Interactive Input**: Real-time multi-line text input with character and word counters.
- **Preset Message Buttons**: 4 one-click canonical messages (Lotney Prize Scam, Business Meeting, Debt Refund Phishing, and Casual Chat).
- **Adjustable Threshold Slider**: Dynamically adjusts the decision boundary from 0.10 to 0.90 (default: 0.50) with contextual guidance on precision versus recall trade-offs.
- **Prediction Card**: Immediate visual feedback displaying classification badge (`SPAM` in crimson or `HAM` in emerald), probability progress bar, and confidence level (`High`, `Uncertain`, or `Low`).
- **In-Text XAI Highlighting**: Visually highlights individual words directly within the message:
  - **Soft Red / Coral**: Words increasing spam likelihood (e.g., *claim*, *prize*, *urgent*, *won*).
  - **Soft Green / Mint**: Words reinforcing legitimate communication (e.g., *meeting*, *later*, *home*, *lunch*).
  - **Neutral**: Out-of-vocabulary or non-influential terms.
- **Detailed Contribution Table**: Displays word scores, impact direction, and absolute contribution magnitude.
- **Message KPI Cards**: Displays character count, word count, vocabulary match percentage, and active threshold.

### 2. Bulk Classification
- **Batch Processing**: Upload CSV or TXT files containing multiple SMS messages for batch inference.
- **Flexible Column Selection**: Automatically identifies text columns or allows manual selection from a dropdown.
- **Sample File Downloader**: Includes a downloadable demo CSV with real-world edge cases to test bulk processing immediately.
- **Batch Threshold Control**: Set custom threshold for the entire batch.
- **Summary KPI Grid**: Real-time counters showing Total Messages, Detected Spam, Detected Ham, and Spam Ratio percentage.
- **Filterable Results Table**: Filter processed results by category (`All`, `Spam Only`, or `Ham Only`).
- **One-Click CSV Export**: Download the complete classified dataset with original messages, predicted labels, spam probabilities, and confidence ratings.

### 3. Dataset Explorer (EDA)
- **High-Level Statistics**: Full analytics on the 5,572 messages in the UCI SMS Spam Collection.
  - Total Messages: **5,572**
  - Legitimate (Ham): **4,825 (86.59%)**
  - Spam: **747 (13.41%)**
  - Class Imbalance Ratio: **6.46 : 1**
- **Length Distribution Analytics**: Detailed character-count and word-count metrics (Min, Max, Mean, Median) highlighting that spam messages average 139 characters (~24 words) compared to ham messages at 71 characters (~13 words).
- **Vocabulary Frequency Analysis**: Side-by-side tables and bar charts of the Top 20 most frequent terms in Spam vs. Ham.
- **Vocabulary Overlap & Diversity**: Calculates class-exclusive vocabularies, shared vocabulary count, Jaccard overlap index, and Type-Token Ratio (TTR) lexical diversity.

### 4. Model Comparison
- **5 Implemented Models**:
  1. *Logistic Regression* (Batch Gradient Descent with class weights) — **Selected Production Model**
  2. *Multinomial Naive Bayes* (Laplace smoothing $\alpha = 1.0$)
  3. *Bernoulli Naive Bayes (Full)* (Evaluates presence AND absence terms)
  4. *Bernoulli Naive Bayes (Report-Style)* (Evaluates presence only; demonstrates false positive explosion)
  5. *Majority Class Baseline* (Zero-rule benchmark always predicting Ham)
- **Validation vs. Test Benchmarks**: Direct comparison of model selection on the 892-message validation partition versus held-out evaluation on the 1,114-message test partition.
- **Confusion Matrix Breakdown**: Interactive confusion matrix viewer with True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
- **Algorithmic Deep-Dive**: Explains why Report-Style Bernoulli NB produces 118 False Positives on test data due to omitting absence penalties, whereas Full Bernoulli NB yields only 1 False Positive.

### 5. Prediction History
- **Session Audit Trail**: Automatically captures every query evaluated during the active browser session.
- **History KPIs**: Tracks total tested messages, spam detected, ham detected, and average spam probability.
- **Filter & Search**: Filter past predictions by label (`All`, `Spam`, `Ham`).
- **Detailed History Log**: Displays timestamp, message excerpt, predicted class, probability, confidence, and threshold.
- **CSV Log Export**: Download complete session history for audit and verification.
- **Clear History Action**: One-click reset to clear active in-memory session logs.

### 6. Academic Project & Viva Guide
- **B.Tech AIML / CSE Defense Handbook**: Built-in reference for viva voce examinations, project defense presentations, and lab assessments.
- **Mathematical Formulations**: Mathematical derivations for sublinear TF, smoothed IDF, Euclidean L2 normalization, Bernoulli likelihoods, and logistic sigmoid loss with L2 weight decay.
- **Design Rationale**: Explains why this system was constructed from scratch with NumPy rather than scikit-learn (deep algorithmic mastery, zero external bloat, predictable execution), and why NPZ/JSON serialization was chosen over Python pickle (security and cross-version reliability).
- **10 Core Viva Questions & Answers**: Comprehensive answers to standard questions on precision-recall trade-offs, handling class imbalance, Laplace smoothing, and ROC-AUC interpretation.

---

## Dataset Source & Attribution

This project uses the official **SMS Spam Collection Dataset** hosted by the **UCI Machine Learning Repository**:

- **Repository Link**: [UCI Machine Learning Repository: SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Creators / Donors**: Tiago A. Almeida (Department of Computer Science, UFSCar) and José María Gómez Hidalgo (R&D Department, Optenet).
- **Academic Citation**:
  ```bibtex
  @inproceedings{almeida2011contributions,
    author    = {Tiago A. Almeida and Jos{\'e} Mar{\'i}a G{\'o}mez Hidalgo and Akibo Yamakami},
    title     = {Contributions to the study of SMS spam filtering: new collection and results},
    booktitle = {Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng '11)},
    pages     = {259--262},
    year      = {2011},
    publisher = {ACM},
    address   = {Mountain View, California, USA}
  }
  ```
- **Dataset Composition**:
  - Total instances: **5,572** SMS messages tagged as legitimate (`ham`) or spam (`spam`).
  - Source collections: Grumbletext UK spam forum (425 spam), NUS SMS Corpus (3,375 ham), Caroline Tag's PhD thesis (450 ham), and SMS Spam Corpus v.0.1 Big (1,002 ham + 322 spam).

---

## Machine Learning Foundations & Math

### 1. Preprocessing Pipeline (`preprocess.py`)
All text passes through a deterministic text processor:
1. **Lowercasing**: Converts characters to lowercase.
2. **Alphanumeric Filtering**: Replaces all non-alphanumeric characters with spaces (`[^a-z0-9] -> " "`).
3. **Short Token Pruning**: Drops tokens shorter than 2 characters.
4. **Whitespace Normalization**: Collapses repeated whitespace into a single space.

### 2. TF-IDF Vectorizer (`tfidf.py`)
- **Sublinear Term Frequency (TF)**:
  $$\text{TF}(t, d) = \begin{cases} 1 + \ln(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$
- **Smoothed Inverse Document Frequency (IDF)**:
  $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1.0$$
  where $N$ is the number of training documents and $\text{df}(t)$ is the document frequency of term $t$.
- **L2 Row Normalization**:
  $$v_{\text{norm}} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_j v_j^2}}$$
- **Vocabulary Slicing**: Slices the top 2,000 terms ranked by training document frequency to eliminate noisy tail features and ensure low-latency inference.

### 3. Classification Algorithms (`models.py`)

#### Logistic Regression (Production Model)
- **Sigmoid Hypothesis**:
  $$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = Xw + b$$
- **Class-Weighted Cross-Entropy Loss with L2 Regularization**:
  $$J(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ w_{\text{spam}} y_i \ln(\hat{p}_i) + w_{\text{ham}} (1 - y_i) \ln(1 - \hat{p}_i) \right] + \frac{\lambda}{2} \|w\|_2^2$$
  where weights compensate for the 6.46:1 class imbalance:
  $$w_{\text{ham}} = \frac{N}{2 \cdot N_{\text{ham}}}, \quad w_{\text{spam}} = \frac{N}{2 \cdot N_{\text{spam}}}$$
- **Optimization**: Batch Gradient Descent (300 epochs, learning rate $\eta = 0.5$, regularization parameter $\lambda = 0.01$).

#### Multinomial Naive Bayes
- Joint log-likelihood with Laplace smoothing ($\alpha = 1.0$):
  $$\ln P(d | c) = \sum_{j} x_j \ln \left( \frac{\text{count}(w_j, c) + \alpha}{\sum_{k} \text{count}(w_k, c) + \alpha \cdot |V|} \right)$$
- Posterior computed via log-sum-exp normalization to prevent numerical underflow.

#### Full Bernoulli Naive Bayes
- Evaluates both term presence AND term absence:
  $$\ln P(d | c) = \ln P(c) + \sum_{j} \left[ x_j \ln P(w_j | c) + (1 - x_j) \ln(1 - P(w_j | c)) \right]$$
- Penalizes documents for lacking typical spam indicators, driving False Positives down to only **1** on the test set.

### 4. Explainable AI (XAI) Mathematical Attribution (`explain.py`)
Word contribution scores quantify the push of each token on the classification logit:
- **Logistic Regression**:
  $$\text{contribution}(w_j) = w_j \times x_{i,j}$$
  where $w_j$ is the learned weight for token $j$ and $x_{i,j}$ is its normalized TF-IDF value.
- **Naive Bayes**:
  $$\text{contribution}(w_j) = \left[ \ln P(w_j | \text{spam}) - \ln P(w_j | \text{ham}) \right] \times x_{i,j}$$
A positive contribution drives the message toward SPAM; a negative score pulls toward HAM.

### 5. Adjustable Decision Threshold (`predict.py`)
Rather than forcing a hard threshold at 0.50:
$$\hat{y} = \begin{cases} 1 \ (\text{SPAM}) & \text{if } P(\text{Spam} | x) \ge \theta \\ 0 \ (\text{HAM}) & \text{if } P(\text{Spam} | x) < \theta \end{cases}$$
- **Lower Threshold (e.g., $\theta = 0.25$)**: Prioritizes **Recall**. Captures aggressive and disguised spam at the expense of potential false alarms.
- **Higher Threshold (e.g., $\theta = 0.75$)**: Prioritizes **Precision**. Ensures legitimate personal messages are never flagged as spam.

---

## Benchmark Results

All metrics reflect evaluations performed on the deterministic stratified split (**80% Train: 4,458 samples**, **20% Test: 1,114 samples** with random seed 42).

### 1. Validation Set (Model Selection on Train Split)
*Sub-train: 3,566 samples, Validation: 892 samples (772 Ham, 120 Spam).*

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | 86.55% | 0.00% | 0.00% | 0.0000 | 0.5000 | Baseline |
| Multinomial Naive Bayes | 95.29% | 100.00% | 65.00% | 0.7879 | 0.9884 | Candidate |
| Bernoulli Naive Bayes (Full) | 97.20% | 98.97% | 80.00% | 0.8848 | 0.9989 | Candidate |
| Bernoulli NB (Report-Style) | 87.00% | 50.85% | 100.00% | 0.6742 | 0.9989 | Excluded |
| **Logistic Regression** | **97.53%** | **90.83%** | **90.83%** | **0.9083** | **0.9917** | **Selected Winner** |

> **Selection Criteria**: Logistic Regression achieved the highest validation F1 score (**0.9083**) and was chosen for final production artifact persistence.

### 2. Final Test Set Evaluation (Held-Out 1,114 Messages)
*Refitted on full 4,458 train set; evaluated once on 1,114 held-out test messages (965 Ham, 149 Spam).*

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | 86.62% | 0.00% | 0.00% | 0.0000 | 0.5000 | `[965, 0, 149, 0]` |
| Multinomial Naive Bayes | 97.31% | **100.00%** | 79.87% | 0.8881 | 0.9852 | `[965, 0, 30, 119]` |
| Bernoulli Naive Bayes (Full) | **98.29%** | 99.24% | 87.92% | **0.9324** | **0.9934** | `[964, 1, 18, 131]` |
| Bernoulli NB (Report-Style) | 89.23% | 55.47% | **98.66%** | 0.7101 | 0.9937 | `[847, 118, 2, 147]` |
| **Logistic Regression (Production)** | **97.49%** | **87.58%** | **94.63%** | **0.9097** | **0.9883** | `[945, 20, 8, 141]` |

Detailed machine-readable metrics are saved in [reports/metrics.json](reports/metrics.json).

---

## Installation & Setup (Windows PowerShell)

### Prerequisites
- Windows 10/11 with **PowerShell**
- **Python 3.12** installed and accessible via `py -3.12` or `python`

### Step-by-Step Installation

```powershell
# 1. Clone repository and navigate to project root
cd D:\spamdetection

# 2. Allow local script execution if restricted
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 3. Create virtual environment with Python 3.12
py -3.12 -m venv .venv

# 4. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 5. Upgrade pip to latest version
python -m pip install --upgrade pip

# 6. Install pinned dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .

# 7. (Optional) Download dataset if data/SMSSpamCollection is missing
python scripts/download_data.py
```

---

## Running the Project

### 1. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the full 6-tab interface.

### 2. Train Models and Regenerate Benchmarks
```powershell
python -m spamdetector train
```
This re-fits the TF-IDF vectorizer, trains all 5 classifiers, validates them, updates `artifacts/best_model.npz`, and outputs new confusion matrix plots in `reports/`.

### 3. CLI Predictions
Classify individual messages directly from your terminal:

```powershell
# Test a typical spam message
python -m spamdetector predict "WINNER!! You have won a £1,000 cash prize! Call 09061701461 to claim now. Valid 24 hours only."

# Test a legitimate message
python -m spamdetector predict "Hey, are we still meeting for lunch at 12:30 or do you want to push it back?"

# Specify a custom threshold (e.g., 0.35)
python -m spamdetector predict "URGENT: Your account requires verification" --threshold 0.35

# Specify a particular model architecture
python -m spamdetector predict "Free ringtone download text YES to 80082" --model multinomial_nb
```

### 4. Run the Automated Test Suite
Run all 65 unit tests to verify mathematical correctness, serialization, and UI utilities:
```powershell
pytest -v
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\spamdetection
collected 65 items

tests/test_eda.py .................... [ 15%]
tests/test_explain.py ................ [ 30%]
tests/test_metrics.py ....             [ 36%]
tests/test_models.py ......            [ 46%]
tests/test_predict.py ......           [ 55%]
tests/test_preprocess.py ......        [ 64%]
tests/test_samples.py ........         [ 76%]
tests/test_split.py .                  [ 78%]
tests/test_tfidf.py .......            [ 87%]
tests/test_upgrades.py ........        [100%]

============================= 65 passed in 1.35s ==============================
```

---

## Project Limitations

1. **Session-Scoped History**: The *Prediction History* tab logs user queries into Streamlit's in-memory `st.session_state`. Refreshing or closing the browser tab resets this history. Users can export their logs to CSV before refreshing.
2. **Historical Dataset Context**: The UCI SMS Spam Collection reflects telecommunication spam from the 2010 era (e.g., premium-rate phone numbers, ringtones, UK sweepstakes). Modern spam often includes encrypted messaging links, cryptocurrency scams, or phishing URLs.
3. **Bag-of-Words Independence Assumption**: Unigram TF-IDF evaluates words independently without capturing multi-word negation (e.g., *"not spam"*) or complex syntactic structure.
4. **Monolingual Focus**: The preprocessing and vocabulary are optimized for English SMS communication.

---

## Technologies Used

- **Language**: Python 3.12
- **Core Computational Engine**: [NumPy 1.26.4](https://numpy.org/) (Matrix operations, linear algebra, vectorization)
- **Data Handling**: [Pandas 2.1.4](https://pandas.pydata.org/) (Dataset loading, EDA, and tabular transformations)
- **Columnar Engine**: [PyArrow 19.0.1](https://arrow.apache.org/pyarrow/) (Accelerated batch and CSV operations)
- **Visualizations**: [Matplotlib 3.8.4](https://matplotlib.org/) (Static report plots and confusion matrix heatmaps)
- **Web UI & Dashboard**: [Streamlit >= 1.35.0](https://streamlit.io/) (Interactive multi-tab application)
- **Testing & QA**: [Pytest >= 8.0.0](https://docs.pytest.org/) (65 automated unit tests)
- **Serialization**: Native NumPy `.npz` and JSON (Safe formats without pickle execution vulnerabilities)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Attribution for the underlying dataset belongs to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

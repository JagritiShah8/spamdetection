# Spam SMS Detection - Model Benchmark Report

## Dataset Split Summary
- **Base Training Samples**: 4458 (Stratified 80%)
- **Augmented Training Samples**: 4519 (Modern smishing & legit bank/OTP samples added only to train)
- **Unseen Held-out Test Samples**: 1114 (Stratified 20% untouched)
- **Test Class Distribution**: 965 Ham / 149 Spam

## Validation Set Comparison (Model Selection)
*Models trained on 80% sub-train; evaluated on 20% validation split carved from train.*

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 86.55% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.50 |
| **Multinomial Naive Bayes** | 97.09% | 100.00% | 78.33% | 0.8785 | 0.9885 | 0.30 |
| **Bernoulli Naive Bayes** | 98.21% | 100.00% | 86.67% | 0.9286 | 0.9983 | 0.20 |
| **Bernoulli NB (Report-Style)** | 90.25% | 57.97% | 100.00% | 0.7339 | 0.9982 | 0.80 |
| **Logistic Regression** | 98.88% | 96.61% | 95.00% | 0.9580 | 0.9972 | 0.60 |

**Selected Best Model**: **Logistic Regression** (highest validation F1: 0.9580, optimal threshold: 0.60).

## Final Test Set Evaluation (Default Threshold = 0.50)
*Refitted on full 4519 training samples; evaluated on 1114 unseen test samples.*

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 86.62% | 0.00% | 0.00% | 0.0000 | 0.5000 | [965, 0, 149, 0] |
| **Multinomial Naive Bayes** | 97.67% | 98.43% | 83.89% | 0.9058 | 0.9855 | [963, 2, 24, 125] |
| **Bernoulli Naive Bayes** | 98.65% | 100.00% | 89.93% | 0.9470 | 0.9941 | [965, 0, 15, 134] |
| **Bernoulli NB (Report-Style)** | 90.75% | 59.35% | 97.99% | 0.7392 | 0.9943 | [865, 100, 3, 146] |
| **Logistic Regression** | 98.92% | 96.60% | 95.30% | 0.9595 | 0.9941 | [960, 5, 7, 142] |

## Final Test Set Evaluation (Validation-Tuned Optimal Threshold)
*Evaluated using the threshold selected via validation set optimization.*

| Model | Threshold | Accuracy | Precision | Recall | F1 Score | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.50 | 86.62% | 0.00% | 0.00% | 0.0000 | [965, 0, 149, 0] |
| **Multinomial Naive Bayes** | 0.30 | 98.65% | 98.55% | 91.28% | 0.9477 | [963, 2, 13, 136] |
| **Bernoulli Naive Bayes** | 0.20 | 98.65% | 100.00% | 89.93% | 0.9470 | [965, 0, 15, 134] |
| **Bernoulli NB (Report-Style)** | 0.80 | 94.34% | 71.29% | 96.64% | 0.8205 | [907, 58, 5, 144] |
| **Logistic Regression** | 0.60 | 98.65% | 97.86% | 91.95% | 0.9481 | [962, 3, 12, 137] |
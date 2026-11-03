# Spam SMS Detection - Model Benchmark Report

## Dataset Split Summary
- **Total Training Samples**: 4458 (Stratified 80%)
- **Total Test Samples**: 1114 (Stratified 20%)
- **Test Class Distribution**: 965 Ham / 149 Spam

## Validation Set Comparison (Model Selection)
*Models trained on 80% sub-train; evaluated on 20% validation split carved from train.*

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 86.55% | 0.00% | 0.00% | 0.0000 | 0.5000 |
| **Multinomial Naive Bayes** | 95.29% | 100.00% | 65.00% | 0.7879 | 0.9884 |
| **Bernoulli Naive Bayes** | 97.20% | 98.97% | 80.00% | 0.8848 | 0.9989 |
| **Bernoulli NB (Report-Style)** | 87.00% | 50.85% | 100.00% | 0.6742 | 0.9989 |
| **Logistic Regression** | 97.53% | 90.83% | 90.83% | 0.9083 | 0.9917 |

**Selected Best Model**: **Logistic Regression** (highest validation F1).

## Final Test Set Evaluation
*Refitted on full 4,458 training samples; evaluated on 1,114 unseen test samples.*

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 86.62% | 0.00% | 0.00% | 0.0000 | 0.5000 | [965, 0, 149, 0] |
| **Multinomial Naive Bayes** | 97.31% | 100.00% | 79.87% | 0.8881 | 0.9852 | [965, 0, 30, 119] |
| **Bernoulli Naive Bayes** | 98.29% | 99.24% | 87.92% | 0.9324 | 0.9934 | [964, 1, 18, 131] |
| **Bernoulli NB (Report-Style)** | 89.23% | 55.47% | 98.66% | 0.7101 | 0.9937 | [847, 118, 2, 147] |
| **Logistic Regression** | 97.49% | 87.58% | 94.63% | 0.9097 | 0.9883 | [945, 20, 8, 141] |
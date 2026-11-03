"""Prototype script to inspect exact metrics for LR and MNB on real data."""

import numpy as np
import pandas as pd
from spamdetector.dataset import load_sms_dataset, stratified_split
from spamdetector.tfidf import TfidfVectorizer
from spamdetector.models import LogisticRegression, MultinomialNaiveBayes
from spamdetector.metrics import compute_classification_metrics

df = load_sms_dataset("data/SMSSpamCollection")
y_all = (df["label"] == "spam").astype(int).values

train_idx, test_idx = stratified_split(y_all, test_ratio=0.2, random_state=42)

train_texts = df["text"].iloc[train_idx].tolist()
test_texts = df["text"].iloc[test_idx].tolist()
y_train = y_all[train_idx]
y_test = y_all[test_idx]

print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
print(f"Test class distribution: Ham={np.sum(y_test==0)}, Spam={np.sum(y_test==1)}")

tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_texts)
X_test = tfidf.transform(test_texts)

# Logistic Regression
lr = LogisticRegression(n_features=2000, epochs=300, lr=0.5, l2=0.01)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_m = compute_classification_metrics(y_test, lr_preds, lr_probs)

# Multinomial Naive Bayes
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_train, y_train)
mnb_preds = mnb.predict(X_test)
mnb_probs = mnb.predict_proba(X_test)[:, 1]
mnb_m = compute_classification_metrics(y_test, mnb_preds, mnb_probs)

print("\n--- Prototype Results on Real Data (Test Set) ---")
print("Logistic Regression (first 2000 features):")
print(f"  Accuracy : {lr_m['accuracy']*100:.2f}%")
print(f"  Precision: {lr_m['precision']*100:.2f}%")
print(f"  Recall   : {lr_m['recall']*100:.2f}%")
print(f"  F1 Score : {lr_m['f1']:.4f}")
print(f"  ROC-AUC  : {lr_m['roc_auc']:.4f}")

print("\nMultinomial Naive Bayes:")
print(f"  Accuracy : {mnb_m['accuracy']*100:.2f}%")
print(f"  Precision: {mnb_m['precision']*100:.2f}%")
print(f"  Recall   : {mnb_m['recall']*100:.2f}%")
print(f"  F1 Score : {mnb_m['f1']:.4f}")
print(f"  ROC-AUC  : {mnb_m['roc_auc']:.4f}")

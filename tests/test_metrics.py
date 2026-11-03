"""Unit tests for manual metrics calculations against hand-computed values."""

import numpy as np
import pytest

from spamdetector.metrics import (
    accuracy_score,
    compute_classification_metrics,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def test_confusion_matrix_hand_computed():
    # 4 Ham (0), 6 Spam (1)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 0])

    # TN: y_true=0, y_pred=0 -> 3
    # FP: y_true=0, y_pred=1 -> 1
    # FN: y_true=1, y_pred=0 -> 2
    # TP: y_true=1, y_pred=1 -> 4
    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[3, 1], [2, 4]]


def test_metrics_hand_computed_values():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 0])

    # Accuracy = (4 + 3) / 10 = 0.70
    assert np.isclose(accuracy_score(y_true, y_pred), 0.70)

    # Precision = TP / (TP + FP) = 4 / (4 + 1) = 4/5 = 0.80
    assert np.isclose(precision_score(y_true, y_pred), 0.80)

    # Recall = TP / (TP + FN) = 4 / (4 + 2) = 4/6 = 2/3
    assert np.isclose(recall_score(y_true, y_pred), 2.0 / 3.0)

    # F1 = 2 * (0.8 * (2/3)) / (0.8 + 2/3) = (3.2/3) / (4.4/3) = 3.2 / 4.4 = 0.7272727...
    expected_f1 = 2.0 * 0.8 * (2.0 / 3.0) / (0.8 + 2.0 / 3.0)
    assert np.isclose(f1_score(y_true, y_pred), expected_f1)


def test_roc_auc_score_hand_computed():
    # Perfect separation
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(roc_auc_score(y_true, y_score), 1.0)

    # Completely inverted
    y_score_inv = np.array([0.9, 0.8, 0.2, 0.1])
    assert np.isclose(roc_auc_score(y_true, y_score_inv), 0.0)

    # Known intermediate example
    # Pairs (true, score): (0, 0.1), (1, 0.4), (0, 0.35), (1, 0.8)
    # Ranks:
    # 0.1 (neg): rank 1
    # 0.35 (neg): rank 2
    # 0.4 (pos): rank 3
    # 0.8 (pos): rank 4
    # Positive pairs vs negative pairs:
    # pos score 0.4 > 0.1 (yes), > 0.35 (yes) -> 2
    # pos score 0.8 > 0.1 (yes), > 0.35 (yes) -> 2
    # AUC = 4 / (2 * 2) = 1.0
    y_t2 = np.array([0, 1, 0, 1])
    y_s2 = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(roc_auc_score(y_t2, y_s2), 1.0)


def test_compute_classification_metrics_dict():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.9, 0.6, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]

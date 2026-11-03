"""Manual classification metrics for spam SMS detection.

Computes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
manually for the positive class (spam: label 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) == 0:
        return 0.0
    return float(np.mean(y_true_arr == y_pred_arr))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision for the positive class (1: spam)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall for the positive class (1: spam)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score for the positive class (1: spam)."""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    denom = prec + rec
    return float(2.0 * prec * rec / denom) if denom > 0.0 else 0.0


def roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve points.

    Returns:
        fpr: False Positive Rates array starting at 0.0 and ending at 1.0.
        tpr: True Positive Rates array starting at 0.0 and ending at 1.0.
        thresholds: Decreasing score thresholds.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    n_pos = np.sum(y_true_arr == 1)
    n_neg = np.sum(y_true_arr == 0)

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    # Sort descending by score
    desc_idx = np.argsort(-y_score_arr)
    y_sorted = y_true_arr[desc_idx]
    scores_sorted = y_score_arr[desc_idx]

    # Find distinct threshold indices
    distinct_value_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_arr.size - 1]

    # Cumulative true positives and false positives
    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    # Prepend (0, 0) point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[scores_sorted[0] + 1.0, scores_sorted[threshold_idxs]]

    tpr = tps / n_pos
    fpr = fps / n_neg
    return fpr, tpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Uses trapezoidal integration over sorted distinct score thresholds.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Trapezoidal integration: area = sum of (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return float(np.trapz(tpr, fpr))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute full suite of classification metrics.

    Args:
        y_true: Ground truth binary labels (0: ham, 1: spam).
        y_pred: Predicted binary labels.
        y_score: Optional predicted probabilities of spam for ROC-AUC.

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, and confusion_matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    auc_val = 0.5
    if y_score is not None:
        try:
            auc_val = roc_auc_score(y_true, y_score)
        except Exception:
            auc_val = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc_val),
        "confusion_matrix": cm.tolist(),
    }

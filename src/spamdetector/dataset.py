"""Dataset loading and stratified splitting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_sms_dataset(filepath: str | Path = "data/SMSSpamCollection") -> pd.DataFrame:
    """Load SMSSpamCollection tab-separated file without header.

    Tries UTF-8 first, falls back to latin-1.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="latin-1",
        )
    return df


def stratified_split(
    labels: np.ndarray | pd.Series,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform deterministic stratified train/test split.

    Mirrors Appendix B authoritative logic:
    - rng = np.random.RandomState(42)
    - for cls in (0, 1): idx = np.where(y == cls)[0]; rng.shuffle(idx); first round(len(idx)*0.2) go to test
    - Train = remaining rows in ORIGINAL file order (no extra shuffles)

    Args:
        labels: 1D array or Series of labels (e.g. 'ham'/'spam' or 0/1).
        test_ratio: Fraction for test set (default 0.2).
        random_state: Random seed for shuffling.

    Returns:
        (train_indices, test_indices) as 1D numpy integer arrays.
    """
    labels_arr = np.asarray(labels)
    # Convert 'ham'/'spam' strings to 0/1 if needed
    if labels_arr.dtype.kind in {"U", "O", "S"}:
        y = (labels_arr == "spam").astype(int)
    else:
        y = labels_arr.astype(int)

    rng = np.random.RandomState(random_state)
    test_indices_list = []
    train_indices_list = []

    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_indices_list.append(idx[:n_test])
        train_indices_list.append(idx[n_test:])

    test_indices = np.concatenate(test_indices_list)
    # Train = remaining rows in ORIGINAL file order (no extra shuffles)
    train_indices = np.sort(np.concatenate(train_indices_list))

    return train_indices, test_indices


# Re-export EDA functions for convenience
from spamdetector.eda import (
    compute_class_distribution,
    compute_full_dataset_summary,
    compute_length_statistics,
    compute_lexical_diversity,
    compute_vocabulary_overlap,
    compute_word_frequencies,
)

__all__ = [
    "load_sms_dataset",
    "stratified_split",
    "compute_class_distribution",
    "compute_length_statistics",
    "compute_word_frequencies",
    "compute_vocabulary_overlap",
    "compute_lexical_diversity",
    "compute_full_dataset_summary",
]


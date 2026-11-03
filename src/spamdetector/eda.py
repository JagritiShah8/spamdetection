"""Exploratory Data Analysis (EDA) and dataset analytics engine.

Provides reusable statistical computations for SMS spam datasets:
- Class distribution and imbalance ratios
- Character and token length distributions (mean, median, std, quartiles)
- Word and token frequency extraction (using shared preprocess_text)
- Vocabulary overlap, class-specific lexicons, and Jaccard similarity
- Lexical diversity (Type-Token Ratio)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from spamdetector.preprocess import preprocess_text


def _validate_dataframe(
    df: Any,
    required_cols: Sequence[str] = ("label", "text"),
) -> pd.DataFrame:
    """Validate input DataFrame and required columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    return df


def compute_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
) -> Dict[str, Any]:
    """Calculate message counts and percentages by class.

    Args:
        df: Input DataFrame containing messages and labels.
        label_col: Name of the class label column.

    Returns:
        Dict with total_messages, ham_count, spam_count, ham_percentage,
        spam_percentage, and imbalance_ratio.
    """
    _validate_dataframe(df, required_cols=[label_col])
    labels = df[label_col].astype(str).str.strip().str.lower()
    total = len(labels)

    ham_count = int((labels == "ham").sum())
    spam_count = int((labels == "spam").sum())

    ham_pct = (ham_count / total * 100.0) if total > 0 else 0.0
    spam_pct = (spam_count / total * 100.0) if total > 0 else 0.0

    imbalance_ratio = (ham_count / max(1, spam_count)) if spam_count > 0 else float("inf")

    return {
        "total_messages": total,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_percentage": round(ham_pct, 2),
        "spam_percentage": round(spam_pct, 2),
        "imbalance_ratio": round(imbalance_ratio, 2),
    }


def _calc_stats(values: Sequence[Union[int, float]]) -> Dict[str, float]:
    """Compute summary statistics for a numeric sequence."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
        }

    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(np.std(arr, ddof=1 if len(arr) > 1 else 0)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "q25": round(q25, 2),
        "q75": round(q75, 2),
        "iqr": round(q75 - q25, 2),
    }


def compute_length_statistics(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Dict[str, Any]:
    """Calculate character and token length distributions per class.

    Tokenization strictly uses the shared preprocess_text function.

    Args:
        df: Input DataFrame containing messages and labels.
        text_col: Name of the text column.
        label_col: Name of the class label column.

    Returns:
        Nested dict with char_length and token_length statistics for 'all', 'ham', and 'spam',
        along with raw length series for distribution plotting.
    """
    _validate_dataframe(df, required_cols=[text_col, label_col])

    df_clean = df.copy()
    df_clean["_norm_label"] = df_clean[label_col].astype(str).str.strip().str.lower()
    df_clean["_text_str"] = df_clean[text_col].fillna("").astype(str)

    # Character lengths
    char_lens_all = df_clean["_text_str"].str.len().tolist()
    char_lens_ham = df_clean[df_clean["_norm_label"] == "ham"]["_text_str"].str.len().tolist()
    char_lens_spam = df_clean[df_clean["_norm_label"] == "spam"]["_text_str"].str.len().tolist()

    # Token lengths (using preprocess_text)
    token_lens_all = [len(preprocess_text(t)) for t in df_clean["_text_str"]]
    token_lens_ham = [
        len(preprocess_text(t)) for t in df_clean[df_clean["_norm_label"] == "ham"]["_text_str"]
    ]
    token_lens_spam = [
        len(preprocess_text(t)) for t in df_clean[df_clean["_norm_label"] == "spam"]["_text_str"]
    ]

    return {
        "char_length": {
            "all": _calc_stats(char_lens_all),
            "ham": _calc_stats(char_lens_ham),
            "spam": _calc_stats(char_lens_spam),
        },
        "token_length": {
            "all": _calc_stats(token_lens_all),
            "ham": _calc_stats(token_lens_ham),
            "spam": _calc_stats(token_lens_spam),
        },
        "series": {
            "char_all": char_lens_all,
            "char_ham": char_lens_ham,
            "char_spam": char_lens_spam,
            "token_all": token_lens_all,
            "token_ham": token_lens_ham,
            "token_spam": token_lens_spam,
        },
    }


def compute_word_frequencies(
    df: pd.DataFrame,
    top_n: int = 20,
    text_col: str = "text",
    label_col: str = "label",
) -> Dict[str, Any]:
    """Calculate most frequent words across all, ham, and spam messages.

    Tokenization strictly uses preprocess_text.

    Args:
        df: Input DataFrame.
        top_n: Number of top words to extract (default 20).
        text_col: Name of text column.
        label_col: Name of label column.

    Returns:
        Dict with top_words for each class, total token counts, and counters.
    """
    _validate_dataframe(df, required_cols=[text_col, label_col])
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")

    df_clean = df.copy()
    df_clean["_norm_label"] = df_clean[label_col].astype(str).str.strip().str.lower()
    df_clean["_text_str"] = df_clean[text_col].fillna("").astype(str)

    ham_texts = df_clean[df_clean["_norm_label"] == "ham"]["_text_str"]
    spam_texts = df_clean[df_clean["_norm_label"] == "spam"]["_text_str"]

    ham_counter: Counter[str] = Counter()
    for text in ham_texts:
        ham_counter.update(preprocess_text(text))

    spam_counter: Counter[str] = Counter()
    for text in spam_texts:
        spam_counter.update(preprocess_text(text))

    all_counter = ham_counter + spam_counter

    return {
        "top_n": top_n,
        "top_spam_words": spam_counter.most_common(top_n),
        "top_ham_words": ham_counter.most_common(top_n),
        "top_all_words": all_counter.most_common(top_n),
        "total_spam_tokens": int(sum(spam_counter.values())),
        "total_ham_tokens": int(sum(ham_counter.values())),
        "total_tokens": int(sum(all_counter.values())),
    }


def compute_vocabulary_overlap(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Dict[str, Any]:
    """Analyze vocabulary intersection, unique lexicons, and Jaccard similarity.

    Args:
        df: Input DataFrame.
        text_col: Name of text column.
        label_col: Name of label column.

    Returns:
        Dict detailing vocabulary sizes, shared words, class-specific words,
        and overlap metrics.
    """
    _validate_dataframe(df, required_cols=[text_col, label_col])

    df_clean = df.copy()
    df_clean["_norm_label"] = df_clean[label_col].astype(str).str.strip().str.lower()
    df_clean["_text_str"] = df_clean[text_col].fillna("").astype(str)

    ham_counter: Counter[str] = Counter()
    for text in df_clean[df_clean["_norm_label"] == "ham"]["_text_str"]:
        ham_counter.update(preprocess_text(text))

    spam_counter: Counter[str] = Counter()
    for text in df_clean[df_clean["_norm_label"] == "spam"]["_text_str"]:
        spam_counter.update(preprocess_text(text))

    ham_vocab = set(ham_counter.keys())
    spam_vocab = set(spam_counter.keys())
    all_vocab = ham_vocab | spam_vocab

    shared_vocab = ham_vocab & spam_vocab
    spam_only_vocab = spam_vocab - ham_vocab
    ham_only_vocab = ham_vocab - spam_vocab

    jaccard = (len(shared_vocab) / len(all_vocab)) if len(all_vocab) > 0 else 0.0
    spam_shared_ratio = (len(shared_vocab) / len(spam_vocab)) if len(spam_vocab) > 0 else 0.0
    ham_shared_ratio = (len(shared_vocab) / len(ham_vocab)) if len(ham_vocab) > 0 else 0.0

    # Top words unique to spam (by occurrence frequency)
    spam_only_ranked = [(w, spam_counter[w]) for w in spam_only_vocab]
    spam_only_ranked.sort(key=lambda x: -x[1])

    # Top words unique to ham (by occurrence frequency)
    ham_only_ranked = [(w, ham_counter[w]) for w in ham_only_vocab]
    ham_only_ranked.sort(key=lambda x: -x[1])

    # Top shared words
    shared_ranked = [(w, spam_counter[w] + ham_counter[w]) for w in shared_vocab]
    shared_ranked.sort(key=lambda x: -x[1])

    return {
        "total_unique_words": len(all_vocab),
        "ham_vocab_size": len(ham_vocab),
        "spam_vocab_size": len(spam_vocab),
        "shared_vocab_size": len(shared_vocab),
        "spam_only_vocab_size": len(spam_only_vocab),
        "ham_only_vocab_size": len(ham_only_vocab),
        "jaccard_similarity": round(jaccard, 4),
        "spam_vocabulary_in_ham_ratio": round(spam_shared_ratio, 4),
        "ham_vocabulary_in_spam_ratio": round(ham_shared_ratio, 4),
        "top_spam_only_words": spam_only_ranked[:20],
        "top_ham_only_words": ham_only_ranked[:20],
        "top_shared_words": shared_ranked[:20],
    }


def compute_lexical_diversity(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Dict[str, Any]:
    """Calculate Type-Token Ratio (TTR) and lexical diversity metrics.

    TTR = Unique Tokens / Total Tokens. Higher TTR indicates greater lexical richness.

    Args:
        df: Input DataFrame.
        text_col: Name of text column.
        label_col: Name of label column.

    Returns:
        Dict with overall, ham, and spam TTR scores and per-message diversity averages.
    """
    _validate_dataframe(df, required_cols=[text_col, label_col])

    df_clean = df.copy()
    df_clean["_norm_label"] = df_clean[label_col].astype(str).str.strip().str.lower()
    df_clean["_text_str"] = df_clean[text_col].fillna("").astype(str)

    def _ttr(texts: Sequence[str]) -> Tuple[float, float]:
        tokens_all = []
        msg_ttrs = []
        for t in texts:
            toks = preprocess_text(t)
            tokens_all.extend(toks)
            if toks:
                msg_ttrs.append(len(set(toks)) / len(toks))
            else:
                msg_ttrs.append(0.0)

        corpus_ttr = (len(set(tokens_all)) / len(tokens_all)) if tokens_all else 0.0
        avg_msg_ttr = float(np.mean(msg_ttrs)) if msg_ttrs else 0.0
        return round(corpus_ttr, 4), round(avg_msg_ttr, 4)

    ham_texts = df_clean[df_clean["_norm_label"] == "ham"]["_text_str"].tolist()
    spam_texts = df_clean[df_clean["_norm_label"] == "spam"]["_text_str"].tolist()
    all_texts = df_clean["_text_str"].tolist()

    corpus_ttr_all, avg_msg_ttr_all = _ttr(all_texts)
    corpus_ttr_ham, avg_msg_ttr_ham = _ttr(ham_texts)
    corpus_ttr_spam, avg_msg_ttr_spam = _ttr(spam_texts)

    return {
        "corpus_ttr": {
            "all": corpus_ttr_all,
            "ham": corpus_ttr_ham,
            "spam": corpus_ttr_spam,
        },
        "average_message_ttr": {
            "all": avg_msg_ttr_all,
            "ham": avg_msg_ttr_ham,
            "spam": avg_msg_ttr_spam,
        },
    }


def compute_full_dataset_summary(
    df: pd.DataFrame,
    top_n_words: int = 20,
    text_col: str = "text",
    label_col: str = "label",
) -> Dict[str, Any]:
    """Execute complete EDA pipeline returning all dataset analytics.

    Args:
        df: Input DataFrame containing messages and labels.
        top_n_words: Number of top words to extract.
        text_col: Column name containing text.
        label_col: Column name containing labels.

    Returns:
        Structured dictionary containing class distribution, length stats,
        word frequencies, vocabulary overlap, and lexical diversity.
    """
    _validate_dataframe(df, required_cols=[text_col, label_col])

    return {
        "class_distribution": compute_class_distribution(df, label_col=label_col),
        "length_statistics": compute_length_statistics(df, text_col=text_col, label_col=label_col),
        "word_frequencies": compute_word_frequencies(
            df, top_n=top_n_words, text_col=text_col, label_col=label_col
        ),
        "vocabulary_overlap": compute_vocabulary_overlap(df, text_col=text_col, label_col=label_col),
        "lexical_diversity": compute_lexical_diversity(df, text_col=text_col, label_col=label_col),
    }

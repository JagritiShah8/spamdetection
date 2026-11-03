"""Spam SMS Detection Package.

Production-grade spam SMS detection built from scratch using NumPy and Pandas.
"""

from __future__ import annotations

from spamdetector.preprocess import preprocess_text
from spamdetector.tfidf import TfidfVectorizer
from spamdetector.models import (
    MajorityBaseline,
    MultinomialNaiveBayes,
    BernoulliNaiveBayes,
    BernoulliNBReportStyle,
    LogisticRegression,
)
from spamdetector.metrics import compute_classification_metrics
from spamdetector.predict import predict, predict_batch
from spamdetector.eda import (
    compute_class_distribution,
    compute_full_dataset_summary,
    compute_length_statistics,
    compute_lexical_diversity,
    compute_vocabulary_overlap,
    compute_word_frequencies,
)
from spamdetector.explain import (
    build_text_segments,
    explain_prediction,
    generate_human_explanation,
    render_highlighted_html,
)

__all__ = [
    "preprocess_text",
    "TfidfVectorizer",
    "MajorityBaseline",
    "MultinomialNaiveBayes",
    "BernoulliNaiveBayes",
    "BernoulliNBReportStyle",
    "LogisticRegression",
    "compute_classification_metrics",
    "predict",
    "predict_batch",
    "compute_class_distribution",
    "compute_length_statistics",
    "compute_word_frequencies",
    "compute_vocabulary_overlap",
    "compute_lexical_diversity",
    "compute_full_dataset_summary",
    "explain_prediction",
    "build_text_segments",
    "render_highlighted_html",
    "generate_human_explanation",
]

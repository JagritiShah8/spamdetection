"""Unit tests for Phase 2: Dataset Analytics and Exploratory Data Analysis (EDA)."""

import numpy as np
import pandas as pd
import pytest

from spamdetector.dataset import load_sms_dataset
from spamdetector.eda import (
    compute_class_distribution,
    compute_full_dataset_summary,
    compute_length_statistics,
    compute_lexical_diversity,
    compute_vocabulary_overlap,
    compute_word_frequencies,
)


@pytest.fixture(scope="module")
def real_dataset():
    """Load actual SMS dataset once for tests."""
    return load_sms_dataset("data/SMSSpamCollection")


@pytest.fixture
def toy_dataset():
    """Create a small controlled dataset for testing."""
    return pd.DataFrame({
        "label": ["ham", "spam", "HAM", "spam", "ham"],
        "text": [
            "Hello there, how are you doing today?",
            "WINNER!! Claim your £500 cash prize now by calling 0905809!",
            "Are we still meeting at the cafe tomorrow?",
            "URGENT: Your mobile account won a holiday voucher call now",
            "Sounds good, see you tomorrow afternoon.",
        ],
    })


# 1. Class Distribution Tests
def test_class_distribution_real_dataset(real_dataset):
    """Verify class distribution calculation on actual UCI dataset."""
    dist = compute_class_distribution(real_dataset)

    assert dist["total_messages"] == 5572
    assert dist["ham_count"] == 4825
    assert dist["spam_count"] == 747
    assert dist["ham_count"] + dist["spam_count"] == dist["total_messages"]
    assert np.isclose(dist["ham_percentage"], 86.59, atol=0.1)
    assert np.isclose(dist["spam_percentage"], 13.41, atol=0.1)
    assert np.isclose(dist["ham_percentage"] + dist["spam_percentage"], 100.0)
    assert dist["imbalance_ratio"] > 6.0


def test_class_distribution_toy_dataset(toy_dataset):
    """Verify class distribution on toy dataset with mixed-case labels."""
    dist = compute_class_distribution(toy_dataset)

    assert dist["total_messages"] == 5
    assert dist["ham_count"] == 3
    assert dist["spam_count"] == 2
    assert dist["ham_percentage"] == 60.0
    assert dist["spam_percentage"] == 40.0
    assert dist["imbalance_ratio"] == 1.5


# 2. Length Statistics Tests
def test_length_statistics_real_dataset(real_dataset):
    """Verify character and token length calculations on real data."""
    stats = compute_length_statistics(real_dataset)

    assert "char_length" in stats
    assert "token_length" in stats
    assert "series" in stats

    # Ham vs Spam character length contrast
    ham_char = stats["char_length"]["ham"]
    spam_char = stats["char_length"]["spam"]
    all_char = stats["char_length"]["all"]

    assert ham_char["count"] == 4825
    assert spam_char["count"] == 747
    assert all_char["count"] == 5572

    # Spam messages are significantly longer on average
    assert spam_char["mean"] > ham_char["mean"]
    assert np.isclose(ham_char["mean"], 71.48, atol=0.5)
    assert np.isclose(spam_char["mean"], 138.67, atol=0.5)

    assert spam_char["median"] > ham_char["median"]
    assert ham_char["median"] == 52.0
    assert spam_char["median"] == 149.0

    # Token length contrast
    ham_token = stats["token_length"]["ham"]
    spam_token = stats["token_length"]["spam"]
    assert spam_token["mean"] > ham_token["mean"]
    assert np.isclose(ham_token["mean"], 13.05, atol=0.5)
    assert np.isclose(spam_token["mean"], 24.24, atol=0.5)
    assert ham_token["median"] == 10.0
    assert spam_token["median"] == 25.0

    # Quartiles and IQR check
    assert ham_char["q75"] >= ham_char["q25"]
    assert np.isclose(ham_char["iqr"], ham_char["q75"] - ham_char["q25"])

    # Raw series check
    assert len(stats["series"]["char_ham"]) == 4825
    assert len(stats["series"]["char_spam"]) == 747


# 3. Word Frequencies Tests
def test_word_frequencies_real_dataset(real_dataset):
    """Verify top-20 word frequencies extraction on real dataset."""
    freq = compute_word_frequencies(real_dataset, top_n=20)

    assert len(freq["top_spam_words"]) == 20
    assert len(freq["top_ham_words"]) == 20
    assert len(freq["top_all_words"]) == 20

    # Verify frequencies are sorted descending
    spam_counts = [count for _, count in freq["top_spam_words"]]
    assert spam_counts == sorted(spam_counts, reverse=True)

    ham_counts = [count for _, count in freq["top_ham_words"]]
    assert ham_counts == sorted(ham_counts, reverse=True)

    # Discriminating words check
    spam_words = [word for word, _ in freq["top_spam_words"]]
    ham_words = [word for word, _ in freq["top_ham_words"]]

    assert any(term in spam_words for term in ["call", "free", "claim", "prize", "mobile", "txt"])
    assert any(term in ham_words for term in ["you", "to", "the", "and", "in", "me"])

    # Token total integrity
    assert freq["total_tokens"] == freq["total_spam_tokens"] + freq["total_ham_tokens"]


# 4. Vocabulary Overlap Tests
def test_vocabulary_overlap_real_dataset(real_dataset):
    """Verify vocabulary set operations and Jaccard similarity on real data."""
    overlap = compute_vocabulary_overlap(real_dataset)

    total_vocab = overlap["total_unique_words"]
    ham_vocab = overlap["ham_vocab_size"]
    spam_vocab = overlap["spam_vocab_size"]
    shared_vocab = overlap["shared_vocab_size"]
    spam_only = overlap["spam_only_vocab_size"]
    ham_only = overlap["ham_only_vocab_size"]

    # Exact mathematical relationships of sets:
    # |A U B| = |A| + |B| - |A n B|
    assert total_vocab == ham_vocab + spam_vocab - shared_vocab
    assert spam_only == spam_vocab - shared_vocab
    assert ham_only == ham_vocab - shared_vocab
    assert total_vocab == ham_only + spam_only + shared_vocab

    # Real data actual figures
    assert total_vocab == 8714
    assert ham_vocab == 6907
    assert spam_vocab == 2861
    assert shared_vocab == 1054

    # Jaccard similarity = shared / total
    expected_jaccard = round(1054 / 8714, 4)
    assert overlap["jaccard_similarity"] == expected_jaccard

    # Disjointness of class-specific top words
    spam_only_words = {w for w, _ in overlap["top_spam_only_words"]}
    ham_only_words = {w for w, _ in overlap["top_ham_only_words"]}
    assert len(spam_only_words & ham_only_words) == 0


# 5. Lexical Diversity Tests
def test_lexical_diversity_real_dataset(real_dataset):
    """Verify Type-Token Ratio (TTR) metrics on real data."""
    div = compute_lexical_diversity(real_dataset)

    assert "corpus_ttr" in div
    assert "average_message_ttr" in div

    # TTR must be in [0, 1]
    assert 0.0 < div["corpus_ttr"]["all"] <= 1.0
    assert 0.0 < div["corpus_ttr"]["ham"] <= 1.0
    assert 0.0 < div["corpus_ttr"]["spam"] <= 1.0

    assert 0.0 < div["average_message_ttr"]["all"] <= 1.0
    assert 0.0 < div["average_message_ttr"]["ham"] <= 1.0
    assert 0.0 < div["average_message_ttr"]["spam"] <= 1.0


# 6. Full Summary Pipeline Test
def test_full_dataset_summary(real_dataset):
    """Verify full summary pipeline combines all components cleanly."""
    summary = compute_full_dataset_summary(real_dataset, top_n_words=15)

    assert "class_distribution" in summary
    assert "length_statistics" in summary
    assert "word_frequencies" in summary
    assert "vocabulary_overlap" in summary
    assert "lexical_diversity" in summary
    assert summary["word_frequencies"]["top_n"] == 15


# 7. Error Handling & Edge Cases
def test_invalid_dataframe_inputs():
    """Verify proper exceptions on non-DataFrames, empty DataFrames, or missing columns."""
    # Non-DataFrame input
    for bad_input in [None, "string", 123, [1, 2, 3], {"a": 1}]:
        with pytest.raises(TypeError):
            compute_class_distribution(bad_input)

    # Empty DataFrame
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        compute_class_distribution(pd.DataFrame())

    # Missing column
    bad_df = pd.DataFrame({"random_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_class_distribution(bad_df, label_col="label")

    # Invalid top_n
    valid_df = pd.DataFrame({"label": ["ham"], "text": ["hello world"]})
    for bad_n in [-5, 0, "10", None]:
        with pytest.raises(ValueError):
            compute_word_frequencies(valid_df, top_n=bad_n)


def test_single_class_edge_cases():
    """Verify that datasets containing only one class do not trigger division by zero."""
    only_ham = pd.DataFrame({
        "label": ["ham", "ham", "ham"],
        "text": ["Hello friend", "How are you", "Good morning"],
    })
    dist_ham = compute_class_distribution(only_ham)
    assert dist_ham["ham_count"] == 3
    assert dist_ham["spam_count"] == 0
    assert dist_ham["ham_percentage"] == 100.0
    assert dist_ham["spam_percentage"] == 0.0

    stats_ham = compute_length_statistics(only_ham)
    assert stats_ham["char_length"]["spam"]["count"] == 0
    assert stats_ham["char_length"]["spam"]["mean"] == 0.0

    only_spam = pd.DataFrame({
        "label": ["spam", "spam"],
        "text": ["WINNER prize cash", "URGENT claim award"],
    })
    dist_spam = compute_class_distribution(only_spam)
    assert dist_spam["spam_count"] == 2
    assert dist_spam["ham_count"] == 0
    assert dist_spam["imbalance_ratio"] == 0.0


def test_nan_and_empty_text_messages():
    """Verify that NaN or empty text strings do not crash length or frequency analysis."""
    df_with_nans = pd.DataFrame({
        "label": ["ham", "spam", "ham"],
        "text": ["Regular message", None, ""],
    })
    stats = compute_length_statistics(df_with_nans)
    assert stats["char_length"]["all"]["count"] == 3

    freq = compute_word_frequencies(df_with_nans)
    assert isinstance(freq["top_ham_words"], list)

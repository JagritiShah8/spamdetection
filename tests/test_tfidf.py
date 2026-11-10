"""Unit tests for from-scratch TF-IDF vectorizer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from spamdetector.tfidf import TfidfVectorizer


def test_tfidf_shape_and_vocabulary():
    corpus = [
        "Free cash prize winner call now",
        "Hello friend how are you doing today",
        "Win cash prize today now",
        "Winner claim free prize cash now",
    ]
    vec = TfidfVectorizer(max_features=5)
    X = vec.fit_transform(corpus)

    assert X.shape == (4, 5)
    assert len(vec.feature_names_) == 5
    assert len(vec.vocabulary_) == 5
    # Vocabulary should be ordered by descending frequency
    assert vec.feature_names_[0] in ["prize", "cash", "now", "winner", "free"]


def test_tfidf_l2_unit_norm():
    corpus = [
        "Urgent call now for prize",
        "See you tomorrow at noon",
        "Just a test",
    ]
    vec = TfidfVectorizer(max_features=10)
    X = vec.fit_transform(corpus)

    # Every non-zero row must have L2 norm == 1.0
    norms = np.linalg.norm(X, axis=1)
    for norm in norms:
        if norm > 0:
            assert np.isclose(norm, 1.0, atol=1e-6)


def test_sublinear_tf_formula():
    corpus = ["test test test test test"]
    vec = TfidfVectorizer(max_features=10, sublinear_tf=True, norm="l2")
    vec.fit(corpus)
    # Token 'test' appears 5 times
    # Sublinear TF = 1 + ln(5)
    expected_tf = 1.0 + np.log(5.0)

    # Create unnormalized vectorizer to inspect raw tf-idf
    vec_unnorm = TfidfVectorizer(max_features=10, sublinear_tf=True, norm="none")
    vec_unnorm.fit(corpus)
    X_unnorm = vec_unnorm.transform(corpus)
    # smoothed idf on 1 doc is ln((1+1)/(1+1)) + 1 = 1.0
    assert np.isclose(X_unnorm[0, 0], expected_tf * 1.0, atol=1e-6)


def test_idf_fitted_on_train_only():
    train_docs = [
        "urgent prize claim free entry",
        "meeting with team tomorrow morning",
    ]
    test_docs = [
        "completely unseen vocabulary terms xyz abc",
        "urgent meeting tomorrow",
    ]
    vec = TfidfVectorizer(max_features=10)
    vec.fit(train_docs)

    original_vocab = dict(vec.vocabulary_)
    X_test = vec.transform(test_docs)

    # Vocabulary and IDF must remain untouched (train only)
    assert vec.vocabulary_ == original_vocab

    # Doc with only unseen tokens must produce all zeros
    assert np.all(X_test[0] == 0.0)

    # Doc with overlapping tokens must produce non-zero vector
    assert np.any(X_test[1] > 0.0)


def test_save_and_load_round_trip():
    train_docs = [
        "free entry win cash prize call now",
        "hey are we still meeting today",
        "urgent customer service alert",
    ]
    test_docs = ["free cash call alert", "unknown terms entirely"]

    vec = TfidfVectorizer(max_features=10)
    X_train_orig = vec.fit_transform(train_docs)
    X_test_orig = vec.transform(test_docs)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tfidf_test.npz"
        vec.save(save_path)

        loaded_vec = TfidfVectorizer.load(save_path)
        X_train_loaded = loaded_vec.transform(train_docs)
        X_test_loaded = loaded_vec.transform(test_docs)

        np.testing.assert_allclose(X_train_orig, X_train_loaded, atol=1e-9)
        np.testing.assert_allclose(X_test_orig, X_test_loaded, atol=1e-9)
        assert loaded_vec.feature_names_ == vec.feature_names_


def test_vocabulary_deterministic_sorting_and_top_2000_consistency():
    # Test document frequency ranking with first appearance order tie-breaking
    docs = [
        "banana cherry date",
        "apple banana cherry",
    ]
    # 'banana' appears in 2 docs, 'cherry' in 2 docs, 'date' in 1 doc, 'apple' in 1 doc
    # 'banana' appeared before 'cherry' in doc 0 -> ['banana', 'cherry']
    # 'date' appeared in doc 0 before 'apple' appeared in doc 1 -> ['date', 'apple']
    vec = TfidfVectorizer(max_features=4)
    vec.fit(docs)
    assert vec.feature_names_ == ["banana", "cherry", "date", "apple"]

    # Test top 2000 feature consistency across two independent fits on real data
    from spamdetector.dataset import load_sms_dataset
    df = load_sms_dataset("data/SMSSpamCollection")
    texts = df["text"].tolist()

    vec1 = TfidfVectorizer(max_features=5000)
    vec1.fit(texts)

    vec2 = TfidfVectorizer(max_features=5000)
    vec2.fit(texts)

    assert vec1.feature_names_[:2000] == vec2.feature_names_[:2000]
    assert len(vec1.feature_names_[:2000]) == 2000


def test_ngram_extraction_and_roundtrip():
    docs = [
        "account suspended verify immediately",
        "verify immediately to claim prize",
    ]
    vec = TfidfVectorizer(max_features=10, ngram_range=(1, 2))
    vec.fit(docs)
    assert "verify immediately" in vec.vocabulary_
    assert "account suspended" in vec.vocabulary_

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tfidf_ng.npz"
        vec.save(save_path)
        loaded = TfidfVectorizer.load(save_path)
        assert loaded.ngram_range == (1, 2)
        assert "verify immediately" in loaded.vocabulary_


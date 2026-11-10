"""Unit tests for from-scratch ML models."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from spamdetector.models import (
    BernoulliNaiveBayes,
    LogisticRegression,
    MajorityBaseline,
    MultinomialNaiveBayes,
)


@pytest.fixture
def toy_data():
    # 6 samples, 4 features
    # 4 Ham, 2 Spam
    X = np.array([
        [0.8, 0.1, 0.0, 0.0],
        [0.7, 0.2, 0.1, 0.0],
        [0.9, 0.0, 0.0, 0.1],
        [0.6, 0.3, 0.0, 0.0],
        [0.1, 0.0, 0.9, 0.7],
        [0.0, 0.1, 0.8, 0.8],
    ], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    return X, y


def test_majority_baseline(toy_data):
    X, y = toy_data
    model = MajorityBaseline()
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert np.all(preds == 0)  # Ham is majority
    assert probs.shape == (6, 2)
    assert np.allclose(probs[:, 0], 4.0 / 6.0)
    assert np.allclose(probs[:, 1], 2.0 / 6.0)


def test_multinomial_naive_bayes(toy_data):
    X, y = toy_data
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    # Probabilities must be in [0, 1] and sum to 1.0
    assert probs.shape == (6, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    # Predictions should be 0 or 1
    assert set(preds).issubset({0, 1})

    # Spam samples should have higher spam prob than ham samples
    assert probs[4, 1] > probs[0, 1]


def test_bernoulli_naive_bayes(toy_data):
    X, y = toy_data
    model = BernoulliNaiveBayes(alpha=1.0)
    model.fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    assert probs.shape == (6, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert set(preds).issubset({0, 1})
    assert probs[4, 1] > probs[0, 1]


def test_logistic_regression(toy_data):
    X, y = toy_data
    model = LogisticRegression(n_features=4, epochs=300, lr=0.5, l2=0.01)
    model.fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    assert probs.shape == (6, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert set(preds).issubset({0, 1})

    # Separation on separable toy dataset
    assert probs[4, 1] > 0.5
    assert probs[0, 1] < 0.5


def test_save_load_round_trip_all_models(toy_data):
    X, y = toy_data
    models = [
        MajorityBaseline(),
        MultinomialNaiveBayes(alpha=1.0),
        BernoulliNaiveBayes(alpha=1.0),
        LogisticRegression(n_features=4, epochs=50, lr=0.5, l2=0.01),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, model in enumerate(models):
            model.fit(X, y)
            p_orig = model.predict_proba(X)
            pred_orig = model.predict(X)

            ext = ".json" if isinstance(model, MajorityBaseline) else ".npz"
            save_path = Path(tmpdir) / f"model_{i}{ext}"
            model.save(save_path)

            loaded = type(model).load(save_path)
            p_loaded = loaded.predict_proba(X)
            pred_loaded = loaded.predict(X)

            np.testing.assert_allclose(p_orig, p_loaded, atol=1e-9)
            np.testing.assert_array_equal(pred_orig, pred_loaded)


def test_word_contributions(toy_data):
    X, y = toy_data
    model = LogisticRegression(n_features=4, epochs=100, lr=0.5, l2=0.01)
    model.fit(X, y)

    sample = X[4]  # spam sample
    contribs = model.get_word_contributions(sample)
    assert len(contribs) == len(sample)
    # Features 2 and 3 are spam indicators, should have positive contribution
    assert contribs[2] > contribs[0]


def test_lr_loss_convergence(toy_data):
    X, y = toy_data
    lr = LogisticRegression(n_features=4, epochs=200, lr=1.0, l2=0.001)
    lr.fit(X, y)
    assert len(lr.training_loss_history_) == 200
    # Loss should decrease over training
    assert lr.training_loss_history_[-1] < lr.training_loss_history_[0]


def test_bernoulli_nb_log_odds_contributions(toy_data):
    X, y = toy_data
    bnb = BernoulliNaiveBayes(alpha=1.0)
    bnb.fit(X, y)
    sample = X[4]  # sample: [0.1, 0.0, 0.9, 0.7] -> feature 1 is absent (0.0)
    contribs = bnb.get_word_contributions(sample)
    # Absent feature 1 must have 0 contribution
    assert contribs[1] == 0.0
    # Present features (0, 2, 3) must have non-zero contributions
    assert contribs[0] != 0.0
    assert contribs[2] != 0.0
    assert contribs[3] != 0.0

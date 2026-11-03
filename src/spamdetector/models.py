"""From-scratch classification models for spam SMS detection.

Includes:
- MajorityBaseline
- MultinomialNaiveBayes (alpha=1.0)
- BernoulliNaiveBayes (alpha=1.0, binary features)
- LogisticRegression (batch GD, 300 epochs, lr 0.5, L2 0.01, inverse-frequency class weights, first 2000 features)

All models expose fit, predict_proba, predict, save, and load without pickle.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for scratch ML models."""

    name: str = "BaseModel"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit model to training data and labels."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for samples in X.

        Returns:
            2D numpy array of shape (n_samples, 2), where col 0 is ham prob, col 1 is spam prob.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class (0: ham, 1: spam)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    @abstractmethod
    def save(self, filepath: str | Path) -> None:
        """Save model parameters to disk without pickle."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path) -> "BaseModel":
        """Load model parameters from disk without pickle."""
        raise NotImplementedError

    def get_word_contributions(self, x_vector: np.ndarray) -> np.ndarray:
        """Return 1D array of word contributions to spam classification for a single sample."""
        return np.zeros_like(x_vector, dtype=np.float64)


class MajorityBaseline(BaseModel):
    """Majority-class baseline model. Always predicts the most common class."""

    name = "Majority Baseline"

    def __init__(self) -> None:
        self.majority_class_: int = 0
        self.class_priors_: np.ndarray = np.array([1.0, 0.0], dtype=np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        n_samples = len(y)
        n_spam = int(np.sum(y == 1))
        n_ham = n_samples - n_spam
        self.majority_class_ = 1 if n_spam > n_ham else 0
        self.class_priors_ = np.array([n_ham / n_samples, n_spam / n_samples], dtype=np.float64)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        return np.tile(self.class_priors_, (n_samples, 1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        n_samples = X.shape[0]
        return np.full(n_samples, self.majority_class_, dtype=int)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        json_path = path if path.suffix == ".json" else path.with_suffix(".json")
        data = {
            "model_type": "MajorityBaseline",
            "majority_class": int(self.majority_class_),
            "class_priors": self.class_priors_.tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "MajorityBaseline":
        path = Path(filepath)
        json_path = path if path.suffix == ".json" else path.with_suffix(".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls()
        model.majority_class_ = int(data["majority_class"])
        model.class_priors_ = np.array(data["class_priors"], dtype=np.float64)
        return model


class MultinomialNaiveBayes(BaseModel):
    """Multinomial Naive Bayes classifier with Laplace smoothing."""

    name = "Multinomial Naive Bayes"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.class_log_prior_: np.ndarray = np.zeros(2, dtype=np.float64)
        self.feature_log_prob_: np.ndarray = np.zeros((2, 0), dtype=np.float64)
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        n_ham = np.sum(y == 0)
        n_spam = np.sum(y == 1)
        self.class_log_prior_ = np.log([n_ham / n_samples, n_spam / n_samples])

        # Separate feature sums by class
        X_ham = X[y == 0]
        X_spam = X[y == 1]

        count_ham = X_ham.sum(axis=0) + self.alpha
        count_spam = X_spam.sum(axis=0) + self.alpha

        total_ham = X_ham.sum() + self.alpha * n_features
        total_spam = X_spam.sum() + self.alpha * n_features

        self.feature_log_prob_ = np.vstack([
            np.log(count_ham / total_ham),
            np.log(count_spam / total_spam),
        ])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # log P(c | x) proportional to log P(c) + sum_j x_j * log P(w_j | c)
        j0 = self.class_log_prior_[0] + X @ self.feature_log_prob_[0]
        j1 = self.class_log_prior_[1] + X @ self.feature_log_prob_[1]

        # Numerically stable log-sum-exp normalization
        max_j = np.maximum(j0, j1)
        log_sum_exp = max_j + np.log(np.exp(j0 - max_j) + np.exp(j1 - max_j))

        p0 = np.exp(j0 - log_sum_exp)
        p1 = np.exp(j1 - log_sum_exp)
        return np.column_stack([p0, p1])

    def get_word_contributions(self, x_vector: np.ndarray) -> np.ndarray:
        """Top-word contribution: (log P(w|spam) - log P(w|ham)) * tfidf."""
        diff = self.feature_log_prob_[1] - self.feature_log_prob_[0]
        length = min(len(diff), len(x_vector))
        contrib = np.zeros_like(x_vector, dtype=np.float64)
        contrib[:length] = diff[:length] * x_vector[:length]
        return contrib

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            model_type=np.array(["MultinomialNaiveBayes"]),
            alpha=np.array([self.alpha]),
            class_log_prior=self.class_log_prior_,
            feature_log_prob=self.feature_log_prob_,
            n_features=np.array([self.n_features_]),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "MultinomialNaiveBayes":
        path = Path(filepath)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        data = np.load(npz_path, allow_pickle=True)
        model = cls(alpha=float(data["alpha"][0]))
        model.class_log_prior_ = np.array(data["class_log_prior"], dtype=np.float64)
        model.feature_log_prob_ = np.array(data["feature_log_prob"], dtype=np.float64)
        model.n_features_ = int(data["n_features"][0])
        return model


class BernoulliNaiveBayes(BaseModel):
    """Bernoulli Naive Bayes classifier with binary features and Laplace smoothing."""

    name = "Bernoulli Naive Bayes"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.class_log_prior_: np.ndarray = np.zeros(2, dtype=np.float64)
        self.feature_log_prob_: np.ndarray = np.zeros((2, 0), dtype=np.float64)
        self.feature_log_prob_neg_: np.ndarray = np.zeros((2, 0), dtype=np.float64)
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNaiveBayes":
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        X_bin = (X > 0.0).astype(np.float64)

        n_ham = np.sum(y == 0)
        n_spam = np.sum(y == 1)
        self.class_log_prior_ = np.log([n_ham / n_samples, n_spam / n_samples])

        X_ham = X_bin[y == 0]
        X_spam = X_bin[y == 1]

        # Smoothed presence probabilities: (count + alpha) / (N_c + 2 * alpha)
        p_ham = (X_ham.sum(axis=0) + self.alpha) / (n_ham + 2.0 * self.alpha)
        p_spam = (X_spam.sum(axis=0) + self.alpha) / (n_spam + 2.0 * self.alpha)

        self.feature_log_prob_ = np.vstack([np.log(p_ham), np.log(p_spam)])
        self.feature_log_prob_neg_ = np.vstack([np.log(1.0 - p_ham), np.log(1.0 - p_spam)])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_bin = (X > 0.0).astype(np.float64)
        # log P(x | c) = log P(c) + sum_j [ x_j * log P_j + (1 - x_j) * log(1 - P_j) ]
        #             = log P(c) + sum_j log(1 - P_j) + X_bin @ (log P_j - log(1 - P_j))
        w0 = self.feature_log_prob_[0] - self.feature_log_prob_neg_[0]
        w1 = self.feature_log_prob_[1] - self.feature_log_prob_neg_[1]

        b0 = self.class_log_prior_[0] + np.sum(self.feature_log_prob_neg_[0])
        b1 = self.class_log_prior_[1] + np.sum(self.feature_log_prob_neg_[1])

        j0 = b0 + X_bin @ w0
        j1 = b1 + X_bin @ w1

        max_j = np.maximum(j0, j1)
        log_sum_exp = max_j + np.log(np.exp(j0 - max_j) + np.exp(j1 - max_j))

        p0 = np.exp(j0 - log_sum_exp)
        p1 = np.exp(j1 - log_sum_exp)
        return np.column_stack([p0, p1])

    def get_word_contributions(self, x_vector: np.ndarray) -> np.ndarray:
        """Top-word contribution: (log P(w|spam) - log P(w|ham)) * tfidf."""
        diff = self.feature_log_prob_[1] - self.feature_log_prob_[0]
        length = min(len(diff), len(x_vector))
        contrib = np.zeros_like(x_vector, dtype=np.float64)
        contrib[:length] = diff[:length] * x_vector[:length]
        return contrib

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            model_type=np.array(["BernoulliNaiveBayes"]),
            alpha=np.array([self.alpha]),
            class_log_prior=self.class_log_prior_,
            feature_log_prob=self.feature_log_prob_,
            feature_log_prob_neg=self.feature_log_prob_neg_,
            n_features=np.array([self.n_features_]),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "BernoulliNaiveBayes":
        path = Path(filepath)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        data = np.load(npz_path, allow_pickle=True)
        model = cls(alpha=float(data["alpha"][0]))
        model.class_log_prior_ = np.array(data["class_log_prior"], dtype=np.float64)
        model.feature_log_prob_ = np.array(data["feature_log_prob"], dtype=np.float64)
        model.feature_log_prob_neg_ = np.array(data["feature_log_prob_neg"], dtype=np.float64)
        model.n_features_ = int(data["n_features"][0])
        return model


class BernoulliNBReportStyle(BaseModel):
    """Bernoulli Naive Bayes (Report-style formulation).

    Uses presence-only log-odds and omits the absence term log(1-p).
    Included for report comparison only; excluded from model selection
    because omitting the absence term causes massive false positives (~144 FP).
    """

    name = "Bernoulli NB (Report-Style)"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.class_log_prior_: np.ndarray = np.zeros(2, dtype=np.float64)
        self.feature_log_prob_: np.ndarray = np.zeros((2, 0), dtype=np.float64)
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNBReportStyle":
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        X_bin = (X > 0.0).astype(np.float64)

        n_ham = np.sum(y == 0)
        n_spam = np.sum(y == 1)
        self.class_log_prior_ = np.log([n_ham / n_samples, n_spam / n_samples])

        X_ham = X_bin[y == 0]
        X_spam = X_bin[y == 1]

        p_ham = (X_ham.sum(axis=0) + self.alpha) / (n_ham + 2.0 * self.alpha)
        p_spam = (X_spam.sum(axis=0) + self.alpha) / (n_spam + 2.0 * self.alpha)

        self.feature_log_prob_ = np.vstack([np.log(p_ham), np.log(p_spam)])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_bin = (X > 0.0).astype(np.float64)
        # Presence-only scores (omitting absence terms)
        j0 = self.class_log_prior_[0] + X_bin @ self.feature_log_prob_[0]
        j1 = self.class_log_prior_[1] + X_bin @ self.feature_log_prob_[1]

        max_j = np.maximum(j0, j1)
        log_sum = max_j + np.log(np.exp(j0 - max_j) + np.exp(j1 - max_j))
        p0 = np.exp(j0 - log_sum)
        p1 = np.exp(j1 - log_sum)
        return np.column_stack([p0, p1])

    def get_word_contributions(self, x_vector: np.ndarray) -> np.ndarray:
        diff = self.feature_log_prob_[1] - self.feature_log_prob_[0]
        length = min(len(diff), len(x_vector))
        contrib = np.zeros_like(x_vector, dtype=np.float64)
        contrib[:length] = diff[:length] * (x_vector[:length] > 0.0).astype(np.float64)
        return contrib

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            model_type=np.array(["BernoulliNBReportStyle"]),
            alpha=np.array([self.alpha]),
            class_log_prior=self.class_log_prior_,
            feature_log_prob=self.feature_log_prob_,
            n_features=np.array([self.n_features_]),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "BernoulliNBReportStyle":
        path = Path(filepath)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        data = np.load(npz_path, allow_pickle=True)
        model = cls(alpha=float(data["alpha"][0]))
        model.class_log_prior_ = np.array(data["class_log_prior"], dtype=np.float64)
        model.feature_log_prob_ = np.array(data["feature_log_prob"], dtype=np.float64)
        model.n_features_ = int(data["n_features"][0])
        return model


class LogisticRegression(BaseModel):
    """Logistic Regression trained via batch gradient descent with class weights and L2 regularization.

    Uses the first 2000 vocabulary features (most frequent terms).
    """

    name = "Logistic Regression"

    def __init__(
        self,
        n_features: int = 2000,
        epochs: int = 300,
        lr: float = 0.5,
        l2: float = 0.01,
        class_weight: str = "inverse_frequency",
    ) -> None:
        self.n_features = int(n_features)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.class_weight = class_weight

        self.weights_: np.ndarray = np.array([], dtype=np.float64)
        self.bias_: float = 0.0
        self.training_loss_history_: list[float] = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow in exp
        z_clipped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        # Slices first n_features (e.g. 2000 most frequent terms)
        n_samples = X.shape[0]
        actual_features = min(self.n_features, X.shape[1])
        X_sub = X[:, :actual_features]

        # Initialize parameters
        self.weights_ = np.zeros(actual_features, dtype=np.float64)
        self.bias_ = 0.0
        self.training_loss_history_ = []

        # Calculate sample weights: inverse-frequency class weights
        if self.class_weight == "inverse_frequency":
            n_ham = float(np.sum(y == 0))
            n_spam = float(np.sum(y == 1))
            w_ham = float(n_samples) / (2.0 * max(1.0, n_ham))
            w_spam = float(n_samples) / (2.0 * max(1.0, n_spam))
            sample_weights = np.where(y == 1, w_spam, w_ham)
        else:
            sample_weights = np.ones(n_samples, dtype=np.float64)

        # Batch Gradient Descent
        for _ in range(self.epochs):
            linear_output = X_sub @ self.weights_ + self.bias_
            predictions = self._sigmoid(linear_output)

            # Weighted error vector
            errors = sample_weights * (predictions - y)

            # Gradients with L2 regularization on weights (not on bias)
            grad_w = (X_sub.T @ errors) / n_samples + self.l2 * self.weights_
            grad_b = np.sum(errors) / n_samples

            # Gradient update step
            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b

            # Track loss periodically
            eps = 1e-12
            loss = -np.mean(
                sample_weights * (y * np.log(predictions + eps) + (1.0 - y) * np.log(1.0 - predictions + eps))
            ) + 0.5 * self.l2 * np.sum(self.weights_ ** 2)
            self.training_loss_history_.append(float(loss))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        actual_features = len(self.weights_)
        X_sub = X[:, :actual_features]
        linear_output = X_sub @ self.weights_ + self.bias_
        p_spam = self._sigmoid(linear_output)
        p_ham = 1.0 - p_spam
        return np.column_stack([p_ham, p_spam])

    def get_word_contributions(self, x_vector: np.ndarray) -> np.ndarray:
        """Top-word contribution: weight x tfidf value for first 2000 features."""
        actual_features = len(self.weights_)
        contrib = np.zeros_like(x_vector, dtype=np.float64)
        length = min(actual_features, len(x_vector))
        contrib[:length] = self.weights_[:length] * x_vector[:length]
        return contrib

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            model_type=np.array(["LogisticRegression"]),
            n_features=np.array([self.n_features]),
            epochs=np.array([self.epochs]),
            lr=np.array([self.lr]),
            l2=np.array([self.l2]),
            weights=self.weights_,
            bias=np.array([self.bias_]),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "LogisticRegression":
        path = Path(filepath)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        data = np.load(npz_path, allow_pickle=True)
        model = cls(
            n_features=int(data["n_features"][0]),
            epochs=int(data["epochs"][0]),
            lr=float(data["lr"][0]),
            l2=float(data["l2"][0]),
        )
        model.weights_ = np.array(data["weights"], dtype=np.float64)
        model.bias_ = float(data["bias"][0])
        return model

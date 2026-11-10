"""Inference module for spam SMS detection.

Loads saved artifacts once, handles all edge cases cleanly (empty string,
long text, unseen tokens), and calculates confidence level and word contributions.
Provides single-message and batch prediction functions with configurable decision thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spamdetector.models import (
    BaseModel,
    BernoulliNaiveBayes,
    BernoulliNBReportStyle,
    LogisticRegression,
    MajorityBaseline,
    MultinomialNaiveBayes,
)
from spamdetector.preprocess import preprocess_text
from spamdetector.tfidf import TfidfVectorizer

MODEL_CLASS_MAP = {
    "MajorityBaseline": MajorityBaseline,
    "MultinomialNaiveBayes": MultinomialNaiveBayes,
    "BernoulliNaiveBayes": BernoulliNaiveBayes,
    "BernoulliNBReportStyle": BernoulliNBReportStyle,
    "LogisticRegression": LogisticRegression,
}


def _validate_threshold(threshold: Any) -> float:
    """Validate that threshold is a numeric value between 0.0 and 1.0."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"Threshold must be a float between 0.0 and 1.0, got {type(threshold).__name__}: {threshold!r}"
        )
    val = float(threshold)
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    return val


def _make_fallback_result() -> Dict[str, Any]:
    """Return default result for empty or unrecognised inputs."""
    return {
        "label": "HAM",
        "spam_probability": 0.0,
        "confidence": "low",
        "top_words": [],
    }


class Predictor:
    """Cached inference engine for spam SMS prediction."""

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        model_name: str = "best_model",
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model_name = model_name
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[BaseModel] = None
        self.metadata: Dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        tfidf_path = self.artifacts_dir / "tfidf.npz"
        if not tfidf_path.exists():
            raise FileNotFoundError(
                f"TF-IDF artifact not found at {tfidf_path}. Run training first."
            )
        self.vectorizer = TfidfVectorizer.load(tfidf_path)

        metadata_path = self.artifacts_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Find specific model file or best model
        target_stem = self.model_name.replace(".npz", "").replace(".json", "")
        model_npz = self.artifacts_dir / f"{target_stem}.npz"
        model_json = self.artifacts_dir / f"{target_stem}.json"

        if model_npz.exists():
            data = np.load(model_npz, allow_pickle=True)
            model_type = str(data["model_type"][0])
            cls = MODEL_CLASS_MAP.get(model_type, BernoulliNaiveBayes)
            self.model = cls.load(model_npz)
        elif model_json.exists():
            self.model = MajorityBaseline.load(model_json)
        else:
            raise FileNotFoundError(f"Model artifact '{target_stem}' not found in {self.artifacts_dir}")

    def predict(self, message: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Predict whether an SMS message is spam or ham.

        Args:
            message: Raw text string to classify.
            threshold: Decision threshold for spam classification (default 0.5).

        Returns:
            {
                "label": "SPAM" | "HAM",
                "spam_probability": float,
                "confidence": "high" | "uncertain" | "low",
                "top_words": list of (word, contribution) tuples,
            }
        """
        th = _validate_threshold(threshold)

        # Edge Case 1: Empty or whitespace-only input
        if not isinstance(message, str) or not message.strip():
            return _make_fallback_result()

        # Safeguard against extremely large inputs (truncate to 50k chars)
        safe_message = message[:50000]
        tokens = preprocess_text(safe_message)

        if not tokens:
            return _make_fallback_result()

        assert self.vectorizer is not None
        assert self.model is not None

        # Check for known vocabulary tokens and n-grams
        terms = self.vectorizer._extract_terms(tokens) if hasattr(self.vectorizer, "_extract_terms") else tokens
        known_terms = [t for t in set(terms) if t in self.vectorizer.vocabulary_]

        # Edge Case 2: No known vocabulary tokens -> predict ham with low confidence
        if not known_terms:
            return _make_fallback_result()

        # Vectorize input
        X = self.vectorizer.transform([safe_message])

        # Compute probability
        proba = self.model.predict_proba(X)[0]
        spam_prob = float(proba[1])
        label = "SPAM" if spam_prob >= th else "HAM"

        # Determine confidence level per user requirements:
        # - No known vocabulary tokens => "low" (handled above)
        # - Spam probability in [0.35, 0.65] => "uncertain"
        # - Otherwise => "high"
        if 0.35 <= spam_prob <= 0.65:
            confidence = "uncertain"
        else:
            confidence = "high"

        # Calculate word contributions
        contrib_vec = self.model.get_word_contributions(X[0])
        word_contribs: List[Tuple[str, float]] = []

        for term in sorted(known_terms):
            col_idx = self.vectorizer.vocabulary_[term]
            weight = float(contrib_vec[col_idx])
            word_contribs.append((term, round(weight, 4)))

        # Sort by contribution descending (strongest spam contributors first)
        word_contribs.sort(key=lambda item: -item[1])

        return {
            "label": label,
            "spam_probability": round(spam_prob, 4),
            "confidence": confidence,
            "top_words": word_contribs,
        }

    def predict_batch(
        self,
        messages: Sequence[str],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Predict spam/ham for a batch of messages efficiently.

        Args:
            messages: Sequence of message strings.
            threshold: Decision threshold for spam classification (default 0.5).

        Returns:
            List of prediction dicts matching predict() output for each message.
        """
        if isinstance(messages, str):
            raise TypeError(
                "predict_batch expects a sequence of message strings, got a single string. Use predict() instead."
            )

        th = _validate_threshold(threshold)

        if not messages:
            return []

        assert self.vectorizer is not None
        assert self.model is not None

        messages_list = list(messages)
        n_messages = len(messages_list)
        results: List[Optional[Dict[str, Any]]] = [None] * n_messages

        # Identify messages that need model evaluation vs fast fallbacks
        valid_indices: List[int] = []
        valid_texts: List[str] = []
        valid_known_terms: List[List[str]] = []

        for idx, msg in enumerate(messages_list):
            if not isinstance(msg, str) or not msg.strip():
                results[idx] = _make_fallback_result()
                continue

            safe_msg = msg[:50000]
            tokens = preprocess_text(safe_msg)
            if not tokens:
                results[idx] = _make_fallback_result()
                continue

            terms = self.vectorizer._extract_terms(tokens) if hasattr(self.vectorizer, "_extract_terms") else tokens
            known_terms = [t for t in set(terms) if t in self.vectorizer.vocabulary_]
            if not known_terms:
                results[idx] = _make_fallback_result()
                continue

            valid_indices.append(idx)
            valid_texts.append(safe_msg)
            valid_known_terms.append(known_terms)

        # Batch vectorization and model prediction for all valid messages
        if valid_indices:
            X_valid = self.vectorizer.transform(valid_texts)
            probs_valid = self.model.predict_proba(X_valid)

            for k, (orig_idx, known_terms) in enumerate(zip(valid_indices, valid_known_terms)):
                spam_prob = float(probs_valid[k, 1])
                label = "SPAM" if spam_prob >= th else "HAM"

                if 0.35 <= spam_prob <= 0.65:
                    confidence = "uncertain"
                else:
                    confidence = "high"

                contrib_vec = self.model.get_word_contributions(X_valid[k])
                word_contribs: List[Tuple[str, float]] = []
                for term in sorted(known_terms):
                    col_idx = self.vectorizer.vocabulary_[term]
                    weight = float(contrib_vec[col_idx])
                    word_contribs.append((term, round(weight, 4)))

                word_contribs.sort(key=lambda item: -item[1])

                results[orig_idx] = {
                    "label": label,
                    "spam_probability": round(spam_prob, 4),
                    "confidence": confidence,
                    "top_words": word_contribs,
                }

        return [res for res in results if res is not None]


_predictors: Dict[Tuple[Path, str], Predictor] = {}


def predict(
    message: str,
    threshold: float = 0.5,
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "best_model",
) -> Dict[str, Any]:
    """Top-level convenience prediction function with caching.

    Args:
        message: Raw text string to classify.
        threshold: Decision threshold for spam classification (default 0.5).
        artifacts_dir: Directory containing trained model artifacts.
        model_name: Model artifact name to use.

    Returns:
        Dict with keys: label, spam_probability, confidence, top_words.
    """
    th = _validate_threshold(threshold)
    key = (Path(artifacts_dir), model_name)
    if key not in _predictors:
        _predictors[key] = Predictor(artifacts_dir=artifacts_dir, model_name=model_name)
    return _predictors[key].predict(message, threshold=th)


def predict_batch(
    messages: Sequence[str],
    threshold: float = 0.5,
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "best_model",
) -> List[Dict[str, Any]]:
    """Predict spam/ham for a batch of SMS messages efficiently.

    Args:
        messages: Sequence of SMS message strings.
        threshold: Decision threshold for spam classification (default 0.5).
        artifacts_dir: Directory containing trained model artifacts.
        model_name: Model artifact name to use.

    Returns:
        List of prediction dicts matching predict() output for each message.
    """
    th = _validate_threshold(threshold)
    key = (Path(artifacts_dir), model_name)
    if key not in _predictors:
        _predictors[key] = Predictor(artifacts_dir=artifacts_dir, model_name=model_name)
    return _predictors[key].predict_batch(messages, threshold=th)


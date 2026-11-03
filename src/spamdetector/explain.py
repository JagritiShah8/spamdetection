"""Explainable AI (XAI) and in-text highlighting engine for SMS spam detection.

Translates model parameters and TF-IDF feature weights into interpretable,
word-level evidence scores mapped directly onto raw message text with safe HTML highlighting.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spamdetector.predict import _predictors, _validate_threshold, predict, Predictor
from spamdetector.preprocess import preprocess_text


def build_text_segments(
    message: str,
    token_contributions: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Segment message text into words and delimiters with assigned contributions.

    Guarantees exact character-level preservation:
    ``"".join(s["text"] for s in segments) == message``.

    Args:
        message: Raw input text.
        token_contributions: Mapping from lowercase token string to model contribution score.

    Returns:
        List of segment dictionaries containing:
        - text: Exact substring from original message.
        - token: Cleaned lowercase token (or None for delimiters/short words).
        - contribution: Feature contribution score (0.0 for neutral/delimiters).
        - category: 'spam', 'ham', or 'neutral'.
        - is_word: Boolean indicating whether this segment is a valid token.
    """
    if not isinstance(message, str) or not message:
        return []

    pos = 0
    segments: List[Dict[str, Any]] = []

    for m in re.finditer(r"[a-zA-Z0-9]+", message):
        start, end = m.span()
        # Capture preceding delimiter (spaces, punctuation, symbols)
        if start > pos:
            delimiter_text = message[pos:start]
            segments.append({
                "text": delimiter_text,
                "token": None,
                "contribution": 0.0,
                "category": "neutral",
                "is_word": False,
            })

        word_text = m.group()
        # Tokens must be >= 2 characters to match preprocess_text
        if len(word_text) >= 2:
            token = word_text.lower()
            contrib = token_contributions.get(token, 0.0)
            if contrib > 0.0:
                cat = "spam"
            elif contrib < 0.0:
                cat = "ham"
            else:
                cat = "neutral"
            is_valid_word = True
        else:
            token = None
            contrib = 0.0
            cat = "neutral"
            is_valid_word = False

        segments.append({
            "text": word_text,
            "token": token,
            "contribution": round(contrib, 4),
            "category": cat,
            "is_word": is_valid_word,
        })
        pos = end

    # Capture any trailing delimiters
    if pos < len(message):
        segments.append({
            "text": message[pos:],
            "token": None,
            "contribution": 0.0,
            "category": "neutral",
            "is_word": False,
        })

    return segments


def render_highlighted_html(
    segments: List[Dict[str, Any]],
    max_alpha: float = 0.70,
    min_alpha: float = 0.18,
) -> str:
    """Render text segments into safe, styled HTML with colored highlight spans.

    All text is safely escaped using ``html.escape()`` to prevent XSS or tag injection.

    Args:
        segments: List of segment dictionaries from build_text_segments.
        max_alpha: Maximum opacity for highest contributing words (default 0.70).
        min_alpha: Minimum opacity for non-zero contributing words (default 0.18).

    Returns:
        Safe HTML string with inline CSS styling.
    """
    if not segments:
        return ""

    # Determine maximum contribution scale for proportional opacity
    non_zero_contribs = [abs(s["contribution"]) for s in segments if s["contribution"] != 0.0]
    max_val = max(non_zero_contribs) if non_zero_contribs else 1.0

    html_parts: List[str] = []

    for seg in segments:
        safe_text = html.escape(seg["text"])
        contrib = seg["contribution"]
        category = seg["category"]
        token = seg.get("token") or seg["text"]

        if category == "spam" and contrib > 0.0:
            scale = abs(contrib) / max_val
            alpha = min_alpha + (max_alpha - min_alpha) * scale
            title_text = f"{token} (+{contrib:.4f}): evidence supporting SPAM"
            span = (
                f'<span style="background-color: rgba(239, 68, 68, {alpha:.2f}); '
                f'padding: 2px 4px; border-radius: 4px; font-weight: 600;" '
                f'title="{html.escape(title_text)}">{safe_text}</span>'
            )
            html_parts.append(span)

        elif category == "ham" and contrib < 0.0:
            scale = abs(contrib) / max_val
            alpha = min_alpha + (max_alpha - min_alpha) * scale
            title_text = f"{token} ({contrib:.4f}): evidence supporting HAM"
            span = (
                f'<span style="background-color: rgba(34, 197, 94, {alpha:.2f}); '
                f'padding: 2px 4px; border-radius: 4px; font-weight: 600;" '
                f'title="{html.escape(title_text)}">{safe_text}</span>'
            )
            html_parts.append(span)

        else:
            # Neutral words and delimiters are rendered as plain escaped text
            html_parts.append(safe_text)

    return "".join(html_parts)


def generate_human_explanation(
    label: str,
    spam_probability: float,
    confidence: str,
    spam_words: List[Dict[str, Any]],
    ham_words: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> str:
    """Produce an objective, human-readable narrative explanation of the prediction.

    Adheres strictly to statistical interpretation without asserting proof or causation.

    Args:
        label: Predicted class label ('SPAM' or 'HAM').
        spam_probability: Predicted spam probability in [0, 1].
        confidence: Confidence level ('high', 'uncertain', 'low').
        spam_words: List of words supporting spam.
        ham_words: List of words supporting ham.
        threshold: Classification decision threshold used.

    Returns:
        Clear multi-sentence explanation string.
    """
    prob_pct = f"{spam_probability * 100:.2f}%"
    conf_str = confidence.upper()

    if not spam_words and not ham_words:
        return (
            "The message contains no recognized vocabulary terms from the training corpus. "
            "Without sufficient lexical evidence, the system safely defaulted to a "
            "HAM (legitimate) classification with LOW confidence."
        )

    explanation_lines: List[str] = []

    if label == "SPAM":
        explanation_lines.append(
            f"The model classified this message as SPAM with a posterior probability of {prob_pct} "
            f"(threshold: {threshold:.2f}, confidence: {conf_str})."
        )
    else:
        explanation_lines.append(
            f"The model classified this message as HAM (legitimate) with a spam probability of {prob_pct} "
            f"(threshold: {threshold:.2f}, confidence: {conf_str})."
        )

    if spam_words:
        top_spam = ", ".join([f"'{w['word']}' (+{w['contribution']:.4f})" for w in spam_words[:4]])
        explanation_lines.append(
            f"The primary features providing statistical evidence supporting a SPAM classification are: {top_spam}."
        )

    if ham_words:
        top_ham = ", ".join([f"'{w['word']}' ({w['contribution']:.4f})" for w in ham_words[:4]])
        explanation_lines.append(
            f"Features providing statistical evidence supporting a HAM classification are: {top_ham}."
        )

    if confidence == "uncertain":
        explanation_lines.append(
            "Note: The prediction falls in the borderline probability region [35%, 65%], "
            "indicating conflicting or mixed lexical indicators. Manual review or threshold tuning is recommended."
        )

    return " ".join(explanation_lines)


def explain_prediction(
    message: str,
    threshold: float = 0.5,
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "best_model",
) -> Dict[str, Any]:
    """Execute complete explainability pipeline on an SMS message.

    Extracts word contribution weights from the trained model and TF-IDF matrix,
    maps them to original raw message text, creates in-text highlighting segments,
    and produces an objective human-readable explanation.

    Args:
        message: Raw SMS message string.
        threshold: Decision threshold for spam classification (default 0.5).
        artifacts_dir: Directory containing model artifacts.
        model_name: Name of model artifact to use.

    Returns:
        Structured dictionary containing:
        - message: Original input text.
        - label: 'SPAM' or 'HAM'.
        - spam_probability: Float probability.
        - confidence: 'high', 'uncertain', or 'low'.
        - threshold: Threshold used.
        - word_explanations: List of unique words with contributions and frequencies.
        - spam_supporting_words: Words with contribution > 0, sorted descending.
        - ham_supporting_words: Words with contribution < 0, sorted ascending (strongest ham first).
        - neutral_words: Words with zero contribution or out of vocabulary.
        - text_segments: Structured tokens and delimiters for in-text rendering.
        - summary_text: Human-readable narrative explanation.
        - highlighted_html: Safe HTML string ready for UI rendering.
    """
    th = _validate_threshold(threshold)

    # 1. Base prediction
    pred = predict(message, threshold=th, artifacts_dir=artifacts_dir, model_name=model_name)
    label = pred["label"]
    spam_prob = pred["spam_probability"]
    confidence = pred["confidence"]

    # Handle empty or whitespace-only messages
    if not isinstance(message, str) or not message.strip():
        summary = (
            "Empty or whitespace-only message received. The system safely defaulted to a "
            "HAM classification with LOW confidence."
        )
        return {
            "message": message or "",
            "label": "HAM",
            "spam_probability": 0.0,
            "confidence": "low",
            "threshold": th,
            "word_explanations": [],
            "spam_supporting_words": [],
            "ham_supporting_words": [],
            "neutral_words": [],
            "text_segments": [],
            "summary_text": summary,
            "highlighted_html": "",
        }

    # 2. Extract model and vectorizer
    key = (Path(artifacts_dir), model_name)
    if key not in _predictors:
        _predictors[key] = Predictor(artifacts_dir=artifacts_dir, model_name=model_name)
    predictor = _predictors[key]

    assert predictor.vectorizer is not None
    assert predictor.model is not None

    safe_msg = message[:50000]
    tokens = preprocess_text(safe_msg)
    token_counts = Counter(tokens)
    known_tokens = [t for t in set(tokens) if t in predictor.vectorizer.vocabulary_]

    # Calculate token contribution dictionary
    token_contributions: Dict[str, float] = {}
    if known_tokens:
        X = predictor.vectorizer.transform([safe_msg])
        contrib_vec = predictor.model.get_word_contributions(X[0])
        for token in known_tokens:
            col_idx = predictor.vectorizer.vocabulary_[token]
            token_contributions[token] = float(contrib_vec[col_idx])

    # 3. Categorize words
    spam_supporting: List[Dict[str, Any]] = []
    ham_supporting: List[Dict[str, Any]] = []
    neutral_words: List[str] = []

    for token, count in token_counts.items():
        contrib = token_contributions.get(token, 0.0)
        item = {
            "word": token,
            "contribution": round(contrib, 4),
            "frequency": count,
            "category": "spam" if contrib > 0.0 else ("ham" if contrib < 0.0 else "neutral"),
        }
        if contrib > 0.0:
            spam_supporting.append(item)
        elif contrib < 0.0:
            ham_supporting.append(item)
        else:
            neutral_words.append(token)

    # Sort spam words descending (strongest evidence for spam first)
    spam_supporting.sort(key=lambda x: -x["contribution"])
    # Sort ham words ascending (most negative contribution first)
    ham_supporting.sort(key=lambda x: x["contribution"])

    all_contributing = spam_supporting + ham_supporting
    all_contributing.sort(key=lambda x: -abs(x["contribution"]))

    # 4. Text segmentation & safe HTML highlighting
    text_segments = build_text_segments(message, token_contributions)
    highlighted_html = render_highlighted_html(text_segments)

    # 5. Human-readable explanation text
    summary_text = generate_human_explanation(
        label=label,
        spam_probability=spam_prob,
        confidence=confidence,
        spam_words=spam_supporting,
        ham_words=ham_supporting,
        threshold=th,
    )

    return {
        "message": message,
        "label": label,
        "spam_probability": spam_prob,
        "confidence": confidence,
        "threshold": th,
        "word_explanations": all_contributing,
        "spam_supporting_words": spam_supporting,
        "ham_supporting_words": ham_supporting,
        "neutral_words": neutral_words,
        "text_segments": text_segments,
        "summary_text": summary_text,
        "highlighted_html": highlighted_html,
    }

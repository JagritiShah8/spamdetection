"""Text preprocessing pipeline for SMS spam detection.

Ensures consistent tokenization between training and inference.
"""

from __future__ import annotations

import re


def preprocess_text(text: str) -> list[str]:
    """Preprocess raw text string into cleaned tokens.

    Pipeline steps:
    1. Check for alphanumeric content.
    2. Normalize URLs and URL shorteners into canonical `url_link` and `url_shortener` tokens.
    3. Normalize currency indicators into `currency_symbol`.
    4. Normalize urgency punctuation into `urgent_punct`.
    5. Detect phone number formats into `phone_number`.
    6. Lowercase and replace non-alphanumeric characters with space (preserving underscores).
    7. Split into tokens and drop tokens with length < 2 characters.

    Args:
        text: Input string.

    Returns:
        List of preprocessed token strings.
    """
    if not isinstance(text, str):
        return []

    # Quick exit for empty or pure punctuation strings
    if not re.search(r"[a-zA-Z0-9]", text):
        return []

    cleaned = text.lower()

    # 1. Detect URL shorteners and general URLs
    shortener_pattern = r"(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly|adf\.ly|bitly\.com)"
    has_shortener = bool(re.search(shortener_pattern, cleaned))

    skip_url_tokens = {
        "http", "https", "www", "com", "org", "net", "edu", "gov", "io",
        "co", "uk", "in", "ly", "bit", "tinyurl", "goo", "gl", "is", "gd", "ow"
    }

    def url_replacer(match: re.Match[str]) -> str:
        url_str = match.group(0)
        path_words = [
            w for w in re.split(r"[^a-z0-9]", url_str)
            if len(w) >= 2 and w not in skip_url_tokens
        ]
        tokens = ["url_link"]
        if has_shortener:
            tokens.append("url_shortener")
        tokens.extend(path_words)
        return " " + " ".join(tokens) + " "

    url_pattern = r"https?://\S+|www\.\S+|(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|uk|in|ly|me|xyz|top|site|app|link)(?:/\S*)?"
    cleaned = re.sub(url_pattern, url_replacer, cleaned)

    # 2. Currency symbol normalization
    cleaned = re.sub(r"[£$€₹]", " currency_symbol ", cleaned)

    # 3. Urgency punctuation (e.g. multiple ! or !?)
    if re.search(r"!{2,}|\?!|!\?", cleaned):
        cleaned += " urgent_punct"

    # 4. Phone number indicators (e.g. UK mobile 07..., premium 08.../09..., toll-free 1-800..., international +...)
    # Excludes tracking numbers or identifiers preceded by #
    phone_pattern = r"(?<![#\w])(?:\b0[789]\d{8,10}\b|\+\d{10,13}\b|(?:\+\d{1,3}[-.\s]?)?\(?8\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b1[-.]?8\d{2}[-.]?\d{3}[-.]?\d{4}\b)"
    cleaned = re.sub(phone_pattern, r" phone_number \g<0> ", cleaned)

    # 5. Replace non-alphanumeric (allowing underscore for special tokens) with space
    cleaned = re.sub(r"[^a-z0-9_\s]", " ", cleaned)

    # 6. Collapse whitespace and split
    tokens = cleaned.split()

    # 7. Drop tokens < 2 chars
    return [token for token in tokens if len(token) >= 2]

"""Text preprocessing pipeline for SMS spam detection.

Ensures consistent tokenization between training and inference.
"""

from __future__ import annotations

import re


def preprocess_text(text: str) -> list[str]:
    """Preprocess raw text string into cleaned tokens.

    Pipeline steps:
    1. Lowercase text.
    2. Replace all non letter/digit/space characters with space.
    3. Collapse whitespace and strip leading/trailing spaces.
    4. Split into tokens.
    5. Drop tokens with length < 2 characters.

    Args:
        text: Input string.

    Returns:
        List of preprocessed token strings.
    """
    if not isinstance(text, str):
        return []

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Replace non letter/digit/space with space
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    # 3. Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    # 4. Split
    tokens = cleaned.split()

    # 5. Drop tokens < 2 chars
    return [token for token in tokens if len(token) >= 2]

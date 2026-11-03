"""Unit tests for text preprocessing."""

from spamdetector.preprocess import preprocess_text


def test_lowercase_conversion():
    tokens = preprocess_text("WINNER!! Congratulations on your PRIZE")
    assert "winner" in tokens
    assert "congratulations" in tokens
    assert "prize" in tokens
    assert "WINNER" not in tokens


def test_non_alphanumeric_replacement():
    tokens = preprocess_text("Call: +44(0)800-123456 now!!!")
    # All punctuation like +, (, ), -, : should be replaced with space
    assert "now" in tokens
    assert "123456" in tokens
    assert "+44" not in tokens


def test_collapse_whitespace():
    tokens = preprocess_text("Hello    \n\n\t  world!   Test   message.")
    assert tokens == ["hello", "world", "test", "message"]


def test_drop_short_tokens():
    # Tokens with length < 2 (like 'a', 'i', single digits) must be dropped
    tokens = preprocess_text("I am a 1 person who won a £1000 prize")
    assert "am" in tokens
    assert "person" in tokens
    assert "who" in tokens
    assert "won" in tokens
    assert "1000" in tokens
    assert "prize" in tokens
    assert "i" not in tokens
    assert "a" not in tokens
    assert "1" not in tokens


def test_empty_and_edge_inputs():
    assert preprocess_text("") == []
    assert preprocess_text("   \t\n  ") == []
    assert preprocess_text("!@#$%^&*()_+") == []
    assert preprocess_text(None) == []  # type: ignore
    assert preprocess_text(12345) == []  # type: ignore


def test_single_function_consistency():
    # Preprocessing is idempotent and produces deterministic tokens
    raw = "URGENT! You have won 1 week FREE membership! Call 09061743811"
    t1 = preprocess_text(raw)
    t2 = preprocess_text(raw)
    assert t1 == t2
    assert "urgent" in t1
    assert "membership" in t1

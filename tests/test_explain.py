"""Unit tests for Phase 3: Explainable AI (XAI) and In-Text Highlighting Engine."""

import pytest

from spamdetector.explain import (
    build_text_segments,
    explain_prediction,
    generate_human_explanation,
    render_highlighted_html,
)
from spamdetector.predict import predict


# 1. Spam Supporting Words Test
def test_spam_supporting_words():
    """Verify that spam-driving words receive positive contributions and correct categories."""
    msg = "URGENT! Call 09058094507 to claim your £2000 prize. Free entry valid now."
    res = explain_prediction(msg, model_name="logistic_regression")

    assert res["label"] == "SPAM"
    assert len(res["spam_supporting_words"]) > 0

    # All spam supporting words must have contribution > 0.0 and category 'spam'
    for item in res["spam_supporting_words"]:
        assert item["contribution"] > 0.0
        assert item["category"] == "spam"
        assert item["frequency"] >= 1

    # Contributions must be sorted descending
    contribs = [item["contribution"] for item in res["spam_supporting_words"]]
    assert contribs == sorted(contribs, reverse=True)

    # Key spam indicators must be identified
    spam_tokens = [item["word"] for item in res["spam_supporting_words"]]
    assert any(term in spam_tokens for term in ["call", "prize", "claim", "urgent", "free"])


# 2. Ham Supporting Words Test
def test_ham_supporting_words():
    """Verify that ham-driving words receive negative contributions and correct categories."""
    msg = "Can you please pick up some milk on your way home tonight? Thanks love."
    res = explain_prediction(msg, model_name="logistic_regression")

    assert res["label"] == "HAM"
    assert len(res["ham_supporting_words"]) > 0

    # All ham supporting words must have contribution < 0.0 and category 'ham'
    for item in res["ham_supporting_words"]:
        assert item["contribution"] < 0.0
        assert item["category"] == "ham"
        assert item["frequency"] >= 1

    # Contributions must be sorted ascending (most negative first)
    contribs = [item["contribution"] for item in res["ham_supporting_words"]]
    assert contribs == sorted(contribs)

    # Common ham features should be present
    ham_tokens = [item["word"] for item in res["ham_supporting_words"]]
    assert any(term in ham_tokens for term in ["milk", "tonight", "pick", "some"])


# 3. Repeated Words Test
def test_repeated_words():
    """Verify consistent contribution and frequency tracking for repeated words."""
    msg = "Free tickets, free entry, free cash! Call now, call today!"
    res = explain_prediction(msg, model_name="logistic_regression")

    word_map = {item["word"]: item for item in res["word_explanations"]}

    # 'free' occurs 3 times, 'call' occurs 2 times
    assert word_map["free"]["frequency"] == 3
    assert word_map["call"]["frequency"] == 2

    # Every occurrence of 'free' in text segments must have identical contribution and category
    free_segments = [s for s in res["text_segments"] if s.get("token") == "free"]
    assert len(free_segments) == 3
    for s in free_segments:
        assert s["contribution"] == word_map["free"]["contribution"]
        assert s["category"] == "spam"

    call_segments = [s for s in res["text_segments"] if s.get("token") == "call"]
    assert len(call_segments) == 2
    for s in call_segments:
        assert s["contribution"] == word_map["call"]["contribution"]
        assert s["category"] == "spam"


# 4. Punctuation and Capitalization Test
def test_punctuation_and_capitalization():
    """Verify that punctuation and mixed capitalization are preserved and segmented accurately."""
    msg = "!!!WINNER??? Free £1,000 CASH prize... CALL now!!!"
    res = explain_prediction(msg)

    segments = res["text_segments"]
    # Exact reconstruction
    reconstructed = "".join(s["text"] for s in segments)
    assert reconstructed == msg

    word_segs = {s["text"]: s for s in segments if s["is_word"]}
    # Uppercase in text, lowercase token
    assert word_segs["WINNER"]["token"] == "winner"
    assert word_segs["WINNER"]["category"] == "spam"

    assert word_segs["CASH"]["token"] == "cash"
    assert word_segs["CASH"]["category"] == "spam"

    assert word_segs["CALL"]["token"] == "call"
    assert word_segs["CALL"]["category"] == "spam"


# 5. Unknown Vocabulary Words Test
def test_unknown_vocabulary_words():
    """Verify that out-of-vocabulary words are categorized as neutral with zero contribution."""
    msg = "zxywqplm foobarbaz blablabla zzzqqq"
    res = explain_prediction(msg)

    assert res["label"] == "HAM"
    assert res["spam_probability"] == 0.0
    assert res["confidence"] == "low"
    assert len(res["spam_supporting_words"]) == 0
    assert len(res["ham_supporting_words"]) == 0
    assert len(res["neutral_words"]) == 4

    for s in res["text_segments"]:
        assert s["contribution"] == 0.0
        assert s["category"] == "neutral"

    assert "no recognized vocabulary terms" in res["summary_text"].lower()


# 6. Empty and Whitespace Messages Test
def test_empty_and_whitespace_messages():
    """Verify safe handling of empty strings and whitespace-only messages."""
    for empty_input in ["", "   ", "\n\t  \r"]:
        res = explain_prediction(empty_input)
        assert res["label"] == "HAM"
        assert res["spam_probability"] == 0.0
        assert res["confidence"] == "low"
        assert res["word_explanations"] == []
        assert res["highlighted_html"] == ""
        assert "empty or whitespace-only" in res["summary_text"].lower()


# 7. Correct Preservation of Original Text
def test_correct_preservation_of_original_text():
    """Verify character-for-character reconstruction across diverse message formats."""
    sample_messages = [
        "WINNER!! As a valued customer you have been selected to receive a £900 prize reward!",
        "Hey mate, are we still meeting for lunch at 12:30 or do you want to push it back?",
        "Special 50% discount at https://example.com/promo?id=123&ref=sms - Valid today!",
        "Hello...   Is anybody there??? 123 456 7890",
        "Multiple lines:\nLine 1: Ham\nLine 2: Spam alert!\nEnd.",
    ]

    for msg in sample_messages:
        res = explain_prediction(msg)
        reconstructed = "".join(s["text"] for s in res["text_segments"])
        assert reconstructed == msg, f"Failed to reconstruct: {msg!r}"


# 8. Consistency with Predict and Underlying Model
def test_consistency_with_predict_and_models():
    """Verify that explain_prediction output strictly matches predict() results."""
    msg = "WINNER!! Call 09061701461 to claim your £900 prize reward!"

    for th in [0.2, 0.5, 0.8]:
        res_exp = explain_prediction(msg, threshold=th)
        res_pred = predict(msg, threshold=th)

        assert res_exp["label"] == res_pred["label"]
        assert res_exp["spam_probability"] == res_pred["spam_probability"]
        assert res_exp["confidence"] == res_pred["confidence"]
        assert res_exp["threshold"] == th

    # Check top words match predict top_words exactly
    exp_top = [(w["word"], w["contribution"]) for w in res_exp["word_explanations"]]
    pred_top = res_pred["top_words"]
    # Check that highest magnitude words match
    for (ew, ec), (pw, pc) in zip(exp_top[:4], pred_top[:4]):
        assert ew == pw
        assert ec == pc


# 9. Safe Handling of HTML and Script Injection
def test_safe_handling_of_html_and_script_injection():
    """Verify that HTML tags, scripts, and attributes are safely escaped in rendered HTML."""
    malicious_inputs = [
        '<script>alert("XSS")</script>',
        '<img src="x" onerror="alert(\'pwned\')">',
        'WINNER <a href="http://malicious.com">Click here</a> to claim £500!',
        '<b>Bold</b> & <i>Italic</i> & "Quotes" & \'Single\'',
    ]

    for malicious_msg in malicious_inputs:
        res = explain_prediction(malicious_msg)
        rendered_html = res["highlighted_html"]

        # Raw tags MUST NOT exist in rendered output
        assert "<script>" not in rendered_html
        assert "</script>" not in rendered_html
        assert "<img" not in rendered_html
        assert "<a href=" not in rendered_html

        # Escaped versions MUST be present
        if "<script>" in malicious_msg:
            assert "&lt;script&gt;" in rendered_html
        if "<img" in malicious_msg:
            assert "&lt;img" in rendered_html
        if "<a href=" in malicious_msg:
            assert "&lt;a href=" in rendered_html


# 10. Human-Readable Explanation Integrity
def test_human_readable_explanation_content():
    """Verify that narrative explanation is objective, accurate, and non-causal."""
    msg = "WINNER!! You have been selected for a £1000 prize. Call 0800 now to claim."
    res = explain_prediction(msg)

    summary = res["summary_text"]
    assert res["label"] in summary
    assert "confidence" in summary.lower()
    assert "evidence supporting" in summary.lower()

    # Mentioned words must be actual contributing features
    for item in res["spam_supporting_words"][:3]:
        assert item["word"] in summary

"""Tests for prediction module and edge cases."""

from spamdetector.predict import predict


def test_predict_spam_message():
    msg = "WINNER! You have won a £1000 cash prize. Call 0800 now to claim."
    res = predict(msg)
    assert res["label"] == "SPAM"
    assert res["spam_probability"] >= 0.5
    assert len(res["top_words"]) > 0
    words = [w for w, _ in res["top_words"]]
    assert any(term in words for term in ["winner", "prize", "claim", "1000", "cash", "call", "won"])


def test_predict_ham_message():
    msg = "Hey are we still meeting for dinner tonight at 7?"
    res = predict(msg)
    assert res["label"] == "HAM"
    assert res["spam_probability"] < 0.5


def test_predict_empty_and_whitespace_edge_cases():
    res_empty = predict("")
    assert res_empty["label"] == "HAM"
    assert res_empty["spam_probability"] == 0.0
    assert res_empty["confidence"] == "low"
    assert res_empty["top_words"] == []

    res_spaces = predict("   \n\t   ")
    assert res_spaces["label"] == "HAM"
    assert res_spaces["spam_probability"] == 0.0
    assert res_spaces["confidence"] == "low"
    assert res_spaces["top_words"] == []


def test_predict_unseen_words_fallback():
    res_unseen = predict("zxywqplm qwertyuiop asdfghjkl zxcvbnm")
    assert res_unseen["label"] == "HAM"
    assert res_unseen["spam_probability"] == 0.0
    assert res_unseen["confidence"] == "low"
    assert res_unseen["top_words"] == []


def test_predict_very_long_input():
    long_msg = "Hello friend! " * 5000 + " Call now to claim free prize"
    res = predict(long_msg)
    assert res["label"] in ["SPAM", "HAM"]
    assert 0.0 <= res["spam_probability"] <= 1.0
    assert res["confidence"] in ["high", "uncertain", "low"]


def test_reproducibility():
    msg = "Urgent service alert. Please reply to verify your account"
    res1 = predict(msg)
    res2 = predict(msg)
    assert res1 == res2

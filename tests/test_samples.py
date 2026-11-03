"""Unit test running the 4 sample messages through the saved model and LR."""

import pytest
from spamdetector.predict import predict

SAMPLE_MESSAGES = [
    # Sample 1: Clear lottery/prize spam
    "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim. Valid 12 hours only.",
    # Sample 2: Urgent mobile award spam
    "URGENT! Your Mobile number has been awarded with a £2000 prize. Call 09058094507 from landline to claim. T&Cs apply.",
    # Sample 3: Casual conversational ham
    "Hey mate, are we still meeting for lunch at 12:30 or do you want to push it back?",
    # Sample 4: Promotional style legitimate ham
    "Hey, we are having a 20% discount celebration at our bakery today for friends & family! Hope you can make it, let me know.",
]


@pytest.mark.parametrize("msg", SAMPLE_MESSAGES)
def test_sample_messages_saved_model(msg):
    res = predict(msg, model_name="best_model")
    assert res["label"] in ["SPAM", "HAM"]
    assert 0.0 <= res["spam_probability"] <= 1.0
    assert res["confidence"] in ["high", "uncertain", "low"]
    assert isinstance(res["top_words"], list)


@pytest.mark.parametrize("msg", SAMPLE_MESSAGES)
def test_sample_messages_logistic_regression(msg):
    res = predict(msg, model_name="logistic_regression")
    assert res["label"] in ["SPAM", "HAM"]
    assert 0.0 <= res["spam_probability"] <= 1.0
    assert res["confidence"] in ["high", "uncertain", "low"]
    assert isinstance(res["top_words"], list)

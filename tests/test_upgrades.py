"""Unit tests for Phase 1 upgrades: configurable threshold and batch inference."""

import pytest

from spamdetector.predict import predict, predict_batch


def test_default_threshold_behavior():
    """Verify that omitting threshold defaults to 0.5 and matches threshold=0.5 exactly."""
    msg = "WINNER!! As a valued customer you have been selected to receive a £900 prize reward!"
    res_default = predict(msg)
    res_explicit = predict(msg, threshold=0.5)

    assert res_default == res_explicit
    assert res_default["label"] in ["SPAM", "HAM"]
    assert 0.0 <= res_default["spam_probability"] <= 1.0

    batch_default = predict_batch([msg])
    batch_explicit = predict_batch([msg], threshold=0.5)

    assert batch_default == batch_explicit == [res_default]


def test_thresholds_modulation_02_05_08():
    """Verify that thresholds 0.2, 0.5, and 0.8 alter classification boundary appropriately."""
    # Message with probability in [0.2, 0.5] (~0.299)
    borderline_msg = "Reminder: your subscription will renew tomorrow unless cancelled."

    res_02 = predict(borderline_msg, threshold=0.2)
    res_05 = predict(borderline_msg, threshold=0.5)
    res_08 = predict(borderline_msg, threshold=0.8)

    # Probabilities and top words must remain identical; only threshold-dependent label changes
    assert res_02["spam_probability"] == res_05["spam_probability"] == res_08["spam_probability"]
    assert res_02["top_words"] == res_05["top_words"] == res_08["top_words"]

    prob = res_05["spam_probability"]

    # Decision boundary checks
    expected_02 = "SPAM" if prob >= 0.2 else "HAM"
    expected_05 = "SPAM" if prob >= 0.5 else "HAM"
    expected_08 = "SPAM" if prob >= 0.8 else "HAM"

    assert res_02["label"] == expected_02
    assert res_05["label"] == expected_05
    assert res_08["label"] == expected_08

    # Borderline message flips between threshold 0.2 and 0.5
    assert res_02["label"] == "SPAM"
    assert res_05["label"] == "HAM"
    assert res_08["label"] == "HAM"

    # Moderate-high spam message (~0.673) flips between threshold 0.5 and 0.8
    spam_msg = "Urgent notice regarding your recent order delivery status update."
    spam_res_02 = predict(spam_msg, threshold=0.2)
    spam_res_05 = predict(spam_msg, threshold=0.5)
    spam_res_08 = predict(spam_msg, threshold=0.8)

    assert spam_res_02["label"] == "SPAM"
    assert spam_res_05["label"] == "SPAM"
    assert spam_res_08["label"] == "HAM"


def test_invalid_thresholds():
    """Verify that invalid threshold values raise ValueError."""
    invalid_values = [-0.1, -5.0, 1.001, 2.5, "0.5", "invalid", None, True, False]

    for val in invalid_values:
        with pytest.raises(ValueError):
            predict("Test message", threshold=val)

        with pytest.raises(ValueError):
            predict_batch(["Test message"], threshold=val)


def test_empty_batch_input():
    """Verify that empty batch inputs return an empty list."""
    assert predict_batch([]) == []
    assert predict_batch(()) == []


def test_batch_matches_individual_predictions():
    """Verify that batch predictions match individual predictions exactly."""
    messages = [
        "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim.",
        "URGENT! Your Mobile number has been awarded with a £2000 prize. Call 09058094507 from landline to claim. T&Cs apply.",
        "Hey mate, are we still meeting for lunch at 12:30 or do you want to push it back?",
        "Hey, we are having a 20% discount celebration at our bakery today for friends & family! Hope you can make it, let me know.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
        "Can you pick up some milk on your way home please?",
    ]

    for th in [0.2, 0.5, 0.8]:
        batch_res = predict_batch(messages, threshold=th)
        indiv_res = [predict(m, threshold=th) for m in messages]

        assert len(batch_res) == len(messages)
        assert batch_res == indiv_res

    # Also verify with logistic_regression model
    batch_lr = predict_batch(messages, threshold=0.5, model_name="logistic_regression")
    indiv_lr = [predict(m, threshold=0.5, model_name="logistic_regression") for m in messages]
    assert batch_lr == indiv_lr


def test_empty_and_unseen_messages():
    """Verify batch handling of edge cases (empty, whitespace, unseen vocabulary)."""
    edge_cases = [
        "",
        "   ",
        "\n\t  ",
        "zxywqplm qwertyuiop asdfghjkl",
        "foobarbaz blablabla zzzqqq",
    ]

    batch_res = predict_batch(edge_cases, threshold=0.5)
    indiv_res = [predict(m, threshold=0.5) for m in edge_cases]

    assert len(batch_res) == len(edge_cases)
    assert batch_res == indiv_res

    for item in batch_res:
        assert item["label"] == "HAM"
        assert item["spam_probability"] == 0.0
        assert item["confidence"] == "low"
        assert item["top_words"] == []


def test_mixed_batch_with_edge_cases_and_valid_messages():
    """Verify batch containing interleaved edge cases and valid messages."""
    mixed_messages = [
        "",
        "WINNER!! Call 09061701461 now to claim prize",
        "   \t\n   ",
        "Hey mate, are we meeting for lunch today?",
        "zxywqplm qwertyuiop asdfghjkl",
        "URGENT! Mobile award 2000 pounds call now",
    ]

    for th in [0.2, 0.5, 0.8]:
        batch_res = predict_batch(mixed_messages, threshold=th)
        indiv_res = [predict(m, threshold=th) for m in mixed_messages]
        assert batch_res == indiv_res


def test_predict_batch_rejects_single_string():
    """Verify predict_batch raises TypeError if passed a single string instead of a sequence."""
    with pytest.raises(TypeError, match="expects a sequence of message strings"):
        predict_batch("Single string message")

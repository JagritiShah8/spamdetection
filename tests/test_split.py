"""Unit test verifying deterministic stratified split sizes."""

from spamdetector.dataset import load_sms_dataset, stratified_split


def test_stratified_split_sizes():
    df = load_sms_dataset("data/SMSSpamCollection")
    train_idx, test_idx = stratified_split(df["label"].values, test_ratio=0.2, random_state=42)

    # Check total sizes
    assert len(train_idx) == 4458, f"Expected 4458 train, got {len(train_idx)}"
    assert len(test_idx) == 1114, f"Expected 1114 test, got {len(test_idx)}"

    # Check class counts in test set
    test_labels = df["label"].iloc[test_idx]
    ham_test_count = (test_labels == "ham").sum()
    spam_test_count = (test_labels == "spam").sum()

    assert ham_test_count == 965, f"Expected 965 ham in test, got {ham_test_count}"
    assert spam_test_count == 149, f"Expected 149 spam in test, got {spam_test_count}"

    # Check class counts in train set
    train_labels = df["label"].iloc[train_idx]
    ham_train_count = (train_labels == "ham").sum()
    spam_train_count = (train_labels == "spam").sum()

    assert ham_train_count == 3860, f"Expected 3860 ham in train, got {ham_train_count}"
    assert spam_train_count == 598, f"Expected 598 spam in train, got {spam_train_count}"


def test_stratified_split_mixed_case():
    import numpy as np
    labels = np.array(["ham", "SPAM", "Ham", "spam", "HAM", "Spam", " ham ", " spam "])
    train_idx, test_idx = stratified_split(labels, test_ratio=0.5, random_state=42)
    assert len(train_idx) == 4
    assert len(test_idx) == 4
    # Ensure indices are valid and disjoint
    assert set(train_idx).isdisjoint(set(test_idx))
    assert len(set(train_idx) | set(test_idx)) == 8


def test_load_augmented_training_data():
    from spamdetector.dataset import load_augmented_training_data
    import numpy as np
    train_texts = ["hello there", "free cash prize"]
    train_labels = np.array([0, 1])
    aug_texts, aug_labels = load_augmented_training_data(train_texts, train_labels)
    import pandas as pd
    aug_df = pd.read_csv("data/smishing_augmentation.tsv", sep="\t")
    n_aug = len(aug_df)
    assert len(aug_texts) == len(train_texts) + n_aug
    assert len(aug_labels) == len(train_labels) + n_aug

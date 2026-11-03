"""Dataset Explorer & Exploratory Data Analysis (EDA) Tab for Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spamdetector.dataset import load_sms_dataset
from spamdetector.eda import (
    compute_class_distribution,
    compute_full_dataset_summary,
    compute_length_statistics,
    compute_lexical_diversity,
    compute_vocabulary_overlap,
    compute_word_frequencies,
)


@st.cache_data
def get_cached_dataset_eda() -> Dict[str, Any]:
    """Load dataset and execute complete EDA pipeline with caching."""
    df = load_sms_dataset("data/SMSSpamCollection")
    return compute_full_dataset_summary(df)


def render_explorer_tab() -> None:
    """Render comprehensive Dataset Analytics & EDA tab."""
    st.markdown("### 📊 Dataset Explorer & Exploratory Data Analysis (EDA)")
    st.markdown(
        "Empirical investigation of the official **UCI SMS Spam Collection** (5,572 messages). "
        "All metrics and statistical distributions are dynamically computed using NumPy and Pandas."
    )

    with st.spinner("Analyzing dataset distributions..."):
        summary = get_cached_dataset_eda()

    dist = summary["class_distribution"]
    len_stats = summary["length_statistics"]
    freq = summary["word_frequencies"]
    overlap = summary["vocabulary_overlap"]
    lexical = summary["lexical_diversity"]

    # 1. Overview KPIs
    st.markdown("#### 1. Dataset Dimensions & Class Imbalance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Total Corpus</div>"
            f"<div class='metric-value'>{dist['total_messages']:,}</div>"
            f"<div class='metric-sub'>SMS Messages</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Ham (Legitimate)</div>"
            f"<div class='metric-value' style='color: #34d399;'>{dist['ham_count']:,}</div>"
            f"<div class='metric-sub'>{dist['ham_percentage']:.2f}% of corpus</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Spam Messages</div>"
            f"<div class='metric-value' style='color: #fb7185;'>{dist['spam_count']:,}</div>"
            f"<div class='metric-sub'>{dist['spam_percentage']:.2f}% of corpus</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Class Imbalance</div>"
            f"<div class='metric-value' style='color: #fbbf24;'>{dist['imbalance_ratio']:.2f} : 1</div>"
            f"<div class='metric-sub'>Ham-to-Spam ratio</div></div>",
            unsafe_allow_html=True,
        )

    st.info(
        f"**Machine Learning Context**: The SMS Spam Collection exhibits an imbalance ratio of "
        f"**{dist['imbalance_ratio']:.2f} to 1**. A naive baseline predicting only the majority class "
        f"would achieve an accuracy of **86.59%** while catching exactly **0% of spam**. "
        f"This justifies using inverse-frequency class weights in Logistic Regression and evaluating on F1-score rather than accuracy alone."
    )

    # 2. Length Dynamics
    st.markdown("---")
    st.markdown("#### 2. Message Length Analysis: Ham vs. Spam")
    st.markdown(
        "Comparison of character counts and token lengths across classes. "
        "In SMS communications, spam messages historically maximized character counts up to the standard 160-character network payload."
    )

    char_ham = len_stats["char_length"]["ham"]
    char_spam = len_stats["char_length"]["spam"]
    tok_ham = len_stats["token_length"]["ham"]
    tok_spam = len_stats["token_length"]["spam"]

    tbl_col, chart_col = st.columns([1.2, 1.8])

    with tbl_col:
        length_df = pd.DataFrame({
            "Metric": ["Mean Characters", "Median Characters", "Std Characters", "Mean Tokens", "Median Tokens", "IQR (Tokens)"],
            "Ham (Legitimate)": [f"{char_ham['mean']:.1f}", f"{char_ham['median']:.0f}", f"{char_ham['std']:.1f}", f"{tok_ham['mean']:.1f}", f"{tok_ham['median']:.0f}", f"{tok_ham['iqr']:.1f}"],
            "Spam Messages": [f"{char_spam['mean']:.1f}", f"{char_spam['median']:.0f}", f"{char_spam['std']:.1f}", f"{tok_spam['mean']:.1f}", f"{tok_spam['median']:.0f}", f"{tok_spam['iqr']:.1f}"],
        })
        st.dataframe(length_df, hide_index=True, use_container_width=True)

    with chart_col:
        # Side by side bar chart comparing mean and median lengths
        comp_df = pd.DataFrame({
            "Ham": [char_ham["mean"], char_ham["median"], tok_ham["mean"] * 5, tok_ham["median"] * 5],
            "Spam": [char_spam["mean"], char_spam["median"], tok_spam["mean"] * 5, tok_spam["median"] * 5],
        }, index=["Mean Chars", "Median Chars", "Mean Tokens (x5)", "Median Tokens (x5)"])
        st.bar_chart(comp_df, color=["#10b981", "#f43f5e"])
        st.caption("Bar chart comparing character and scaled token dimensions (Ham in green, Spam in red).")

    # 3. Top Words in Ham vs Spam
    st.markdown("---")
    st.markdown("#### 3. Top Vocabulary Terms & Discriminative Tokens")

    w_col1, w_col2 = st.columns(2)

    with w_col1:
        st.markdown(f"**Top 15 Most Frequent SPAM Words:**")
        spam_words_df = pd.DataFrame(
            freq["top_spam_words"][:15],
            columns=["Word", "Occurrence Count"],
        ).set_index("Word")
        st.bar_chart(spam_words_df, color="#f43f5e")

    with w_col2:
        st.markdown(f"**Top 15 Most Frequent HAM Words:**")
        ham_words_df = pd.DataFrame(
            freq["top_ham_words"][:15],
            columns=["Word", "Occurrence Count"],
        ).set_index("Word")
        st.bar_chart(ham_words_df, color="#10b981")

    # 4. Vocabulary Overlap & Exclusivity
    st.markdown("---")
    st.markdown("#### 4. Vocabulary Overlap & Lexical Exclusivity")

    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Total Vocabulary</div>"
            f"<div class='metric-value'>{overlap['total_unique_words']:,}</div>"
            f"<div class='metric-sub'>Distinct terms</div></div>",
            unsafe_allow_html=True,
        )
    with ov2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Shared Vocabulary</div>"
            f"<div class='metric-value'>{overlap['shared_vocab_size']:,}</div>"
            f"<div class='metric-sub'>Ham ∩ Spam</div></div>",
            unsafe_allow_html=True,
        )
    with ov3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Jaccard Similarity</div>"
            f"<div class='metric-value'>{overlap['jaccard_similarity'] * 100:.2f}%</div>"
            f"<div class='metric-sub'>Lexicon overlap</div></div>",
            unsafe_allow_html=True,
        )
    with ov4:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Corpus Diversity</div>"
            f"<div class='metric-value'>{lexical['corpus_ttr']['all'] * 100:.1f}%</div>"
            f"<div class='metric-sub'>Type-Token Ratio</div></div>",
            unsafe_allow_html=True,
        )

    # Class-exclusive words preview
    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        st.markdown(f"**Top Exclusive SPAM Words** (Appear in spam, never in ham):")
        spam_only_chips = [
            f"<span class='chip-spam'>{w} ({c})</span>"
            for w, c in overlap["top_spam_only_words"][:14]
        ]
        st.markdown(" ".join(spam_only_chips), unsafe_allow_html=True)

    with ex_c2:
        st.markdown(f"**Top Exclusive HAM Words** (Appear in ham, never in spam):")
        ham_only_chips = [
            f"<span class='chip-ham'>{w} ({c})</span>"
            for w, c in overlap["top_ham_only_words"][:14]
        ]
        st.markdown(" ".join(ham_only_chips), unsafe_allow_html=True)

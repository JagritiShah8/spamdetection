"""Live SMS Detector Tab for Streamlit UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import streamlit as st

from spamdetector.explain import explain_prediction


SAMPLE_MESSAGES = {
    "Spam: Lottery Prize": (
        "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! "
        "Call 09061701461 to claim. Valid 12 hours only."
    ),
    "Spam: Urgent Bank Alert": (
        "URGENT! Your Mobile number has been awarded with a £2000 prize. "
        "Call 09058094507 from landline to claim. T&Cs apply."
    ),
    "Ham: Casual Meeting": (
        "Hey mate, are we still meeting for lunch at 12:30 or do you want to push it back?"
    ),
    "Ham: Bakery Promotion": (
        "Hey, we are having a 20% discount celebration at our bakery today for friends & family! "
        "Hope you can make it, let me know."
    ),
}


def render_detector_tab() -> None:
    """Render interactive Live SMS Detection tab."""
    st.markdown("### 🎯 Live SMS Classification & Explainable AI")
    st.markdown(
        "Enter any SMS message text or select an interactive example below. "
        "Adjust the classification threshold to observe how the decision boundary changes."
    )

    # 1. Interactive Example Buttons
    st.markdown("**Quick Test Examples:**")
    cols = st.columns(4)
    for i, (title, sample_text) in enumerate(SAMPLE_MESSAGES.items()):
        with cols[i]:
            if st.button(title, key=f"ex_btn_{i}", use_container_width=True):
                st.session_state["sms_input"] = sample_text

    # Default message text
    default_text = st.session_state.get(
        "sms_input",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    )

    # 2. Text Area & Threshold Control
    col_input, col_ctrl = st.columns([3, 1])

    with col_input:
        user_message = st.text_area(
            "Enter SMS Message Text:",
            value=default_text,
            height=130,
            placeholder="Paste or type an SMS message to classify...",
            key="detector_text_area",
        )

    with col_ctrl:
        st.markdown("**Decision Threshold ($T$):**")
        threshold = st.slider(
            "Spam Cutoff Threshold",
            min_value=0.10,
            max_value=0.90,
            value=st.session_state.get("active_threshold", 0.50),
            step=0.05,
            help="Probability cutoff. Messages with spam probability >= threshold are labeled SPAM. Lower threshold catches more spam; higher threshold protects against false positives.",
            label_visibility="collapsed",
        )
        st.session_state["active_threshold"] = threshold

        if threshold < 0.40:
            st.caption("🛡️ **Aggressive**: Prioritizes zero spam leakage (higher FP risk).")
        elif threshold > 0.60:
            st.caption("🔒 **Conservative**: Strictly prevents blocking legitimate SMS/OTPs.")
        else:
            st.caption("⚖️ **Balanced**: Standard 0.50 Bayesian decision threshold.")

    # Check button
    btn_col1, btn_col2, _ = st.columns([1.5, 2, 3])
    with btn_col1:
        check_clicked = st.button("🔍 Classify Message", type="primary", use_container_width=True)

    # Execute Prediction
    if check_clicked or user_message:
        try:
            explanation = explain_prediction(user_message, threshold=threshold)
            label = explanation["label"]
            prob = explanation["spam_probability"]
            confidence = explanation["confidence"]
            spam_words = explanation["spam_supporting_words"]
            ham_words = explanation["ham_supporting_words"]
            summary_text = explanation["summary_text"]
            highlighted_html = explanation["highlighted_html"]

            st.markdown("---")
            st.markdown("#### 📊 Prediction & Confidence Analysis")

            res_c1, res_c2, res_c3 = st.columns([1.2, 1.2, 1.6])

            with res_c1:
                if confidence == "uncertain":
                    badge_html = "<span class='badge-uncertain'>⚠️ UNCERTAIN</span>"
                elif label == "SPAM":
                    badge_html = "<span class='badge-spam'>🚨 SPAM</span>"
                else:
                    badge_html = "<span class='badge-ham'>✅ HAM (LEGITIMATE)</span>"

                st.markdown(f"<div style='margin-bottom: 8px;'>{badge_html}</div>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Level**: `{confidence.upper()}`")

            with res_c2:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-title'>Spam Probability</div>"
                    f"<div class='metric-value'>{prob * 100:.2f}%</div>"
                    f"<div class='metric-sub'>Threshold: {threshold:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with res_c3:
                st.markdown("**Probability Meter:**")
                st.progress(prob)
                st.caption(f"0% (Legitimate) ─── Cutoff: {threshold:.2f} ─── 100% (Spam)")

            # In-Text Highlighting
            st.markdown("#### 🔍 Explainable AI (In-Text Feature Highlights)")
            st.markdown(
                "Words highlighted in <span class='chip-spam'>Red</span> represent statistical evidence pushing towards **SPAM**. "
                "Words highlighted in <span class='chip-ham'>Green</span> represent statistical evidence pulling towards **HAM**. "
                "Hover over any highlighted word to inspect its exact contribution weight.",
                unsafe_allow_html=True,
            )

            if highlighted_html:
                st.markdown(f"<div class='highlight-container'>{highlighted_html}</div>", unsafe_allow_html=True)
            else:
                st.info("No text or recognizable vocabulary detected to highlight.")

            # Word Breakdown Columns
            col_sp, col_hm = st.columns(2)
            with col_sp:
                st.markdown(f"**Evidence Supporting SPAM (+)** ({len(spam_words)} words):")
                if spam_words:
                    chips = [
                        f"<span class='chip-spam'>{w['word']} (+{w['contribution']:.4f})</span>"
                        for w in spam_words[:10]
                    ]
                    st.markdown(" ".join(chips), unsafe_allow_html=True)
                else:
                    st.caption("No significant spam-supporting indicators detected.")

            with col_hm:
                st.markdown(f"**Evidence Supporting HAM (-)** ({len(ham_words)} words):")
                if ham_words:
                    chips = [
                        f"<span class='chip-ham'>{w['word']} ({w['contribution']:.4f})</span>"
                        for w in ham_words[:10]
                    ]
                    st.markdown(" ".join(chips), unsafe_allow_html=True)
                else:
                    st.caption("No significant ham-supporting indicators detected.")

            # Human-Readable Reasoning
            st.markdown("#### 🧠 Model Reasoning & Justification")
            st.info(summary_text)

            # Save to Session History Button
            with btn_col2:
                if st.button("💾 Save to Session History", key="save_hist_btn", use_container_width=True):
                    if "prediction_history" not in st.session_state:
                        st.session_state["prediction_history"] = []

                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "message": user_message,
                        "label": label,
                        "spam_probability": prob,
                        "threshold": threshold,
                        "confidence": confidence,
                    }
                    st.session_state["prediction_history"].append(record)
                    st.toast("Saved to prediction audit history!", icon="✅")

        except Exception as exc:
            st.error(f"Inference error: {exc}")

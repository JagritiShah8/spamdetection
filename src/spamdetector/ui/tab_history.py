"""Session Prediction History & Audit Log Tab for Streamlit UI."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def render_history_tab() -> None:
    """Render Session-level Prediction History and Audit Log."""
    st.markdown("### 📜 Session Prediction History & Audit Trail")
    st.markdown(
        "Audit log of all SMS messages evaluated during your current browser session. "
        "Allows real-time inspection, label filtering, and CSV export."
    )

    history: List[Dict[str, Any]] = st.session_state.get("prediction_history", [])

    if not history:
        st.info(
            "No predictions saved in session history yet. "
            "Go to the **Live SMS Detector** tab, classify any message, and click **Save to Session History**."
        )
        return

    # 1. Summary KPIs
    df_hist = pd.DataFrame(history)
    total_evals = len(df_hist)
    spam_count = int((df_hist["label"] == "SPAM").sum())
    ham_count = int((df_hist["label"] == "HAM").sum())
    avg_prob = float(df_hist["spam_probability"].mean()) if total_evals > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Session Evaluated</div>"
            f"<div class='metric-value'>{total_evals}</div><div class='metric-sub'>Messages</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Spam Flagged</div>"
            f"<div class='metric-value' style='color: #fb7185;'>{spam_count}</div>"
            f"<div class='metric-sub'>{spam_count/total_evals*100:.1f}% of session</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Legitimate (Ham)</div>"
            f"<div class='metric-value' style='color: #34d399;'>{ham_count}</div>"
            f"<div class='metric-sub'>{ham_count/total_evals*100:.1f}% of session</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Average Probability</div>"
            f"<div class='metric-value' style='color: #38bdf8;'>{avg_prob*100:.1f}%</div>"
            f"<div class='metric-sub'>Mean posterior prob</div></div>",
            unsafe_allow_html=True,
        )

    # 2. Filter & Display Table
    st.markdown("---")
    st.markdown("#### 📋 Prediction Log")

    col_fil, col_dl, col_clr = st.columns([2, 1.5, 1.5])

    with col_fil:
        filter_choice = st.radio(
            "Filter History by Label:",
            options=["All", "SPAM Only", "HAM Only"],
            horizontal=True,
            key="hist_filter_radio",
        )

    display_df = df_hist
    if filter_choice == "SPAM Only":
        display_df = df_hist[df_hist["label"] == "SPAM"]
    elif filter_choice == "HAM Only":
        display_df = df_hist[df_hist["label"] == "HAM"]

    # Format probability as percentage string
    formatted_display = display_df.copy()
    formatted_display["spam_probability"] = formatted_display["spam_probability"].apply(lambda p: f"{p*100:.2f}%")
    formatted_display["threshold"] = formatted_display["threshold"].apply(lambda t: f"{t:.2f}")

    st.dataframe(
        formatted_display[["timestamp", "label", "spam_probability", "threshold", "confidence", "message"]],
        hide_index=True,
        use_container_width=True,
        height=320,
    )

    # 3. Actions: Download & Clear History
    with col_dl:
        csv_buffer = io.StringIO()
        df_hist.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Download Audit Log (CSV)",
            data=csv_buffer.getvalue(),
            file_name="spam_detection_session_audit_log.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    with col_clr:
        confirm_clear = st.checkbox("Confirm Clear History", key="confirm_clear_chk")
        if confirm_clear:
            if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.rerun()

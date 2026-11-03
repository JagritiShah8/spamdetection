"""Bulk SMS Classification Tab for Streamlit UI."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from spamdetector.predict import predict_batch


SAMPLE_CSV_DATA = """id,message
1,WINNER! You have won a £1000 cash prize. Call 0800 now to claim your reward.
2,Hey are we still meeting for lunch today at 12:30 or push it back?
3,URGENT! Your Mobile number has been awarded with a £2000 prize. Call 09058094507 to claim.
4,Hey, we are having a 20% discount celebration at our bakery today for friends & family!
5,Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121.
6,Can you please pick up some groceries on your way back from work?
7,Congratulations! You have been selected for a free trial. Reply YES to activate.
8,What time does the movie start tonight? Let me know when you leave.
9,
10,zxywqplm qwertyuiop asdfghjkl randomunseenword
"""


def render_bulk_tab() -> None:
    """Render Bulk SMS Batch Classification tab."""
    st.markdown("### 📂 Bulk SMS Classification via CSV Upload")
    st.markdown(
        "Upload a CSV file containing SMS text messages for automated batch processing. "
        "The engine uses the vectorized `predict_batch` pipeline for high-throughput inference."
    )

    # 1. Download Sample CSV
    col_d1, col_d2 = st.columns([2, 1])
    with col_d2:
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=SAMPLE_CSV_DATA,
            file_name="sample_sms_messages.csv",
            mime="text/csv",
            help="Download a ready-to-test CSV with sample spam, ham, and edge-case messages.",
            use_container_width=True,
        )

    # 2. File Uploader
    uploaded_file = st.file_uploader(
        "Upload SMS Dataset (.CSV):",
        type=["csv"],
        help="Upload a CSV file with at least one text column.",
    )

    # Threshold slider for bulk classification
    bulk_threshold = st.slider(
        "Classification Threshold for Batch Processing",
        min_value=0.10,
        max_value=0.90,
        value=st.session_state.get("active_threshold", 0.50),
        step=0.05,
        key="bulk_threshold_slider",
    )

    if uploaded_file is not None:
        try:
            # Try UTF-8 first, fallback to Latin-1
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")

            if df.empty:
                st.warning("Uploaded CSV file contains no data rows.")
                return

            st.success(f"Successfully loaded CSV with {len(df):,} rows and {len(df.columns)} columns.")

            # Column Selection
            candidate_cols = list(df.columns)
            default_col_idx = 0
            for idx, col_name in enumerate(candidate_cols):
                if any(k in str(col_name).lower() for k in ["text", "sms", "message", "body"]):
                    default_col_idx = idx
                    break

            selected_col = st.selectbox(
                "Select the SMS Message Column:",
                options=candidate_cols,
                index=default_col_idx,
            )

            # Process Batch Button
            if st.button("🚀 Run Batch Classification", type="primary", use_container_width=False):
                with st.spinner("Executing vectorized batch classification..."):
                    progress_bar = st.progress(0.0)

                    # Extract raw texts, safely treating NaN as empty string
                    messages = df[selected_col].fillna("").astype(str).tolist()
                    n_msgs = len(messages)

                    # Batch prediction using official predict_batch
                    progress_bar.progress(0.4)
                    results = predict_batch(messages, threshold=bulk_threshold)
                    progress_bar.progress(0.9)

                    # Enrich DataFrame
                    pred_labels = [r["label"] for r in results]
                    pred_probs = [r["spam_probability"] for r in results]
                    pred_conf = [r["confidence"].upper() for r in results]

                    df_results = df.copy()
                    df_results["predicted_label"] = pred_labels
                    df_results["spam_probability"] = pred_probs
                    df_results["confidence"] = pred_conf

                    progress_bar.progress(1.0)
                    st.session_state["bulk_classified_df"] = df_results
                    st.success(f"Batch processing completed for all {n_msgs:,} messages!")

        except Exception as exc:
            st.error(f"Error reading or processing CSV: {exc}")

    # Display Batch Results if available in session
    if "bulk_classified_df" in st.session_state:
        df_res: pd.DataFrame = st.session_state["bulk_classified_df"]
        st.markdown("---")
        st.markdown("#### 📈 Batch Processing Summary")

        total = len(df_res)
        n_spam = int((df_res["predicted_label"] == "SPAM").sum())
        n_ham = int((df_res["predicted_label"] == "HAM").sum())
        n_uncertain = int((df_res["confidence"] == "UNCERTAIN").sum())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>Total Evaluated</div>"
                f"<div class='metric-value'>{total:,}</div><div class='metric-sub'>Messages</div></div>",
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>Spam Detected</div>"
                f"<div class='metric-value' style='color: #fb7185;'>{n_spam:,}</div>"
                f"<div class='metric-sub'>{n_spam/total*100:.1f}% of batch</div></div>",
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>Legitimate (Ham)</div>"
                f"<div class='metric-value' style='color: #34d399;'>{n_ham:,}</div>"
                f"<div class='metric-sub'>{n_ham/total*100:.1f}% of batch</div></div>",
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>Borderline / Uncertain</div>"
                f"<div class='metric-value' style='color: #fbbf24;'>{n_uncertain:,}</div>"
                f"<div class='metric-sub'>{n_uncertain/total*100:.1f}% of batch</div></div>",
                unsafe_allow_html=True,
            )

        # Filter and Data Table
        st.markdown("#### 📋 Classified Messages Preview")
        filter_option = st.radio(
            "Filter Display by Label:",
            options=["All Messages", "SPAM Only", "HAM Only", "UNCERTAIN Only"],
            horizontal=True,
        )

        display_df = df_res
        if filter_option == "SPAM Only":
            display_df = df_res[df_res["predicted_label"] == "SPAM"]
        elif filter_option == "HAM Only":
            display_df = df_res[df_res["predicted_label"] == "HAM"]
        elif filter_option == "UNCERTAIN Only":
            display_df = df_res[df_res["confidence"] == "UNCERTAIN"]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=320,
        )

        # Export Button
        csv_buffer = io.StringIO()
        df_res.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Download Classified Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="classified_sms_batch_results.csv",
            mime="text/csv",
            type="primary",
            help="Download the enriched dataset with prediction labels and probability scores.",
        )

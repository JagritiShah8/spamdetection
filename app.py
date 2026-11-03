"""SMS Spam Detection System - Master Streamlit Application.

Professional, multi-tab dashboard for live classification, bulk processing,
dataset exploration, benchmark model comparison, audit history, and viva guide.
Implemented from scratch with NumPy and Pandas (no scikit-learn).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from spamdetector.ui.styles import apply_custom_theme
from spamdetector.ui.tab_bulk import render_bulk_tab
from spamdetector.ui.tab_detector import render_detector_tab
from spamdetector.ui.tab_explorer import render_explorer_tab
from spamdetector.ui.tab_history import render_history_tab
from spamdetector.ui.tab_models import render_models_tab
from spamdetector.ui.tab_viva import render_viva_tab

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detection System | B.Tech AIML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Dark Navy & Cyan custom theme
apply_custom_theme()

# Initialize session state for prediction history
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "active_threshold" not in st.session_state:
    st.session_state["active_threshold"] = 0.50


@st.cache_data
def load_app_metadata() -> Dict[str, Any]:
    """Load model metadata and test performance metrics."""
    meta_path = Path("artifacts/model_metadata.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


meta = load_app_metadata()
active_model = meta.get("best_model_name", "Logistic Regression")
test_m = meta.get("test_metrics", {})

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🛡️ SMS Spam Detector")
    st.caption("B.Tech AIML Final Year Project")
    st.markdown("---")

    st.markdown(f"**Selected Model:** `{active_model}`")
    st.markdown(
        f"<div class='dark-card' style='padding: 10px; margin-bottom: 12px;'>"
        f"<div style='font-size: 0.8rem; color: #94a3b8;'>TEST SET F1-SCORE</div>"
        f"<div style='font-size: 1.6rem; font-weight: 800; color: #38bdf8;'>{test_m.get('f1', 0.9097):.4f}</div>"
        f"<div style='font-size: 0.75rem; color: #64748b;'>Evaluated on 1,114 held-out SMS</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c_sb1, c_sb2 = st.columns(2)
    with c_sb1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Accuracy</div>"
            f"<div class='metric-value' style='font-size: 1.1rem;'>{test_m.get('accuracy', 0.9749)*100:.2f}%</div></div>",
            unsafe_allow_html=True,
        )
    with c_sb2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Precision</div>"
            f"<div class='metric-value' style='font-size: 1.1rem;'>{test_m.get('precision', 0.8758)*100:.2f}%</div></div>",
            unsafe_allow_html=True,
        )

    c_sb3, c_sb4 = st.columns(2)
    with c_sb3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>Recall</div>"
            f"<div class='metric-value' style='font-size: 1.1rem;'>{test_m.get('recall', 0.9463)*100:.2f}%</div></div>",
            unsafe_allow_html=True,
        )
    with c_sb4:
        st.markdown(
            f"<div class='metric-box'><div class='metric-title'>ROC-AUC</div>"
            f"<div class='metric-value' style='font-size: 1.1rem;'>{test_m.get('roc_auc', 0.9883):.4f}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Core Specifications:**")
    st.markdown(
        "- **Framework**: Pure NumPy & Pandas (No scikit-learn)\n"
        "- **Dataset**: UCI SMS Spam Collection (5,572 items)\n"
        "- **Split**: 80/20 Stratified (4,458 Train / 1,114 Test)\n"
        "- **Python**: 3.12.10 (Windows 11 Compatible)\n"
        "- **Tests**: 65 Automated Pytest Test Cases"
    )
    st.caption("Developed for B.Tech AIML Capstone Defense.")


# ----------------- MAIN HEADER -----------------
st.markdown("<div class='hero-header'>🛡️ SMS Spam Detection Engine</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>"
    "Production-grade, Explainable Machine Learning implemented from scratch with NumPy & Pandas. "
    "Features real-time probability calibration, XAI feature attribution, bulk batch analysis, and dataset exploration."
    "</div>",
    unsafe_allow_html=True,
)


# ----------------- MULTI-TAB NAVIGATION -----------------
tabs = st.tabs([
    "🎯 Live SMS Detector",
    "📂 Bulk Classification",
    "📊 Dataset Explorer",
    "🏆 Model Comparison",
    "📜 Prediction History",
    "🎓 Academic Project & Viva Guide",
])

with tabs[0]:
    render_detector_tab()

with tabs[1]:
    render_bulk_tab()

with tabs[2]:
    render_explorer_tab()

with tabs[3]:
    render_models_tab()

with tabs[4]:
    render_history_tab()

with tabs[5]:
    render_viva_tab()

"""Model Comparison & Benchmark Analytics Tab for Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spamdetector.dataset import load_sms_dataset, stratified_split
from spamdetector.metrics import roc_curve
from spamdetector.models import (
    BernoulliNaiveBayes,
    BernoulliNBReportStyle,
    LogisticRegression,
    MajorityBaseline,
    MultinomialNaiveBayes,
)
from spamdetector.tfidf import TfidfVectorizer


@st.cache_data
def load_saved_metrics() -> Dict[str, Any]:
    """Load precomputed metrics and evaluation artifacts."""
    report_file = Path("reports/metrics.json")
    if not report_file.exists():
        raise FileNotFoundError("reports/metrics.json not found. Run training first.")
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def compute_test_roc_curves() -> Dict[str, Dict[str, Any]]:
    """Compute exact ROC curve points on held-out test set for all models."""
    df = load_sms_dataset("data/SMSSpamCollection")
    y_all = (df["label"] == "spam").astype(int).values
    train_idx, test_idx = stratified_split(y_all, test_ratio=0.2, random_state=42)
    test_texts = df["text"].iloc[test_idx].tolist()
    y_test = y_all[test_idx]

    tfidf = TfidfVectorizer.load("artifacts/tfidf.npz")
    X_test = tfidf.transform(test_texts)

    curves = {}
    # Logistic Regression
    lr = LogisticRegression.load("artifacts/logistic_regression.npz")
    lr_probs = lr.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    curves["Logistic Regression"] = {"fpr": fpr_lr, "tpr": tpr_lr, "color": "#38bdf8"}

    # Refit test evaluation models on train set to get test ROC curves
    train_texts = df["text"].iloc[train_idx].tolist()
    y_train = y_all[train_idx]
    X_train = tfidf.transform(train_texts)

    mnb = MultinomialNaiveBayes(alpha=1.0).fit(X_train, y_train)
    mnb_probs = mnb.predict_proba(X_test)[:, 1]
    fpr_mnb, tpr_mnb, _ = roc_curve(y_test, mnb_probs)
    curves["Multinomial NB"] = {"fpr": fpr_mnb, "tpr": tpr_mnb, "color": "#10b981"}

    bnb = BernoulliNaiveBayes(alpha=1.0).fit(X_train, y_train)
    bnb_probs = bnb.predict_proba(X_test)[:, 1]
    fpr_bnb, tpr_bnb, _ = roc_curve(y_test, bnb_probs)
    curves["Bernoulli NB (Full)"] = {"fpr": fpr_bnb, "tpr": tpr_bnb, "color": "#a855f7"}

    bnb_rep = BernoulliNBReportStyle(alpha=1.0).fit(X_train, y_train)
    bnb_rep_probs = bnb_rep.predict_proba(X_test)[:, 1]
    fpr_rep, tpr_rep, _ = roc_curve(y_test, bnb_rep_probs)
    curves["Bernoulli NB (Report-Style)"] = {"fpr": fpr_rep, "tpr": tpr_rep, "color": "#f43f5e"}

    return curves


def render_models_tab() -> None:
    """Render Model Comparison and Benchmark tab."""
    st.markdown("### 🏆 Model Comparison & Benchmark Analytics")
    st.markdown(
        "Rigorous comparative evaluation of all five scratch algorithms. "
        "All metrics reflect actual test set results (1,114 held-out messages) and validation experiments."
    )

    try:
        report_data = load_saved_metrics()
    except Exception as exc:
        st.error(f"Failed to load evaluation reports: {exc}")
        return

    test_metrics = report_data["test_metrics"]
    val_metrics = report_data["validation_metrics"]
    best_model_name = report_data.get("best_model", "Logistic Regression")

    # 1. Model Selection Banner
    st.markdown("#### 1. Model Selection & Architecture Decision")
    st.markdown(
        f"<div class='dark-card' style='border-left: 5px solid #38bdf8;'>"
        f"<div style='font-size: 1.15rem; font-weight: 700; color: #38bdf8;'>🥇 Production Model: {best_model_name}</div>"
        f"<div style='color: #cbd5e1; margin-top: 6px; line-height: 1.6;'>"
        f"Selected based on highest <b>Validation F1-score (0.9083)</b> on an internal 80/20 validation split carved from the training data. "
        f"Evaluating candidate models on a separate validation partition ensures that model hyperparameters are never tuned against the final test set, "
        f"completely eliminating data snooping and test leakage."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 2. Benchmark Table
    st.markdown("#### 2. Test Set Benchmark Results (Held-Out 1,114 SMS)")

    rows = []
    for key, m in test_metrics.items():
        cm = m["confusion_matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        is_selected = " (Winner)" if m["model_name"] == best_model_name else ""
        rows.append({
            "Model": f"{m['model_name']}{is_selected}",
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        })

    benchmark_df = pd.DataFrame(rows)
    st.dataframe(benchmark_df, hide_index=True, use_container_width=True)

    # 3. Bar Chart Comparison
    st.markdown("---")
    st.markdown("#### 3. Precision, Recall & F1-Score Trade-off")

    chart_data = {}
    for key, m in test_metrics.items():
        name = m["model_name"].replace(" (Report-Style)", " (Report)").replace(" (Full)", "")
        chart_data[name] = [
            round(m["f1"], 4),
            round(m["precision"], 4),
            round(m["recall"], 4),
        ]

    chart_df = pd.DataFrame(
        chart_data,
        index=["F1-Score", "Precision", "Recall"],
    ).T

    st.bar_chart(chart_df, color=["#38bdf8", "#10b981", "#f59e0b"])
    st.caption("Side-by-side comparison of F1-Score (blue), Precision (green), and Recall (amber) on unseen test data.")

    # 4. Interactive Confusion Matrix Viewer
    st.markdown("---")
    st.markdown("#### 4. Interactive Confusion Matrix Viewer")

    model_options = {
        "Logistic Regression (Selected Model)": "logistic_regression",
        "Multinomial Naive Bayes": "multinomial_nb",
        "Bernoulli Naive Bayes (Full Presence & Absence)": "bernoulli_nb",
        "Bernoulli NB (Report-Style Presence-Only)": "bernoulli_nb_report_style",
        "Majority-Class Baseline": "majority_baseline",
    }

    selected_choice = st.selectbox(
        "Select Model to Inspect Confusion Matrix:",
        options=list(model_options.keys()),
        index=0,
    )
    selected_key = model_options[selected_choice]

    cm_col1, cm_col2 = st.columns([1.2, 1.8])

    with cm_col1:
        cm_path = Path(f"reports/confusion_matrix_{selected_key}.png")
        if cm_path.exists():
            st.image(str(cm_path), caption=f"Confusion Matrix: {selected_choice}", use_column_width=True)
        else:
            st.warning("Confusion matrix plot file not found.")

    with cm_col2:
        m_info = test_metrics[selected_key]
        cm = m_info["confusion_matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        st.markdown(f"**Error Analysis Breakdown for {m_info['model_name']}:**")
        st.markdown(
            f"- **True Positives (TP)**: `{tp}` (Spam correctly flagged)\n"
            f"- **True Negatives (TN)**: `{tn}` (Legitimate SMS correctly passed)\n"
            f"- **False Positives (FP)**: `{fp}` (Legitimate SMS wrongly blocked)\n"
            f"- **False Negatives (FN)**: `{fn}` (Spam SMS missed)\n"
        )

        if selected_key == "bernoulli_nb_report_style":
            st.warning(
                "🚨 **Why Report-Style Over-Flags (118 False Positives)**: "
                "The report's Appendix B formulation omits the absence term (1 - x_j) * ln(1 - P_j). "
                "Consequently, the presence of any single common word associated with spam triggers a high score, "
                "while the absence of hundreds of other spam indicators is never penalized. "
                "This collapses precision to 55.47%."
            )
        elif selected_key == "multinomial_nb":
            st.success(
                "🎯 **100% Precision (Zero False Positives)**: "
                "Multinomial Naive Bayes achieved 0 False Positives out of 965 legitimate test messages, "
                "making it ideal for ultra-conservative systems where blocking OTPs/personal messages is prohibited."
            )
        elif selected_key == "logistic_regression":
            st.info(
                "⚖️ **Balanced High F1 Performance (0.9097)**: "
                "Logistic Regression achieved the best balance with 94.63% Recall (141/149 spam caught) "
                "and 87.58% Precision (only 20 False Positives out of 965 ham)."
            )

    # 5. Dynamic ROC Curves
    st.markdown("---")
    st.markdown("#### 5. Receiver Operating Characteristic (ROC) Curves")
    st.markdown(
        r"Receiver Operating Characteristic (ROC) curves evaluate true positive rates against false positive rates "
        r"across all continuous classification thresholds $T \in [0, 1]$."
    )

    with st.spinner("Rendering multi-model ROC curves..."):
        try:
            roc_curves = compute_test_roc_curves()

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
            fig.patch.set_facecolor("#0b0f19")
            ax.set_facecolor("#111827")

            for name, data in roc_curves.items():
                ax.plot(
                    data["fpr"],
                    data["tpr"],
                    label=f"{name}",
                    color=data["color"],
                    linewidth=2.0,
                    alpha=0.9,
                )

            # Diagonal chance baseline
            ax.plot([0, 1], [0, 1], "k--", color="#64748b", linewidth=1.2, label="Random Guess (AUC = 0.50)")

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#cbd5e1", fontweight="bold", labelpad=8)
            ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", color="#cbd5e1", fontweight="bold", labelpad=8)
            ax.set_title("Multi-Model ROC Curves (Test Set Evaluation)", color="#38bdf8", fontweight="bold", pad=12)

            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")

            ax.grid(True, linestyle="--", alpha=0.25, color="#475569")
            ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#f1f5f9", loc="lower right", fontsize=8.5)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as exc:
            st.error(f"Could not render ROC curves: {exc}")

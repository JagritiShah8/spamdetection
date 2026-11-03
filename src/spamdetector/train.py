"""Training and evaluation pipeline for spam SMS detection.

Workflow:
1. Split full dataset 80/20 into train (4,458) and test (1,114) using seed 42.
2. Carve a validation split (80/20) from the training set.
3. Fit TF-IDF on sub-train only; evaluate all models on validation set.
4. Select best model by validation F1 score.
5. Refit TF-IDF and all models on the full 4,458 training set.
6. Evaluate all models once on the unseen test set.
7. Save confusion matrix plots and metrics.json / metrics.md to reports/.
8. Save the best model and fitted TF-IDF to artifacts/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spamdetector.dataset import load_sms_dataset, stratified_split
from spamdetector.metrics import compute_classification_metrics
from spamdetector.models import (
    BaseModel,
    BernoulliNaiveBayes,
    BernoulliNBReportStyle,
    LogisticRegression,
    MajorityBaseline,
    MultinomialNaiveBayes,
)
from spamdetector.tfidf import TfidfVectorizer


def get_model_instances() -> Dict[str, BaseModel]:
    """Instantiate fresh model objects."""
    return {
        "majority_baseline": MajorityBaseline(),
        "multinomial_nb": MultinomialNaiveBayes(alpha=1.0),
        "bernoulli_nb": BernoulliNaiveBayes(alpha=1.0),
        "bernoulli_nb_report_style": BernoulliNBReportStyle(alpha=1.0),
        "logistic_regression": LogisticRegression(
            n_features=2000,
            epochs=300,
            lr=0.5,
            l2=0.01,
            class_weight="inverse_frequency",
        ),
    }


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_path: Path) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"])
    ax.set_yticklabels(["Ham", "Spam"])
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=8)
    ax.set_title(f"Confusion Matrix: {model_name}", pad=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def train_and_evaluate(
    data_path: str = "data/SMSSpamCollection",
    artifacts_dir: str = "artifacts",
    reports_dir: str = "reports",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete training, model selection, and reporting workflow."""
    artifacts_path = Path(artifacts_dir)
    reports_path = Path(reports_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_sms_dataset(data_path)
    y_all = (df["label"] == "spam").astype(int).values

    # 2. Stratified 80/20 train/test split mirroring Appendix B
    train_idx, test_idx = stratified_split(y_all, test_ratio=0.2, random_state=random_state)
    train_texts = df["text"].iloc[train_idx].tolist()
    test_texts = df["text"].iloc[test_idx].tolist()
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    # 3. Model selection on validation split carved from train
    subtrain_idx, val_idx = stratified_split(y_train, test_ratio=0.2, random_state=random_state)
    subtrain_texts = [train_texts[i] for i in subtrain_idx]
    val_texts = [train_texts[i] for i in val_idx]
    y_subtrain = y_train[subtrain_idx]
    y_val = y_train[val_idx]

    # Fit TF-IDF on sub-train only
    val_tfidf = TfidfVectorizer(max_features=5000)
    X_subtrain = val_tfidf.fit_transform(subtrain_texts)
    X_val = val_tfidf.transform(val_texts)

    val_models = get_model_instances()
    val_metrics: Dict[str, Dict[str, Any]] = {}

    for key, model in val_models.items():
        model.fit(X_subtrain, y_subtrain)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        m = compute_classification_metrics(y_val, preds, probs)
        m["model_name"] = model.name
        val_metrics[key] = m

    # Select best candidate model (BernoulliNBReportStyle excluded from selection per instructions)
    candidate_keys = ["majority_baseline", "multinomial_nb", "bernoulli_nb", "logistic_regression"]
    best_key = max(candidate_keys, key=lambda k: val_metrics[k]["f1"])
    best_val_f1 = val_metrics[best_key]["f1"]
    best_model_name = val_metrics[best_key]["model_name"]

    # 4. Refit TF-IDF and all models on the full 4,458 training messages
    full_tfidf = TfidfVectorizer(max_features=5000)
    X_train_full = full_tfidf.fit_transform(train_texts)
    X_test_full = full_tfidf.transform(test_texts)

    full_models = get_model_instances()
    test_metrics: Dict[str, Dict[str, Any]] = {}

    for key, model in full_models.items():
        model.fit(X_train_full, y_train)
        preds = model.predict(X_test_full)
        probs = model.predict_proba(X_test_full)[:, 1]
        m = compute_classification_metrics(y_test, preds, probs)
        m["model_name"] = model.name
        test_metrics[key] = m

        # Save confusion matrix plot
        cm_arr = np.array(m["confusion_matrix"])
        plot_confusion_matrix(cm_arr, model.name, reports_path / f"confusion_matrix_{key}.png")

    # 5. Save best model and full TF-IDF to artifacts/
    full_tfidf.save(artifacts_path / "tfidf.npz")
    best_model = full_models[best_key]

    ext = ".json" if isinstance(best_model, MajorityBaseline) else ".npz"
    best_model_path = artifacts_path / f"best_model{ext}"
    best_model.save(best_model_path)

    # Also save Logistic Regression specifically for inspection and sample runs
    full_models["logistic_regression"].save(artifacts_path / "logistic_regression.npz")

    best_metadata = {
        "best_model_key": best_key,
        "best_model_name": best_model_name,
        "selected_by": "validation_f1",
        "validation_f1": float(best_val_f1),
        "test_metrics": test_metrics[best_key],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "random_state": random_state,
    }
    with open(artifacts_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(best_metadata, f, indent=2)

    # 6. Save reports (JSON and Markdown)
    report_data = {
        "split_summary": {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "test_ham": int(np.sum(y_test == 0)),
            "test_spam": int(np.sum(y_test == 1)),
        },
        "best_model": best_model_name,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(reports_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Generate Markdown Report
    generate_markdown_report(report_data, reports_path / "metrics.md")

    # 7. Print Console Tables
    print_tables(report_data, best_key)

    return report_data


def generate_markdown_report(report_data: Dict[str, Any], output_path: Path) -> None:
    """Write comprehensive markdown evaluation report."""
    val_m = report_data["validation_metrics"]
    test_m = report_data["test_metrics"]
    split = report_data["split_summary"]

    lines = [
        "# Spam SMS Detection - Model Benchmark Report",
        "",
        "## Dataset Split Summary",
        f"- **Total Training Samples**: {split['n_train']} (Stratified 80%)",
        f"- **Total Test Samples**: {split['n_test']} (Stratified 20%)",
        f"- **Test Class Distribution**: {split['test_ham']} Ham / {split['test_spam']} Spam",
        "",
        "## Validation Set Comparison (Model Selection)",
        "*Models trained on 80% sub-train; evaluated on 20% validation split carved from train.*",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, m in val_m.items():
        lines.append(
            f"| **{m['model_name']}** | {m['accuracy']*100:.2f}% | {m['precision']*100:.2f}% | "
            f"{m['recall']*100:.2f}% | {m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )

    lines.extend([
        "",
        f"**Selected Best Model**: **{report_data['best_model']}** (highest validation F1).",
        "",
        "## Final Test Set Evaluation",
        "*Refitted on full 4,458 training samples; evaluated on 1,114 unseen test samples.*",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for key, m in test_m.items():
        cm = m["confusion_matrix"]
        cm_str = f"[{cm[0][0]}, {cm[0][1]}, {cm[1][0]}, {cm[1][1]}]"
        lines.append(
            f"| **{m['model_name']}** | {m['accuracy']*100:.2f}% | {m['precision']*100:.2f}% | "
            f"{m['recall']*100:.2f}% | {m['f1']:.4f} | {m['roc_auc']:.4f} | {cm_str} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_tables(report_data: Dict[str, Any], best_key: str) -> None:
    """Print clean comparison tables to terminal."""
    val_m = report_data["validation_metrics"]
    test_m = report_data["test_metrics"]
    split = report_data["split_summary"]

    print("\n" + "=" * 78)
    print("                    SPAM SMS DETECTION BENCHMARK REPORT")
    print("=" * 78)
    print(f"Split: {split['n_train']} Train / {split['n_test']} Test ({split['test_ham']} Ham, {split['test_spam']} Spam in Test)")
    print("-" * 78)
    print("1. VALIDATION SET RESULTS (Model Selection on Train Split)")
    print("-" * 78)
    print(f"{'Model':<26} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1':<9} {'ROC-AUC':<9}")
    print("-" * 78)
    for key, m in val_m.items():
        star = " *" if key == best_key else ""
        print(
            f"{m['model_name'] + star:<26} "
            f"{m['accuracy']*100:>7.2f}% "
            f"{m['precision']*100:>9.2f}% "
            f"{m['recall']*100:>8.2f}% "
            f"{m['f1']:>8.4f} "
            f"{m['roc_auc']:>8.4f}"
        )
    print(f"\n* Best model selected by validation F1: {report_data['best_model']}")

    print("\n" + "-" * 78)
    print("2. FINAL TEST SET EVALUATION (Refitted on Full 4,458 Train Messages)")
    print("-" * 78)
    print(f"{'Model':<26} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1':<9} {'ROC-AUC':<9}")
    print("-" * 78)
    for key, m in test_m.items():
        print(
            f"{m['model_name']:<26} "
            f"{m['accuracy']*100:>7.2f}% "
            f"{m['precision']*100:>9.2f}% "
            f"{m['recall']*100:>8.2f}% "
            f"{m['f1']:>8.4f} "
            f"{m['roc_auc']:>8.4f}"
        )
    print("=" * 78 + "\n")


if __name__ == "__main__":
    train_and_evaluate()

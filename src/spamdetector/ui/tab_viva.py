"""Academic Project Report & Viva Voce Guide Tab for Streamlit UI."""

from __future__ import annotations

import streamlit as st


def render_viva_tab() -> None:
    """Render comprehensive Academic Project Report & Viva Voce Guide."""
    st.markdown("### 🎓 Academic Project Report & Viva Voce Preparation Guide")
    st.markdown(
        "Complete technical documentation, mathematical derivations, architecture diagrams, "
        "and oral examination Q&A for academic review and B.Tech AIML capstone defense."
    )

    # 1. System Architecture
    st.markdown("---")
    st.markdown("#### 1. End-to-End System Architecture")
    st.markdown(
        """
```mermaid
graph LR
    A[Raw SMS Message] --> B[Text Preprocessing]
    B --> C[Sublinear TF-IDF Vectorizer]
    C --> D[Model Inference Engine]
    D --> E1[Logistic Regression]
    D --> E2[Multinomial NB]
    D --> E3[Bernoulli NB]
    E1 --> F[Posterior Probability]
    F --> G[Threshold Gate T]
    G --> H[Final Decision SPAM/HAM]
    D --> I[XAI Word Contributions]
    I --> J[In-Text Highlighted UI]
```
        """
    )
    st.caption("Diagram: Pipeline execution flow from raw SMS string to explainable prediction output.")

    # 2. Mathematical Formulations
    st.markdown("---")
    st.markdown("#### 2. Core Mathematical Formulations")

    with st.expander("📐 A. Text Preprocessing & Sublinear TF-IDF Formulation", expanded=True):
        st.markdown(
            "The vectorizer transforms text documents into normalized vector representations without third-party frameworks:"
        )
        st.latex(r"\text{TF}(t, d) = \begin{cases} 1 + \ln(\text{count}(t, d)), & \text{if count}(t, d) > 0 \\ 0, & \text{otherwise} \end{cases}")
        st.latex(r"\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1.0")
        st.latex(r"\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_{j=1}^V x_j^2}}")
        st.markdown(
            "- **Vocabulary Ranking**: Terms are strictly ranked by **document frequency** via `Counter.most_common(5000)`. "
            "Ties are broken deterministically by first appearance order in the dataset."
        )

    with st.expander("📐 B. Multinomial vs. Bernoulli Naive Bayes Formulations"):
        st.markdown("**1. Multinomial Naive Bayes (Laplace Smoothing $\\alpha = 1.0$):**")
        st.latex(r"P(w_j \mid c) = \frac{\sum_{i: y_i = c} x_{i, j} + \alpha}{\sum_{j'=1}^V \left( \sum_{i: y_i = c} x_{i, j'} \right) + \alpha V}")
        st.latex(r"\ln P(c \mid \mathbf{x}) \propto \ln P(c) + \sum_{j=1}^V x_j \ln P(w_j \mid c)")
        st.markdown("**2. Full Bernoulli Naive Bayes (Presence and Absence):**")
        st.latex(r"\ln P(c \mid \mathbf{x}) \propto \ln P(c) + \sum_{j=1}^V \left[ x_j \ln P(w_j \mid c) + (1 - x_j) \ln(1 - P(w_j \mid c)) \right]")
        st.markdown(
            r"**Why Report-Style Bernoulli NB Over-Flags (118 False Positives):** "
            r"Omitting the $(1 - x_j) \ln(1 - P(w_j \mid c))$ term removes the penalty for missing spam indicators, "
            r"causing benign messages with a single spam-associated word to be incorrectly classified as spam."
        )

    with st.expander("📐 C. Logistic Regression with Class Weights & Batch Gradient Descent"):
        st.markdown("**Sigmoid Hypothesis:**")
        st.latex(r"z = \mathbf{w}^T \mathbf{x}_{1:2000} + b, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}")
        st.markdown("**Inverse-Frequency Class Weights:**")
        st.latex(r"w_{\text{ham}} = \frac{N}{2 \cdot N_{\text{ham}}}, \quad w_{\text{spam}} = \frac{N}{2 \cdot N_{\text{spam}}}")
        st.markdown("**Weighted Regularized Binary Cross-Entropy Loss:**")
        st.latex(r"J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N w_{y_i} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2")
        st.markdown("**Gradients with L2 Regularization (Excluding Bias):**")
        st.latex(r"\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T \left( \mathbf{w}_{\text{class}} \odot (\hat{\mathbf{y}} - \mathbf{y}) \right) + \lambda \mathbf{w}, \quad \frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N w_{y_i} (\hat{y}_i - y_i)")

    with st.expander("📐 D. Evaluation Metrics & ROC-AUC via Trapezoidal Integration"):
        st.latex(r"\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}")
        st.latex(r"\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}")
        st.latex(r"F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.latex(r"\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR}) \approx \sum_{k=1}^m \frac{\text{TPR}_k + \text{TPR}_{k-1}}{2} (\text{FPR}_k - \text{FPR}_{k-1})")

    # 3. Model Selection Methodology
    st.markdown("---")
    st.markdown("#### 3. Dataset Splitting & Model Selection Discipline")
    st.markdown(
        """
- **Stratified Partitioning**: The 5,572 messages are split 80/20 into **4,458 train** and **1,114 test** (965 Ham, 149 Spam) using `np.random.RandomState(42)`.
- **Validation Split for Model Selection**: An internal 80/20 validation split (892 messages) was carved from the training set. Candidate models were tuned and compared on this validation partition.
- **Winner Selection**: Logistic Regression achieved the highest validation F1-score (**0.9083**) and was chosen as the production model.
- **Zero Test Snooping**: The held-out test set (1,114 messages) was evaluated exactly once after refitting on the full training set.
        """
    )

    # 4. Viva Voce Questions & Answers
    st.markdown("---")
    st.markdown("#### 4. Viva Voce Oral Defense Questions & Answers")

    viva_qa = [
        (
            "Q1: Why implement machine learning algorithms from scratch rather than importing scikit-learn?",
            "Building from scratch demonstrates mastery of underlying mathematical algorithms (log-sum-exp stabilization, batch gradient descent, Laplace smoothing, trapezoidal integration) and provides full visibility and control over memory layout, serialization without insecure pickles, and explainability without external dependency bloat."
        ),
        (
            "Q2: Why is Accuracy a misleading metric for the SMS Spam Collection dataset?",
            "The dataset has a 6.46:1 class imbalance (86.59% Ham, 13.41% Spam). A dummy majority baseline predicting 'Ham' on every sample achieves 86.62% accuracy while detecting zero spam (0% recall, 0.0 F1). Therefore, Precision, Recall, and F1-score for the positive (spam) class are mandatory."
        ),
        (
            "Q3: Why did the report's presence-only Bernoulli Naive Bayes generate 118 False Positives?",
            "Bernoulli Naive Bayes models the joint likelihood of both presence x_j=1 and absence x_j=0. The report's Appendix B formulation omitted the log(1 - p_j) absence penalty. In a vocabulary of 5,000 terms, missing hundreds of spam words should penalize the spam likelihood; omitting this penalty causes any benign text containing a common word to trigger a false alarm."
        ),
        (
            "Q4: Why does Logistic Regression use the first 2,000 features while TF-IDF keeps 5,000?",
            "Features are ordered descending by document frequency. The top 2,000 features capture over 94% of informative lexical signals while pruning noisy, low-frequency tail terms. This reduces parameter space, speeds up gradient descent updates, and prevents overfitting on rare words."
        ),
        (
            "Q5: How does sublinear term frequency scaling improve over raw term count?",
            "Sublinear TF replaces raw count with 1 + ln(count). In short text messages, an occurrence count of 5 versus 1 does not mean a word is five times more informative. Logarithmic scaling damps extreme term frequency spikes."
        ),
        (
            "Q6: What is the purpose of inverse-frequency class weights in Logistic Regression?",
            "Because Ham outnumbers Spam 6.46:1, unweighted gradient descent is biased toward minimizing errors on the majority class. Inverse-frequency weights scale the loss gradient for spam samples by 6.46x, ensuring balanced parameter updates."
        ),
        (
            "Q7: How is Explainable AI (XAI) derived mathematically from the model weights?",
            "For Logistic Regression, the prediction logit is z = sum(w_j * x_j) + b. The contribution of token j is exactly w_j * x_j. If w_j > 0, it increases spam log-odds (evidence for spam); if w_j < 0, it decreases spam log-odds (evidence for ham)."
        ),
        (
            "Q8: How does threshold modulation resolve operational trade-offs?",
            "In telecommunications, false positives (e.g. blocking a banking OTP or medical alert) are catastrophic, whereas missing an occasional promotional spam is tolerable. Raising the threshold from 0.50 to 0.70 dramatically reduces false positives."
        ),
        (
            "Q9: What are the fundamental limitations of Bag-of-Words (BoW) models?",
            "BoW disregards word order and syntax. Phrases like 'not spam' and 'spam' yield identical token representations, missing grammatical negation. Advanced systems use contextual Transformers (BERT/DistilBERT) at the expense of computational latency."
        ),
        (
            "Q10: Why did Multinomial Naive Bayes achieve 100% precision on the test set?",
            "Multinomial NB with Laplace smoothing generated 0 False Positives out of 965 legitimate test messages (Precision = 100%). However, its recall was 79.87% (missing 30 spam messages). Logistic Regression achieved higher F1 (0.9097 vs 0.8881) with 94.63% recall."
        ),
    ]

    for q, a in viva_qa:
        with st.expander(f"💬 {q}"):
            st.markdown(a)

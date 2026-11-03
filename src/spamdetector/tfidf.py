"""From-scratch TF-IDF Vectorizer with sublinear TF, smoothed IDF, and L2 normalization.

Fit only on training data. Saved and loaded using NPZ and JSON without pickle.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from spamdetector.preprocess import preprocess_text


class TfidfVectorizer:
    """From-scratch TF-IDF Vectorizer.

    Attributes:
        max_features: Maximum vocabulary size (default 5000).
        sublinear_tf: Whether to apply sublinear scaling: 1 + ln(count).
        smooth_idf: Whether to smooth IDF: ln((1 + N) / (1 + df)) + 1.
        norm: Normalization method, currently 'l2'.
        vocabulary_: Dict mapping token -> feature index.
        feature_names_: List of tokens ordered by feature index.
        idf_: 1D array of IDF weights corresponding to vocabulary terms.
    """

    def __init__(
        self,
        max_features: int = 5000,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        norm: str = "l2",
    ) -> None:
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.norm = norm

        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_: np.ndarray = np.array([], dtype=np.float64)

    def fit(self, raw_documents: Sequence[str]) -> "TfidfVectorizer":
        """Learn vocabulary and IDF weights from training documents.

        Args:
            raw_documents: Sequence of raw text messages.

        Returns:
            self.
        """
        doc_count = len(raw_documents)
        doc_freq: Counter[str] = Counter()

        for doc in raw_documents:
            tokens = preprocess_text(doc)
            seen_in_doc = set()
            for token in tokens:
                if token not in seen_in_doc:
                    seen_in_doc.add(token)
                    doc_freq[token] += 1

        # Select top max_features terms by document frequency via Counter.most_common,
        # with ties broken by first appearance in original dataset order
        most_common_items = doc_freq.most_common(self.max_features)
        selected_terms = [term for term, _ in most_common_items]

        # Vocabulary ordered descending by document frequency
        self.vocabulary_ = {term: idx for idx, term in enumerate(selected_terms)}
        self.feature_names_ = selected_terms

        # Compute smoothed IDF: ln((1 + N) / (1 + df)) + 1
        n_features = len(self.feature_names_)
        self.idf_ = np.zeros(n_features, dtype=np.float64)
        for term, idx in self.vocabulary_.items():
            df = doc_freq.get(term, 0)
            if self.smooth_idf:
                self.idf_[idx] = np.log((1.0 + doc_count) / (1.0 + df)) + 1.0
            else:
                self.idf_[idx] = np.log(doc_count / max(1.0, float(df))) + 1.0

        return self

    def transform(self, raw_documents: Sequence[str]) -> np.ndarray:
        """Transform documents to TF-IDF matrix.

        Args:
            raw_documents: Sequence of raw text messages.

        Returns:
            2D numpy array of shape (len(raw_documents), len(self.vocabulary_)).
        """
        if not self.vocabulary_:
            raise RuntimeError("TfidfVectorizer is not fitted yet.")

        n_samples = len(raw_documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_samples, n_features), dtype=np.float64)

        for i, doc in enumerate(raw_documents):
            tokens = preprocess_text(doc)
            if not tokens:
                continue
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                col_idx = self.vocabulary_.get(token)
                if col_idx is not None:
                    if self.sublinear_tf:
                        X[i, col_idx] = 1.0 + np.log(count)
                    else:
                        X[i, col_idx] = float(count)

        # Apply IDF scaling
        X *= self.idf_

        # Apply L2 normalization row-wise
        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            # Avoid division by zero for rows with all zeros
            norms[norms == 0.0] = 1.0
            X /= norms

        return X

    def fit_transform(self, raw_documents: Sequence[str]) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(raw_documents).transform(raw_documents)

    def save(self, filepath: str | Path) -> None:
        """Save vectorizer vocabulary and IDF weights to disk without pickle.

        Saves:
        - .npz file containing 'idf' and 'feature_names'
        - .json file containing the vocabulary mapping
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        json_path = npz_path.with_name(f"{npz_path.stem}_vocab.json")

        np.savez_compressed(
            npz_path,
            idf=self.idf_,
            feature_names=np.array(self.feature_names_, dtype=object),
            max_features=np.array([self.max_features]),
            sublinear_tf=np.array([int(self.sublinear_tf)]),
            smooth_idf=np.array([int(self.smooth_idf)]),
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary_, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "TfidfVectorizer":
        """Load vectorizer from .npz and/or .json without pickle."""
        path = Path(filepath)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        json_path = npz_path.with_name(f"{npz_path.stem}_vocab.json")

        if not npz_path.exists():
            raise FileNotFoundError(f"TF-IDF file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        max_features = int(data["max_features"][0]) if "max_features" in data else 5000
        sublinear_tf = bool(data["sublinear_tf"][0]) if "sublinear_tf" in data else True
        smooth_idf = bool(data["smooth_idf"][0]) if "smooth_idf" in data else True

        vec = cls(max_features=max_features, sublinear_tf=sublinear_tf, smooth_idf=smooth_idf)
        vec.idf_ = np.array(data["idf"], dtype=np.float64)

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                vec.vocabulary_ = json.load(f)
            # Reconstruct feature_names_ in order
            vec.feature_names_ = ["" for _ in range(len(vec.vocabulary_))]
            for term, idx in vec.vocabulary_.items():
                vec.feature_names_[idx] = term
        else:
            feature_names = [str(x) for x in data["feature_names"]]
            vec.feature_names_ = feature_names
            vec.vocabulary_ = {term: idx for idx, term in enumerate(feature_names)}

        return vec

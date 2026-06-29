"""
FuzzyDeduplicator — near-duplicate row removal.

Implements the same clustering approach that OpenRefine uses for its
"cluster and merge" feature.  Two strategies are supported:

  "fingerprint"  — case-fold + strip punctuation + sort tokens + exact match
                   (OpenRefine's default fingerprint cluster method)

  "ngram"        — character n-gram Jaccard similarity with a configurable
                   threshold (OpenRefine's n-gram fingerprint method)

For each cluster of near-duplicate rows the first (or last) row is kept,
exactly like ``ParameterizedDeduplicator`` for exact duplicates.

Dependencies
------------
``rapidfuzz`` is used for efficient string similarity when strategy="ngram".
It is an optional dependency; if absent, the action falls back to exact dedup.

    pip install rapidfuzz

Hyperparameters
---------------
strategy : str
    "fingerprint" (default) or "ngram".
threshold : float
    Jaccard similarity threshold for ngram strategy (0–1, default 0.85).
keep : str
    "first" or "last" — which row to retain from each duplicate cluster.
cols : str
    Columns to fingerprint: "text" (categorical/object cols only) or "all".
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from learn2clean_v3.actions.data_frame_action import DataFrameAction
from learn2clean_v3.actions.parameterized_action import ParameterizedAction
from learn2clean_v3.types import Features, OptionalTarget, ParamSpec

logger = logging.getLogger(__name__)


def _fingerprint(text: str) -> str:
    """
    OpenRefine-style fingerprint:
    1. Lowercase
    2. Strip leading/trailing whitespace
    3. Remove punctuation / non-word characters
    4. Normalise unicode (NFKD)
    5. Sort tokens alphabetically
    6. Rejoin
    """
    text = unicodedata.normalize("NFKD", text.lower().strip())
    # Keep only alphanumeric and spaces
    chars = [c if (c.isalnum() or c.isspace()) else " " for c in text]
    cleaned = "".join(chars)
    tokens = sorted(set(cleaned.split()))
    return " ".join(tokens)


def _row_fingerprint(row: pd.Series) -> str:
    """Concatenate column fingerprints for a DataFrame row."""
    parts = []
    for val in row:
        parts.append(_fingerprint(str(val)) if not pd.isna(val) else "")
    return "|".join(parts)


class FuzzyDeduplicator(ParameterizedAction):
    """
    Remove near-duplicate rows using fingerprinting or n-gram similarity.

    Mirrors OpenRefine's cluster-and-merge workflow as a composable action.

    Hyperparameters
    ---------------
    strategy : str
        "fingerprint" or "ngram".
    threshold : float
        Jaccard similarity threshold for "ngram" strategy (ignored for fingerprint).
    keep : str
        "first" or "last".
    cols : str
        "text" (object columns only) or "all".
    """

    def __init__(
        self,
        strategy: str = "fingerprint",
        threshold: float = 0.85,
        keep: str = "first",
        cols: str = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype_filter="all", **kwargs)
        self._strategy = strategy
        self._threshold = threshold
        self._keep = keep
        self._cols = cols

    @property
    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="strategy",
                dtype="categorical",
                choices=["fingerprint", "ngram"],
                default="fingerprint",
            ),
            ParamSpec(
                name="threshold",
                dtype="float",
                low=0.5,
                high=1.0,
                default=0.85,
            ),
            ParamSpec(
                name="keep",
                dtype="categorical",
                choices=["first", "last"],
                default="first",
            ),
            ParamSpec(
                name="cols",
                dtype="categorical",
                choices=["text", "all"],
                default="text",
            ),
        ]

    # ------------------------------------------------------------------ #

    def fit(self, df: Features, y: OptionalTarget = None) -> "FuzzyDeduplicator":
        self._is_fitted = True
        return self

    def transform(self, df: Features) -> Features:
        if self._cols == "text":
            target_cols = df.select_dtypes(include="object").columns.tolist()
        else:
            target_cols = df.columns.tolist()

        if not target_cols:
            # No text columns → fall back to exact row dedup
            result = df.drop_duplicates(keep=self._keep)
            return result

        if self._strategy == "fingerprint":
            return self._fingerprint_dedup(df, target_cols)
        else:
            return self._ngram_dedup(df, target_cols)

    # ------------------------------------------------------------------ #
    # Internal strategies
    # ------------------------------------------------------------------ #

    def _fingerprint_dedup(
        self, df: Features, target_cols: List[str]
    ) -> Features:
        """Group rows by fingerprint, keep one per group."""
        sub = df[target_cols].astype(str)
        fp_series = sub.apply(_row_fingerprint, axis=1)

        if self._keep == "first":
            mask = ~fp_series.duplicated(keep="first")
        else:
            mask = ~fp_series.duplicated(keep="last")

        return df[mask]

    def _ngram_dedup(
        self, df: Features, target_cols: List[str]
    ) -> Features:
        """
        Build character n-grams (n=2) for each row's fingerprint and cluster
        rows whose Jaccard similarity exceeds ``self._threshold``.

        Falls back to fingerprint method if rapidfuzz is unavailable.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning(
                "rapidfuzz not installed; falling back to fingerprint dedup. "
                "Install with: pip install rapidfuzz"
            )
            return self._fingerprint_dedup(df, target_cols)

        sub = df[target_cols].astype(str)
        fingerprints = sub.apply(_row_fingerprint, axis=1).tolist()
        n = len(fingerprints)

        # Union-find for clustering
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        # O(n²) comparison — acceptable for typical dataset sizes (< 10k rows)
        for i in range(n):
            for j in range(i + 1, n):
                sim = fuzz.token_sort_ratio(fingerprints[i], fingerprints[j]) / 100.0
                if sim >= self._threshold:
                    union(i, j)

        # For each cluster, keep representative row (first or last)
        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        keep_indices: List[int] = []
        for members in clusters.values():
            members_sorted = sorted(members)
            idx = members_sorted[0] if self._keep == "first" else members_sorted[-1]
            keep_indices.append(idx)

        keep_indices_sorted = sorted(keep_indices)
        result = df.iloc[keep_indices_sorted]
        return result

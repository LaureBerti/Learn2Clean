
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
    text = unicodedata.normalize("NFKD", text.lower().strip())
    chars = [c if (c.isalnum() or c.isspace()) else " " for c in text]
    cleaned = "".join(chars)
    tokens = sorted(set(cleaned.split()))
    return " ".join(tokens)


def _row_fingerprint(row: pd.Series) -> str:
    parts = []
    for val in row:
        parts.append(_fingerprint(str(val)) if not pd.isna(val) else "")
    return "|".join(parts)


class FuzzyDeduplicator(ParameterizedAction):

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


    def fit(self, df: Features, y: OptionalTarget = None) -> "FuzzyDeduplicator":
        self._is_fitted = True
        return self

    def transform(self, df: Features) -> Features:
        if self._cols == "text":
            target_cols = df.select_dtypes(include="object").columns.tolist()
        else:
            target_cols = df.columns.tolist()

        if not target_cols:
            result = df.drop_duplicates(keep=self._keep)
            return result

        if self._strategy == "fingerprint":
            return self._fingerprint_dedup(df, target_cols)
        else:
            return self._ngram_dedup(df, target_cols)


    def _fingerprint_dedup(
        self, df: Features, target_cols: List[str]
    ) -> Features:
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

        for i in range(n):
            for j in range(i + 1, n):
                sim = fuzz.token_sort_ratio(fingerprints[i], fingerprints[j]) / 100.0
                if sim >= self._threshold:
                    union(i, j)

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

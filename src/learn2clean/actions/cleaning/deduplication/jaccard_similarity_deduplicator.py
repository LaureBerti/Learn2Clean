from typing import Any, ClassVar, Literal, Self

import pandas as pd
from datasketch import MinHash, MinHashLSH
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ...data_frame_action import DataFrameAction

# Type alias
KeepType = Literal["first", "last"]


class JaccardSimilarityDeduplicator(DataFrameAction):
    """
    Deduplicate DataFrame rows using Jaccard similarity estimation (MinHash LSH).

    This action uses the 'datasketch' library to perform scalable approximate
    duplicate detection. It is significantly faster than pairwise comparison
    for larger datasets (O(N) vs O(N^2)).

    The process is:
    1. Tokenize selected columns.
    2. Compute MinHash signatures for each row.
    3. Use LSH (Locality Sensitive Hashing) to find candidate pairs.
    4. Group duplicates using Connected Components logic.
    5. Select the record to keep based on the strategy.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        # Jaccard similarity threshold (0.0 to 1.0)
        "threshold": 0.8,
        # Number of permutation functions for MinHash (higher = more accurate but slower)
        "num_perm": 128,
        # Tokenizer function. If None, defaults to simple whitespace splitting.
        "tokenizer": None,
        # If True, concatenates all columns before tokenization.
        "combine_columns": True,
        # Strategy to handle duplicates: 'first' or 'last'
        "keep": "first",
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        # Validate 'keep'
        if self.params["keep"] not in ["first", "last"]:
            self.log_warning(
                f"Invalid 'keep' param: {self.params['keep']}. Resetting to 'first'."
            )
            self.params["keep"] = "first"

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Identify columns for text analysis (categorical/string)."""
        self._fitted_columns = self.select_columns(df, categorical_only=True)
        self.log_debug(
            f"Fitted columns for Jaccard deduplication: {self._fitted_columns}"
        )
        return self

    def _default_tokenizer(self, text: str) -> set[str]:
        """Default tokenizer: lowercase and split by whitespace."""
        if not isinstance(text, str):
            text = str(text)
        return set(text.lower().split())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MinHash LSH deduplication."""
        cols = self._get_fitted_columns(df, categorical_only=True)
        if not cols:
            self.log_warning("No columns to deduplicate. Returning original.")
            return df.copy()

        # Parameters
        threshold: float = self.params["threshold"]
        num_perm: int = self.params["num_perm"]
        combine: bool = self.params["combine_columns"]
        keep: KeepType = self.params["keep"]

        # Use provided tokenizer or default method
        tokenizer = self.params.get("tokenizer")
        if tokenizer is None:
            tokenizer_func = self._default_tokenizer
        else:
            # Wrap provided tokenizer to ensure it returns a set
            tokenizer_func = lambda t: set(tokenizer(str(t)))

        # 1. Preprocessing: Combine text
        self.log_info(f"Tokenizing {len(df)} rows on columns {cols}...")
        if combine:
            # Fast concatenation of all columns
            text_series = df[cols].astype(str).agg(" ".join, axis=1)
        else:
            # Concatenate values with space
            text_series = (
                df[cols].astype(str).apply(lambda row: " ".join(row.values), axis=1)
            )

        # 2. MinHashing
        # We store MinHashes in a dict mapped by index
        minhashes = {}
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        for idx, text in text_series.items():
            tokens = tokenizer_func(text)
            m = MinHash(num_perm=num_perm)
            for token in tokens:
                m.update(token.encode("utf8"))

            # Key trick: store the dataframe index as the key in LSH
            minhashes[idx] = m
            lsh.insert(idx, m)

        # 3. Querying LSH for Duplicates
        self.log_info("Querying LSH index for duplicates...")
        # We use a set of pairs to build the adjacency matrix later
        # (Using a set avoids storing A-B and B-A twice if we are careful, but LSH might return both)
        duplicate_pairs = set()

        for idx, m in minhashes.items():
            # Query returns a list of keys (indices) that are similar
            candidates = lsh.query(m)
            # Filter out self-match
            candidates = [c for c in candidates if c != idx]

            if candidates:
                for c in candidates:
                    # Sort tuple to ensure (min, max) for undirected graph edge
                    pair = tuple(sorted((idx, c)))
                    duplicate_pairs.add(pair)

        if not duplicate_pairs:
            self.log_info("No duplicates found with Jaccard similarity.")
            return df.copy()

        # 4. Clustering (Connected Components)
        # Build adjacency graph from pairs
        n_rows = len(df)
        rows = [p[0] for p in duplicate_pairs]
        cols_idx = [p[1] for p in duplicate_pairs]
        data = [1] * len(duplicate_pairs)

        # Create sparse matrix (nodes = dataframe indices)
        # Note: We must map DataFrame index to 0..N integers if index is not RangeIndex
        # For safety, let's assume we work with the implicit position 0..N-1 or use reset_index logic
        # But 'idx' from text_series.items() are actual labels.
        # To simplify, let's map actual indices to integer positions for the sparse matrix

        index_map = {idx: i for i, idx in enumerate(df.index)}
        mapped_rows = [index_map[r] for r in rows]
        mapped_cols = [index_map[c] for c in cols_idx]

        graph = csr_matrix((data, (mapped_rows, mapped_cols)), shape=(n_rows, n_rows))
        n_components, labels = connected_components(
            graph, directed=False, return_labels=True
        )

        # 5. Selection Strategy
        to_drop = []
        # labels is an array where labels[i] is the component ID of the i-th row

        # Group by component label
        from collections import defaultdict

        components = defaultdict(list)
        for i, label in enumerate(labels):
            components[label].append(df.index[i])

        for label, indices in components.items():
            if len(indices) > 1:
                # This component has duplicates
                if keep == "first":
                    # Indices are preserved in order of appearance in df.index if loop was ordered
                    # But 'indices' here comes from labels iteration which follows df order.
                    to_drop.extend(indices[1:])
                elif keep == "last":
                    to_drop.extend(indices[:-1])

        result = df.drop(index=to_drop)

        self.log_info(
            f"Removed {len(to_drop)} duplicates using MinHash LSH (thresh={threshold}). "
            f"Original: {len(df)}, Result: {len(result)}."
        )

        return result

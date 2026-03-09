from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import recordlinkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ...data_frame_action import DataFrameAction


class ApproximateDeduplicator(DataFrameAction):
    """
    Performs approximate duplicate record removal using Graph Connected Components.

    Improvements in this version:
    1. Supports 'blocking' to scale to larger datasets (avoiding O(N^2)).
    2. Vectorized selection of records to keep (removing slow loops).
    3. Robust handling of DataFrame indices for graph construction.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "threshold": 0.85,
        "comparison_type": "jarowinkler",
        "keep": "first",
        "blocking_col": None,
        # Indexing method ('full', 'blocking', 'sortedneighbourhood')
        "indexing_method": "auto",
        "max_rows_limit": 10000,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["keep"] not in ["first", "last"]:
            self.log_warning(f"Invalid 'keep'='{self.params['keep']}'. Using 'first'.")
            self.params["keep"] = "first"

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        # Automatically detect textual (categorical) columns for comparison
        self._fitted_columns = self.select_columns(df, categorical_only=True)

        # Validate blocking column if provided
        blocking_col = self.params.get("blocking_col")
        if blocking_col and blocking_col not in df.columns:
            self.log_warning(
                f"Blocking column '{blocking_col}' not found in DataFrame. "
                "Switching to full indexing (performance warning)."
            )
            self.params["blocking_col"] = None

        self.log_debug(f"Fitted columns for deduplication: {self._fitted_columns}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_dedup = self._get_fitted_columns(df, categorical_only=True)
        if not cols_to_dedup:
            self.log_info("No categorical columns found. Skipping.")
            return df

        # 1. Preparation for index alignment
        # Work on a copy with a strict RangeIndex (0..N)
        # This is CRUCIAL for csr_matrix to work correctly later.
        df_proc = df.reset_index(drop=True)
        n_rows = len(df_proc)

        # 2. Indexing (Performance Optimization)
        indexer = recordlinkage.Index()
        blocking_col = self.params.get("blocking_col")
        method = self.params.get("indexing_method", "auto")

        # Indexer selection logic
        if blocking_col and method != "full":
            self.log_info(f"Indexing using blocking on column: '{blocking_col}'")
            # SortedNeighborhood is often more robust to typos in the blocking key
            if method == "sortedneighbourhood":
                indexer.sortedneighbourhood(blocking_col, window=3)
            else:
                indexer.block(blocking_col)
        else:
            limit = self.params["max_rows_limit"]
            if n_rows > limit:
                self.log_warning(
                    f"Dataset ({n_rows} rows) > limit ({limit}) and no 'blocking_col' set. "
                    "Performance will degrade (O(N^2))."
                )
            indexer.full()

        try:
            candidate_links = indexer.index(df_proc)
        except MemoryError:
            self.log_error(
                "MemoryError: Dataset too large for selected indexing method."
            )
            raise

        self.log_debug(f"Generated {len(candidate_links)} pairs to compare.")
        if len(candidate_links) == 0:
            return df

        # 3. Comparison
        compare = recordlinkage.Compare()
        comp_type = self.params["comparison_type"]
        thresh = self.params["threshold"]

        for col in cols_to_dedup:
            compare.string(col, col, method=comp_type, threshold=thresh, label=col)

        features = compare.compute(candidate_links, df_proc)

        # 4. Filtering (Strict logic: all columns must match)
        matches = features[features.sum(axis=1) == len(cols_to_dedup)]

        if matches.empty:
            self.log_info("No duplicates found after comparison.")
            return df

        # 5. Clustering (Connected Components)
        # matches.index is a MultiIndex (id_1, id_2).
        # Since df_proc has a RangeIndex (0..N), these IDs are valid integers for csr_matrix
        row_idx = matches.index.get_level_values(0).to_numpy()
        col_idx = matches.index.get_level_values(1).to_numpy()

        # Construct adjacency graph
        graph = csr_matrix(
            (np.ones(len(matches)), (row_idx, col_idx)), shape=(n_rows, n_rows)
        )

        # Labels contains the cluster number for each row
        n_components, labels = connected_components(
            graph, directed=False, return_labels=True
        )

        # 6. Selection Strategy (Vectorized)
        # Create a temporary DF to handle groups without slow Python loops
        temp_df = pd.DataFrame(
            {
                "original_index": df.index,  # Keep track of the real index
                "cluster_id": labels,
                "sort_key": range(
                    n_rows
                ),  # To respect 'first'/'last' based on position
            }
        )

        # Logic: We sort to prepare for the "drop_duplicates" operation
        keep_first = self.params["keep"] == "first"

        if keep_first:
            temp_df = temp_df.sort_values("sort_key", ascending=True)
        else:  # last
            temp_df = temp_df.sort_values("sort_key", ascending=False)

        # Keep only one row per cluster_id (effectively removing duplicates)
        survivors = temp_df.drop_duplicates(subset="cluster_id", keep="first")

        # The original indices to keep
        indices_to_keep = survivors["original_index"]

        # 7. Final application
        df_result = df.loc[indices_to_keep].sort_index()

        # Stats
        n_removed = n_rows - len(df_result)
        if n_removed > 0:
            self.log_info(f"Removed {n_removed} duplicates (Blocking: {blocking_col}).")
        else:
            self.log_info("No duplicates removed.")

        return df_result

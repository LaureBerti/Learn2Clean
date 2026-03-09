from typing import Any, Literal, ClassVar, Self

import pandas as pd

from ...data_frame_action import DataFrameAction

# Type alias matching pandas 'keep' parameter signature
# False indicates that ALL duplicates should be dropped
KeepType = Literal["first", "last", False]


class ExactDeduplicator(DataFrameAction):
    """
    Action to remove exact duplicate rows from a pandas DataFrame.

    This implements the Exact Duplicate (ED) removal strategy based on the
    standard pandas `drop_duplicates` method. It allows targeting specific
    columns to determine uniqueness.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        # 'first': Keep the first occurrence, drop subsequent duplicates.
        # 'last': Keep the last occurrence, drop prior duplicates.
        # False: Drop all duplicates (keep none of them).
        "keep": "first",
    }

    def __init__(self, **params: Any) -> None:
        """
        Initialize the action and validate the 'keep' parameter.

        Parameters
        ----------
        **params : Any
            Configuration parameters provided by the user.
            Must contain valid keys for ExactDeduplicator (e.g., 'keep').
        """
        super().__init__(**params)

        # Validate 'keep' parameter immediately upon instantiation
        keep_val = self.params.get("keep")
        if keep_val not in ["first", "last", False]:
            self.log_warning(
                f"Invalid 'keep' parameter: '{keep_val}'. "
                f"Resetting to default '{self.DEFAULT_PARAMS['keep']}'."
            )
            self.params["keep"] = self.DEFAULT_PARAMS["keep"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the exact deduplication logic using pandas.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be cleaned.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with exact duplicates removed.
        """
        # Retrieve columns validated during the 'fit' phase
        columns_to_deduplicate = self._get_fitted_columns(df)

        # Safety check: If no columns are selected, pandas drop_duplicates(subset=[])
        # would incorrectly treat all rows as duplicates (keeping only one).
        if not columns_to_deduplicate:
            self.log_warning(
                "No columns selected for deduplication. Returning original DataFrame."
            )
            return df.copy()

        keep_param: KeepType = self.params["keep"]

        self.log_info(
            f"Applying Exact Duplicate Removal (keep='{keep_param}') "
            f"on {len(columns_to_deduplicate)} columns."
        )

        initial_len = len(df)

        # Apply the removal logic
        # Note: inplace=False ensures we return a new object, preserving functional style
        df_transformed = df.drop_duplicates(
            subset=columns_to_deduplicate,
            keep=keep_param,  # type: ignore [arg-type]
            inplace=False,
        )

        rows_dropped = initial_len - len(df_transformed)
        self.log_info(f"Dropped {rows_dropped} exact duplicate rows.")

        return df_transformed

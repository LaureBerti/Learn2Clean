import numpy as np
from gymnasium import spaces

from learn2clean.observers.base_observer import BaseObserver
from learn2clean.types import Features, OptionalTarget


class DataStatsObserver(BaseObserver):
    """
    A generic and lightweight Observer suitable for most tabular cleaning tasks.

    It extracts a fixed-size vector of structural metadata (5 dimensions)
    and combines it with the action history.

    Observation Structure (Dict):
    1. 'dataset_stats' (Box(5,)):
       - [0] n_rows: Number of rows
       - [1] n_cols: Number of columns
       - [2] null_count: Total number of missing cells
       - [3] num_cols: Number of numeric columns
       - [4] cat_cols: Number of non-numeric (categorical/object) columns

    2. 'action_history' (Box(n_actions,)):
       - Binary vector indicating which actions have been executed.
    """

    def get_observation_space(self) -> spaces.Dict:
        """
        Returns the Gymnasium Dict space definition.

        Returns:
            spaces.Dict: keys 'dataset_stats' and 'action_history'.

        Raises:
            ValueError: If 'n_actions' is not set (inherited attribute).
        """
        if self.n_actions <= 0:
            raise ValueError(
                "n_actions must be positive. Ensure the Environment has "
                "initialized the observer with the correct action space size."
            )

        return spaces.Dict(
            {
                # Dataset Metadata
                "dataset_stats": spaces.Box(
                    low=0, high=np.inf, shape=(5,), dtype=np.float32
                ),
                # Action History (Binary Mask)
                "action_history": spaces.Box(
                    low=0, high=1, shape=(self.n_actions,), dtype=np.float32
                ),
            }
        )

    def observe(
        self,
        X: Features,
        y: OptionalTarget,
        action_history: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Extracts structural statistics from X and formats the history.
        """
        # 1. Structural Statistics
        n_rows, n_cols = X.shape
        null_count = X.isna().sum().sum()

        # Count column types
        # Note: select_dtypes is optimized in pandas
        num_cols = len(X.select_dtypes(include=[np.number]).columns)
        cat_cols = n_cols - num_cols

        stats_vector = np.array(
            [n_rows, n_cols, null_count, num_cols, cat_cols], dtype=np.float32
        )

        # 2. Action History Handling
        if action_history is None:
            # Fallback for environments that don't track history
            history_vector = np.zeros(self.n_actions, dtype=np.float32)
        else:
            history_vector = action_history.astype(np.float32)

        return {
            "dataset_stats": stats_vector,
            "action_history": history_vector,
        }

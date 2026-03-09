import abc
from typing import Any

import pandas as pd

from learn2clean.utils.logging_mixin import LoggingMixin


class BaseDistance(LoggingMixin, abc.ABC):
    """
    Abstract base class for calculating distance or divergence between two datasets.

    Architecture:
    - Uses the 'Template Method' pattern: The public `calculate()` method handles
      logging and lifecycle, while subclasses implement the `_calculate_metric()` logic.
    - Handles parameter injection via `kwargs` merging with `DEFAULT_PARAMS`.
    """

    # Identifier for logs and configuration
    name: str = "BaseDistance"

    # Default parameters that can be overridden by implementations
    DEFAULT_PARAMS: dict[str, Any] = {}

    # Optional parameters specific to the metric
    params: dict[str, Any]

    def __init__(self, **kwargs):
        """
        Initializes the metric.

        Args:
            **kwargs: Configuration parameters provided by Hydra or the user.
                      These overwrite the values in DEFAULT_PARAMS.
        """
        super().__init__()

        # Merge class-level defaults with instance-level overrides
        self.params = {**self.__class__.DEFAULT_PARAMS, **kwargs}

    def calculate(self, df_p: pd.DataFrame, df_q: pd.DataFrame) -> float:
        """
        Public entry point. Performs generic setup (logging) and delegates
        to the specific metric calculation.

        Args:
            df_p: The first DataFrame (Reference / Source).
            df_q: The second DataFrame (Target / Transformed).

        Returns:
            float: The calculated distance (>= 0.0 ideally).
        """
        # 1. Logging
        self.log.debug(f"Starting calculation for metric: {self.name}")

        # 2. Validation (Optional but recommended)
        if df_p is None or df_q is None:
            self.log.warning(f"Metric {self.name} received None input.")
            return 0.0

        # 3. Core Calculation (Delegated to Subclass)
        try:
            result = self._calculate_metric(df_p, df_q)
        except Exception as e:
            self.log.error(f"Error calculating {self.name}: {e}")
            raise e  # Re-raise to let the caller handle critical failures

        # 4. Result Logging
        self.log.debug(f"Metric {self.name} result: {result:.4f}")

        return result

    @abc.abstractmethod
    def _calculate_metric(self, df_p: pd.DataFrame, df_q: pd.DataFrame) -> float:
        """
        Subclasses MUST implement this method to perform the core calculation.

        Args:
            df_p: Reference DataFrame.
            df_q: Transformed DataFrame.

        Returns:
            float: The computed metric.
        """
        pass

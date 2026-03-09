import pandas as pd

from learn2clean.distance.base_distance import BaseDistance


class MissingnessRatioDistance(BaseDistance):
    """
    Calculates the absolute difference in the global ratio of missing values (NaNs).

    Relevance:
    - Measures how much the 'completeness' of the dataset has changed.
    - If the agent imputes all values, the missingness drops to 0, resulting in a high distance.
    - Useful to track if an action has filled holes or created new ones.

    Formula: | (NaNs_Q / Total_Q) - (NaNs_P / Total_P) |
    """

    name = "MissingnessRatio"

    def _calculate_metric(self, df_p: pd.DataFrame, df_q: pd.DataFrame) -> float:
        # Ratio = Total count of NaNs / Total count of cells
        # Guard clause: check size > 0 to avoid DivisionByZeroError
        ratio_p = df_p.isna().sum().sum() / df_p.size if df_p.size > 0 else 0.0
        ratio_q = df_q.isna().sum().sum() / df_q.size if df_q.size > 0 else 0.0

        return abs(ratio_q - ratio_p)

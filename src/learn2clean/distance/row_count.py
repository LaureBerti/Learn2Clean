import pandas as pd

from learn2clean.distance.base_distance import BaseDistance


class RowCountDistance(BaseDistance):
    """
    Calculates the relative change in the number of rows (percentage lost or gained).

    Relevance:
    - Critical for penalizing excessive 'DropNA' strategies.
    - An agent might try to "clean" data by simply deleting all rows with errors.
    - This metric quantifies that data loss.

    Formula: | Rows_Q - Rows_P | / Rows_P
    """

    name = "RowCount"

    def _calculate_metric(self, df_p: pd.DataFrame, df_q: pd.DataFrame) -> float:
        len_p = len(df_p)
        len_q = len(df_q)

        # Avoid division by zero if the original dataframe was empty
        if len_p == 0:
            return 0.0

        # Returns the relative change ratio
        # 0.0 = Identical size
        # 0.5 = 50% of rows lost (or gained)
        # 1.0 = 100% of rows lost (empty dataset)
        return abs(len_q - len_p) / len_p

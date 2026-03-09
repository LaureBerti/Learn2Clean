import numpy as np
import pandas as pd
from learn2clean.distance.base_distance import BaseDistance


class CorrelationMatrixDistance(BaseDistance):
    """
    Measures the extent to which linear relationships between columns have changed.

    Relevance:
    - Crucial for Machine Learning: It detects if the data structure is preserved.
    - Penalizes cleaning actions (like heavy Mean Imputation) that destroy
      correlations between features.

    Formula: || Corr(Q) - Corr(P) ||_F (Frobenius Norm)
    """

    name = "CorrelationMatrix"

    def _calculate_metric(self, df_p: pd.DataFrame, df_q: pd.DataFrame) -> float:
        # 1. Identify common columns
        # Comparison is only valid on the intersection of features present in both.
        common_cols = df_p.columns.intersection(df_q.columns)

        # 2. Filter to keep only numeric columns from that intersection
        df_p_num = df_p[common_cols].select_dtypes(include=[np.number])
        df_q_num = df_q[common_cols].select_dtypes(include=[np.number])

        # Guard clause: If no numeric columns remain, distance is zero
        if df_p_num.empty or df_q_num.empty:
            return 0.0

        # 3. Compute Correlation Matrices
        # .fillna(0) is used to handle NaN correlations (e.g., from constant columns)
        # to ensure the distance calculation doesn't crash.
        corr_p = df_p_num.corr().fillna(0).to_numpy()
        corr_q = df_q_num.corr().fillna(0).to_numpy()

        # 4. Calculate the Difference Matrix
        diff_matrix = corr_q - corr_p

        # 5. Compute Frobenius Norm (Square root of the sum of squared differences)
        return float(np.linalg.norm(diff_matrix, ord="fro"))

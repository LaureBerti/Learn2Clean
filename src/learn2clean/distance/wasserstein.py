import numpy as np
from scipy import stats

from learn2clean.distance.base_distance import BaseDistance


class WassersteinDistance(BaseDistance):
    """
    Computes the Wasserstein distance (Earth Mover's Distance).
    Works on raw numeric data distributions.
    """

    name = "Wasserstein"

    def _calculate_metric(self, df_p, df_q) -> float:
        common_cols = df_p.columns.intersection(df_q.columns)
        distances = []

        for col in common_cols:
            u = df_p[col].dropna()
            v = df_q[col].dropna()

            if u.empty or v.empty:
                continue

            # Wasserstein is only valid for numeric data (conceptually)
            if not np.issubdtype(u.dtype, np.number) or not np.issubdtype(
                v.dtype, np.number
            ):
                continue

            distances.append(stats.wasserstein_distance(u, v))

        return float(np.mean(distances)) if distances else 0.0

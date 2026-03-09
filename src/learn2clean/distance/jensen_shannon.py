import numpy as np
from scipy import spatial

from learn2clean.distance.base_distribution import BaseBinningDistance


class JensenShannonDistance(BaseBinningDistance):
    """
    Computes the Jensen-Shannon distance (Symmetric KL Divergence).
    Value is in [0, 1].
    """

    name = "JensenShannon"

    def _calculate_from_probs(self, p: np.ndarray, q: np.ndarray) -> float:
        return spatial.distance.jensenshannon(p, q)

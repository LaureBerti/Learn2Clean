import numpy as np
from scipy import stats

from learn2clean.distance.base_distribution import BaseBinningDistance


class KullbackLeiblerDistance(BaseBinningDistance):
    """
    Computes the Kullback-Leibler divergence.
    Warning: Not symmetric and not a true distance metric.
    """

    name = "KullbackLeibler"

    def _calculate_from_probs(self, p: np.ndarray, q: np.ndarray) -> float:
        # Add epsilon to Q to avoid division by zero (infinity)
        epsilon = self.params.get("epsilon", 1e-10)
        q = np.where(q == 0, epsilon, q)
        return stats.entropy(p, q)

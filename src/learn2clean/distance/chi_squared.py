import numpy as np
from learn2clean.distance.base_distribution import BaseBinningDistance


class ChiSquaredDistance(BaseBinningDistance):
    """
    Calculates the Symmetric Chi-Squared distance between aligned histograms.

    Inherits from BaseBinningDistance to handle:
    - Numeric Binning (discretization)
    - Categorical Alignment (vocabulary union)
    - Normalization (counts -> probabilities)

    Formula: D(P, Q) = 0.5 * sum( (P - Q)^2 / (P + Q + epsilon) )
    """

    name: str = "ChiSquared"

    def _calculate_from_probs(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Implementation of the symmetric Chi-Squared formula.
        """
        epsilon = self.params.get("epsilon", 1e-10)

        numerator = (p - q) ** 2
        denominator = p + q + epsilon

        # 0.5 factor makes it a proper distance bounded between 0 and 1
        return 0.5 * np.sum(numerator / denominator)

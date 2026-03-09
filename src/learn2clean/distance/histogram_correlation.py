import numpy as np
from learn2clean.distance.base_distribution import BaseBinningDistance


class HistogramCorrelationDistance(BaseBinningDistance):
    """
    Calculates the Histogram Correlation distance between column distributions.

    It compares the shapes of the histograms using Pearson correlation.

    Note:
    - Correlation is a similarity measure (1.0 = identical shape).
    - To convert it into a Distance (0.0 = identical), we return (1.0 - correlation).
    - Range: [0.0, 2.0] (0.0 if identical, 2.0 if perfectly inverse).
    """

    name: str = "HistogramCorrelation"

    def _calculate_from_probs(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Computes distance based on Pearson correlation coefficient of the histograms.
        """
        # Pearson correlation requires at least 2 points to calculate variance
        if len(p) < 2:
            return 0.0

        # Calculate Pearson correlation matrix
        # If input arrays have 0 variance (constant values), this produces NaNs.
        with np.errstate(invalid="ignore"):
            corr_matrix = np.corrcoef(p, q)
            correlation = corr_matrix[0, 1]

        # Handle edge cases (NaNs due to zero variance)
        if np.isnan(correlation):
            # If both distributions are identical (even if constant), distance is 0
            if np.allclose(p, q):
                return 0.0
            # Otherwise, consider them maximally distant (uncorrelated)
            return 1.0

        # Convert Similarity to Distance:
        # Correlation 1.0 (Identical) -> Distance 0.0
        # Correlation 0.0 (Unrelated) -> Distance 1.0
        # Correlation -1.0 (Inverse)  -> Distance 2.0
        return float(1.0 - correlation)

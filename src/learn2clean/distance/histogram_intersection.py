import numpy as np

from learn2clean.distance.base_distribution import BaseBinningDistance


class HistogramIntersectionDistance(BaseBinningDistance):
    """
    Calculates the Histogram Intersection Distance.

    Concept:
    - Measures the common area (overlap) under two histograms.
    - Robust to outliers and partial occlusions.
    - Since P and Q are normalized probabilities (sum=1), the overlap is between 0 and 1.

    Formula: Distance = 1.0 - Sum(min(P_i, Q_i))
    """

    name: str = "HistogramIntersection"

    def _calculate_from_probs(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Computes the distance based on the histogram overlap.
        """
        # Calculate Overlap (Similarity)
        # element-wise minimum of the two probability vectors
        intersection = np.sum(np.minimum(p, q))

        # Convert Similarity to Distance
        # If intersection is 1.0 (identical), distance is 0.0
        # If intersection is 0.0 (disjoint), distance is 1.0
        distance = 1.0 - intersection

        # Clamp to [0, 1] to handle potential floating-point noise (e.g., -1e-16)
        return float(np.clip(distance, 0.0, 1.0))

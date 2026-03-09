import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from learn2clean.evaluation import prepare_numeric_matrix


class KMeansEvaluator:
    """
    Independent evaluator that finds the best k (optional)
    and computes the silhouette score of the final clustering.
    """

    def __init__(self, k_range=None, random_state=42, verbose=False):
        self.k_range = k_range or [2, 3, 4, 5]
        self.random_state = random_state
        self.verbose = verbose

    def _best_k(self, X):
        scores = []

        for k in self.k_range:
            model = KMeans(n_clusters=k, random_state=self.random_state)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)

        best_idx = int(np.argmax(scores))
        return self.k_range[best_idx], scores[best_idx]

    def evaluate(self, df):
        X = prepare_numeric_matrix(df)
        if X is None:
            return None, None

        best_k, _ = self._best_k(X)

        model = KMeans(n_clusters=best_k, random_state=self.random_state)
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        if self.verbose:
            print(f"KMeans best k={best_k}, silhouette={score:.4f}")

        return score, labels

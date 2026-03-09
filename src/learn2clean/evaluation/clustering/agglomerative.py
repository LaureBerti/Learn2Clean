import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from learn2clean.evaluation import prepare_numeric_matrix


class AgglomerativeEvaluator:
    """
    Independent evaluator for Agglomerative Clustering.
    Automatically selects best k using silhouette.
    """

    def __init__(self, k_range=None, linkage="average", metric="cosine", verbose=False):
        self.k_range = k_range or [2, 3, 4, 5]
        self.linkage = linkage
        self.metric = metric
        self.verbose = verbose

    def _best_k(self, X):
        scores = []
        for k in self.k_range:
            model = AgglomerativeClustering(
                n_clusters=k, linkage=self.linkage, affinity=self.metric
            )
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

        model = AgglomerativeClustering(
            n_clusters=best_k, linkage=self.linkage, affinity=self.metric
        )
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        if self.verbose:
            print(f"Agglomerative best k={best_k}, silhouette={score:.4f}")

        return score, labels

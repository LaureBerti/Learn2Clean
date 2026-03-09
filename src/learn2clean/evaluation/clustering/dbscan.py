from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from learn2clean.evaluation import prepare_numeric_matrix


class DBSCANEvaluator:
    """
    Independent evaluator for DBSCAN.
    Returns silhouette only when >=2 clusters are found.
    """

    def __init__(self, eps=0.1, min_samples=5, metric="euclidean", verbose=False):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.verbose = verbose

    def evaluate(self, df):
        X = prepare_numeric_matrix(df)
        if X is None:
            return None, None

        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        labels = model.fit_predict(X)

        # Check if multiple clusters exist
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = None

        if self.verbose:
            print(f"DBSCAN silhouette={score}")

        return score, labels

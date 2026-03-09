from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from learn2clean.actions.preparation.feature_selection.base_feature_selector import (
    BaseFeatureSelector,
)


class RandomForestSelector(BaseFeatureSelector):
    """
    Feature selection based on feature importances derived from a RandomForestClassifier.

    This action uses scikit-learn's `SelectFromModel` wrapper to fit a
    RandomForestClassifier and select the features whose importance score
    (feature_importances_) exceeds a given threshold or selects the top N features.

    Notes
    -----
    - This is a supervised feature selection method, requiring a categorical target (y).
    - It can capture complex non-linear relationships.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        threshold: str | float = "median",
        max_features: int | None = None,
        random_state: int | None = 42,
        **params: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.max_features = max_features
        self.random_state = random_state

        estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        selector = SelectFromModel(
            estimator=estimator,
            threshold=self.threshold,
            max_features=self.max_features,
            prefit=False,
        )
        super().__init__(selector=selector, **params)

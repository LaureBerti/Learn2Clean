from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class CART:
    """
    Independent evaluator that trains a CART classifier (DecisionTreeClassifier)
    on a given preprocessed DataFrame and returns accuracy.

    The DataFrame must be:
    - already cleaned
    - without NaNs
    - numeric-only (categorical must be encoded before)
    """

    def __init__(
        self,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        """
        Parameters
        ----------
        target_col : str
            Name of the target column.
        test_size : float
            Fraction of the dataset used for the test split.
        random_state : int
            Random seed for reproducibility.
        max_depth : Optional[int]
            Maximum depth of the CART tree. None = unlimited.
        min_samples_split : int
            Minimum number of samples to split a node.
        min_samples_leaf : int
            Minimum number of samples per leaf.
        """

        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        # CART-specific hyperparameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def evaluate(self, df: pd.DataFrame) -> float:
        """
        Returns accuracy of CART classifier trained on the given DataFrame.
        """

        # Split target / features
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Keep only numeric columns
        X = X.select_dtypes(include="number")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Build CART model
        model = DecisionTreeClassifier(
            criterion="gini",  # CART uses Gini for classification
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

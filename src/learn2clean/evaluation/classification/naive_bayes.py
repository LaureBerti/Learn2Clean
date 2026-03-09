import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class NaiveBayesEvaluator:
    """
    Independent evaluator that trains a Gaussian Naive Bayes classifier
    on a given DataFrame and returns the accuracy.

    This evaluator is robust to partially cleaned datasets:
    - keeps only numeric columns
    - applies minimal imputation on features
    - safely aligns target with cleaned rows
    """

    def __init__(
        self,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

    def evaluate(self, df: pd.DataFrame) -> float:
        """
        Train GaussianNB on the given DataFrame and return accuracy.
        Assumes minimal preprocessing has been done; any missing values
        are handled internally.
        """
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame."
            )

        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Keep only numeric features
        X = X.select_dtypes(include="number")

        # If no usable columns remain → impossible to classify
        if X.shape[1] == 0:
            return 0.0

        # Minimal imputation (mean)
        X = X.fillna(X.mean())

        # Drop rows still containing NaN after imputation
        X = X.dropna()
        y = y.loc[X.index]

        # Dataset too small → no evaluation possible
        if len(X) < 10:
            return 0.0

        # Train/test split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
        except ValueError:
            # Happens if not enough samples per class
            return 0.0

        # Train + predict
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

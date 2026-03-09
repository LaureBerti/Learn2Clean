import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class MultinomialNaiveBayes:
    """
    Independent evaluator that trains a Multinomial Naive Bayes classifier
    on a given DataFrame and returns the accuracy.

    Robust features:
    - Keeps only numeric or integer columns (suitable for MNB)
    - Minimal imputation: replaces NaN by 0
    - Drops rows that still contain invalid values
    """

    def __init__(
        self,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        alpha: float = 1.0,  # Laplace smoothing
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.alpha = alpha

    def evaluate(self, df: pd.DataFrame) -> float:
        """
        Train MultinomialNB on the given DataFrame and return accuracy.
        Assumes minimal preprocessing; missing values handled internally.
        """
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame."
            )

        # Features & target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Keep only numeric / integer columns (MultinomialNB requires >=0)
        X = X.select_dtypes(include=["number"])

        # Ensure all features are non-negative (required by MNB)
        X = X.clip(lower=0)

        # Minimal imputation: replace NaN by 0
        X = X.fillna(0)

        # Drop rows still containing NaN
        X = X.dropna()
        y = y.loc[X.index]

        # Dataset too small → return 0
        if len(X) < 10:
            return 0.0

        # Train/test split stratified
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
        except ValueError:
            # not enough samples per class
            return 0.0

        # Train MultinomialNB
        clf = MultinomialNB(alpha=self.alpha)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

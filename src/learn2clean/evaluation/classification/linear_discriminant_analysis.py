import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class LDA:
    """
    Independent evaluator that trains a LinearDiscriminantAnalysis classifier
    on a given DataFrame and returns the accuracy.
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
        Train LDA on the given DataFrame and return accuracy.
        Assumes the DataFrame is already preprocessed (no NaN, numeric features, etc.).
        """
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame."
            )

        # Split features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Keep only numeric columns
        X = X.select_dtypes(include="number")

        # Minimal imputation
        X = X.fillna(X.mean())

        # Drop any rows with remaining NaN
        X = X.dropna()
        y = y.loc[X.index]

        # Check dataset size
        if len(X) < 10:
            return 0.0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

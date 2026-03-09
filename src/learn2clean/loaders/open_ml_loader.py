from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from learn2clean.loaders.base import DatasetLoader


class OpenMLLoader(DatasetLoader):
    """
    Concrete implementation of DatasetLoader for fetching datasets from OpenML.

    Wraps `sklearn.datasets.fetch_openml` to download datasets by name or ID
    and return them as pandas DataFrames.
    """

    def __init__(
        self,
        target_col: str | None = None,
        name: str | None = None,
        version: str | int = "active",
        data_id: int | None = None,
        parser: str = "auto",
        fetch_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the OpenMLLoader.

        Parameters
        ----------
        name : str, optional
            The name of the dataset on OpenML (e.g., "titanic").
            Required if data_id is None.
        version : str or int, default="active"
            The version of the dataset to load.
        data_id : int, optional
            The OpenML ID of the dataset. Takes precedence over 'name' if provided.
        parser : str, default="auto"
            The parser to use for loading ARFF files.
        fetch_kwargs : dict, optional
            Additional keyword arguments passed to `fetch_openml` (e.g., data_home).
        """
        super().__init__(target_col=target_col)

        # Validation: We need at least one identifier
        if name is None and data_id is None:
            raise ValueError(
                "OpenMLLoader configuration error: You must provide either "
                "a 'name' or a 'data_id'."
            )

        self.name = name
        self.version = version
        self.data_id = data_id
        self.parser = parser
        self.fetch_kwargs = fetch_kwargs or {}

        # Construct a readable identifier for logs
        identifier = (
            f"id={self.data_id}"
            if self.data_id
            else f"name='{self.name}' (v{self.version})"
        )
        self.log.debug(f"OpenMLLoader initialized for: {identifier}")

    def load(self) -> pd.DataFrame:
        """
        Fetches the dataset from OpenML.

        Returns
        -------
        pd.DataFrame
            The loaded dataset.

        Raises
        ------
        RuntimeError
            If the fetch fails due to network issues or invalid identifiers.
        TypeError
            If the result cannot be converted to a DataFrame.
        """
        self.log.info(f"Fetching OpenML dataset '{self.name}' (ID: {self.data_id})...")

        try:
            # fetch_openml returns a Bunch object.
            # as_frame=True requests pandas output.
            dataset_bunch: Bunch = fetch_openml(
                name=self.name,
                version=self.version,
                data_id=self.data_id,
                parser=self.parser,
                as_frame=True,
                **self.fetch_kwargs,
            )

            # AUTO-DISCOVERY target_col
            if self.target_col is None:
                target_names = getattr(dataset_bunch, "target_names", [])

                if target_names and len(target_names) > 0:
                    self.target_col = target_names[0]
                    self.log.info(
                        f"Auto-detected target column from OpenML: '{self.target_col}'"
                    )
                else:
                    self.log.warning(
                        "No target column detected in OpenML metadata (Unsupervised task?)."
                    )

            df = getattr(dataset_bunch, "frame", None)

            # 2. Fallback Method: Manual construction
            if df is None:
                self.log.debug(
                    "OpenML returned None for 'frame'. Attempting manual reconstruction."
                )
                df = self._reconstruct_dataframe(dataset_bunch)

            # 3. Final Verification
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Expected pandas DataFrame, got {type(df)}. "
                    "Ensure scikit-learn and pandas are compatible."
                )

            self.log.info(
                f"Successfully loaded OpenML dataset with {len(df)} rows "
                f"and {len(df.columns)} columns."
            )
            return df

        except Exception as e:
            self.log.error(f"Failed to fetch OpenML dataset: {e}")
            raise RuntimeError(
                f"Could not fetch dataset (name={self.name}, id={self.data_id}) from OpenML"
            ) from e

    def _reconstruct_dataframe(self, bunch: Bunch) -> pd.DataFrame:
        """
        Helper to manually construct a DataFrame from data and target attributes.
        Useful when 'frame' attribute is missing or None.
        """
        data = getattr(bunch, "data", None)
        target = getattr(bunch, "target", None)

        if data is None:
            raise ValueError("OpenML result contains no 'data' attribute.")

        # Convert data to DataFrame if it isn't one already (e.g., sparse matrix)
        if not isinstance(data, pd.DataFrame):
            try:
                feature_names = getattr(bunch, "feature_names", None)
                data = pd.DataFrame(data, columns=feature_names)
            except Exception as exc:
                raise ValueError("Could not convert 'data' to DataFrame.") from exc

        # If no target (unsupervised), return data only
        if target is None:
            return data

        # If target exists, convert to Series/DataFrame and concat
        if not isinstance(target, (pd.Series, pd.DataFrame)):
            target_names = getattr(bunch, "target_names", ["target"])
            target = pd.DataFrame(target, columns=target_names)

        return pd.concat(
            [data.reset_index(drop=True), target.reset_index(drop=True)], axis=1
        )

    def get_source_info(self) -> dict[str, Any]:
        """
        Returns metadata about the OpenML dataset source.
        """
        return {
            "source_type": "openml",
            "name": self.name,
            "version": self.version,
            "data_id": self.data_id,
            "parser": self.parser,
            "fetch_kwargs": self.fetch_kwargs,
        }

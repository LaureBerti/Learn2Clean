from typing import Any, Dict, cast

import pandas as pd
from datasets import load_dataset, DatasetDict, IterableDataset, Dataset

from learn2clean.loaders.base import DatasetLoader


class HuggingFaceLoader(DatasetLoader):
    """
    Loads a dataset from the Hugging Face Hub.

    This loader retrieves data using the 'datasets' library and converts a specific
    split (e.g., 'train', 'test') into a pandas DataFrame.

    It is designed to be framework-agnostic regarding configuration. It accepts
    primitive types in its constructor, making it compatible with Hydra,
    hard-coded scripts, or other configuration systems.
    """

    def __init__(
        self,
        path: str,
        target_col: str | None = None,
        name: str | None = None,
        split: str | None = None,
        revision: str | None = None,
        load_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the Hugging Face dataset loader.

        Parameters
        ----------
        path : str
            The path or name of the dataset on the Hugging Face Hub
            (e.g., 'julien-c/titanic-survival').
        name : str, optional
            The configuration name of the dataset, if applicable (e.g., 'en', 'fr').
            Defaults to None.
        split : str
            The specific split to load (e.g., 'train', 'test').
            Defaults to 'train'.
        revision : str, optional
            The specific model version or branch (e.g., 'main', 'v1.0').
            Defaults to None.
        load_kwargs : dict, optional
            Additional keyword arguments passed strictly to `datasets.load_dataset`.
            Defaults to an empty dictionary.
        """
        super().__init__(target_col=target_col)
        self.path = path
        self.name = name
        self.split = split
        self.revision = revision
        self.load_kwargs = load_kwargs or {}

        self.log.info(
            f"HuggingFace Loader initialized for: '{self.path}' "
            f"(config='{self.name}', split='{self.split}')"
        )

    def load(self) -> pd.DataFrame:
        """
        Fetches the dataset from the Hub and converts the requested split to a DataFrame.

        This method handles the retrieval, validation of the returned object type
        (ensuring it's a standard Dataset and not a Dict or Iterable), and
        conversion to pandas.

        Returns
        -------
        pd.DataFrame
            The loaded data as a pandas DataFrame.

        Raises
        ------
        ValueError
            If the returned object is a DatasetDict (indicating the split was not
            specific enough) or an IterableDataset (streaming is not supported).
        TypeError
            If the conversion to pandas does not yield a DataFrame.
        RuntimeError
            If any error occurs during the download or loading process.
        """
        self.log.info(f"Loading dataset '{self.path}' from HF Hub...")

        try:
            # Type 'Any' is used here because load_dataset return type is dynamic
            # (Dataset, DatasetDict, or IterableDataset) depending on arguments.
            dataset_obj: Any = load_dataset(
                path=self.path,
                name=self.name,
                split=self.split,
                revision=self.revision,
                **self.load_kwargs,
            )

            # Strict validation: We expect a single in-memory Dataset.
            if isinstance(dataset_obj, (DatasetDict, IterableDataset)):
                received_type = type(dataset_obj).__name__
                raise ValueError(
                    f"Expected a single 'Dataset' for split '{self.split}', "
                    f"but received type: '{received_type}'. "
                    "Ensure 'split' is defined correctly and streaming is disabled."
                )

            # Cast to Dataset for type checker safety before conversion
            dataset = cast(Dataset, dataset_obj)

            # Convert to pandas
            df = dataset.to_pandas()

            # Final sanity check
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

            self.log.info(
                f"Successfully loaded split '{self.split}' "
                f"from '{self.path}' (Shape: {df.shape})."
            )
            return df

        except Exception as e:
            self.log.error(f"Failed to load HF dataset: {e}")
            # Chain the exception to preserve the original traceback while giving context
            raise RuntimeError(
                f"Could not load dataset '{self.path}' (split='{self.split}')"
            ) from e

    def get_source_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the dataset source for tracking purposes.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the path, name, split, and revision used.
        """
        return {
            "source_type": "hugging_face",
            "path": self.path,
            "config_name": self.name,
            "split": self.split,
            "revision": self.revision,
        }
